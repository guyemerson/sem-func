from math import exp, log
from numpy import array, random, dot, zeros, zeros_like, outer, unravel_index, bool_, empty, histogram, count_nonzero, inf, tril, nan_to_num, tensordot, argpartition, flatnonzero, integer
from numpy.linalg import norm
from scipy.special import expit
from warnings import warn

from utils import make_shared, shared_zeros, is_verb, init_alias, alias_sample, product, sparse_like, index, cosine


class SemFuncModel():
    """
    The core semantic function model, including the background distribution
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise the model
        """
        raise NotImplementedError
    
    def make_shared(self):
        """
        Convert to shared memory
        """
        raise NotImplementedError
    
    def collect(self):
        """
        Package the weights into lists for training setup
        """
        raise NotImplementedError
    
    # Semantic functions
    
    def prob(self, ent, pred):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: a probability
        """
        raise NotImplementedError
    
    def prob_all(self, ent):
        """
        Calculate the probabilities of all predicates being true of an entity
        :param ent: an entity vector
        :return: a probability for each predicate
        """
        raise NotImplementedError
    
    # Sampling
    
    def resample_background(self, out_labels, out_vectors, in_labels, in_vectors):
        """
        Resample a latent entity, given links to other entities
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param in_labels: an iterable of link labels
        :param in_vectors: an iterable of entity vectors
        :return: a sampled entity vector
        """
        # The negative energy of each component depends on the links 
        negenergy = - self.ent_bias * (len(out_labels) + len(in_labels))
        for i, label in enumerate(out_labels):
            # 1st axis of link_wei[label]
            negenergy += dot(self.link_wei[label], out_vectors[i])
        for i, label in enumerate(in_labels):
            # 0th axis of link_wei[label]
            negenergy += dot(in_vectors[i], self.link_wei[label])
        # Expit gives the probability of each component if there are no sparsity constraints
        p = expit(negenergy)
        # Warning! If the negenergy is above 710, expit returns nan
        # Warning! If the negenergy is above e.g. 15 for 50 units, we may get underflow 
        return self.sample_card_restr(p)
        
    def sample_card_restr(self, prob):
        """
        Sample a vector from component probabilities,
        restricting the total cardinality.
        :param prob: the probability of each component being on
        """
        # If components are definitely on or off, the cardinality constraint may break:
        maxed_out = (prob == 1)
        n_max = maxed_out.sum()
        if n_max > self.C:
            warn("{} units with prob 1!".format(n_max))
            # In this case, just pick units at random from these
            inds = random.choice(flatnonzero(maxed_out), size=self.C, replace=False)
            vec = zeros(self.D)
            vec[inds] = 1
            return vec
        
        # Sparsity constraints can be enforced using belief propagation (sum-product algorithm)
        # We introduce intermediate nodes which count how many components have been turned on so far
        # Pass messages up
        messages = self.pass_messages_up(prob)
        # Pass messages down
        return self.pass_messages_down(prob, messages)
    
    def pass_messages_up(self, prob):
        """
        Pass messages for belief propagation from individual components to the cardinality potential
        :param prob: non-sparse probabilities of each component
        :return: cardinality probabilities for successive subsets of components
        """
        intermed = zeros((self.D-1, self.C+1))
        # intermed[i,j] is the probability, ignoring the cardinality potential,
        # that the units from 0 to i have total cardinality j 
        intermed[0,0] = 1-prob[0]
        intermed[0,1] = prob[0]
        # TODO This loop is a bottleneck, because it's done in Python rather than NumPy or C.
        # (We call this function often enough that it matters)
        # It's not easy to vectorise in NumPy, because intermed[i] depends on intermed[i-1]
        for i in range(1,self.D-1):
            intermed[i] = intermed[i-1] * (1 - prob[i])  # unit i is off
            intermed[i,1:] += intermed[i-1,:-1] * prob[i]  # unit i is on

        return intermed
    
    def pass_messages_down(self, prob, intermed):
        """
        Sample a vector, using the cardinality probabilities
        :param prob: non-sparse probabilities of each component
        :param intermed: cardinality probabilities for successive subsets of components
        :return: a sampled vector
        """
        # Fix total number of components, and pass messages down
        vec = empty(self.D, dtype=bool_)  # Output vector
        aux = self.C  # Number of components still to come
        # Iteratively sample
        try:
            for i in range(self.D-1, -1, -1):  # [D-1, D-2, ..., 0] 
                if aux == i+1:  # All remaining components are on
                    vec[:i+1] = 1  # [0, ..., i]
                    break
                elif aux == 0:  # All remaining components are off
                    vec[:i+1] = 0
                    break
                else:
                    # Unnormalised probabilities of being on or off:
                    p = prob[i]
                    ein = p * intermed[i-1][aux-1]
                    aus = (1-p) * intermed[i-1][aux]
                    if ein == 0 and aus == 0:
                        raise Exception('div zero!')  # TODO make this cleaner
                    # Probability of being on:
                    on = ein/(ein+aus)
                    # Random sample:
                    if random.binomial(1, on):
                        # Update vector and count
                        vec[i] = 1
                        aux -= 1
                    else:
                        # Update vector
                        vec[i] = 0
        except Exception as e:  # If too many components have high probs (giving underflow errors), just take the highest
            if e.args[0] != 'div zero!':
                raise e
            if self.verbose:
                print('div zero!')
                print(prob)
                print(intermed[-1].sum())
                print(i, aux)
            vec[:] = 0
            vec[prob.argpartition(-self.C)[-self.C:]] = 1
        return vec
    
    def propose_ent(self, ent):
        """
        Propose a Metropolis-Hastings step, by switching on component on and one off
        :param ent: the current entity vector
        :return: a new entity vector, the component switched off, and the component switched on
        """
        # Pick units to switch off and on
        old_i = flatnonzero(ent)[random.randint(self.C)]
        new_i = flatnonzero(ent - 1)[random.randint(self.D - self.C)]
        # Propose new entity
        new_ent = ent.copy()
        new_ent[old_i] = 0
        new_ent[new_i] = 1
        
        return new_ent, old_i, new_i
    
    def metro_switch(self, ratio):
        """
        Stochastically decide whether or not to make a Metropolis-Hastings step
        :param ratio: the ratio of probabilities
        :return: True or False, whether to switch
        """
        if ratio > 1:
            return True
        else:
            return random.binomial(1, ratio)
    
    def resample_conditional(self, old_ent, pred, out_labels, out_vectors, in_labels, in_vectors, chosen=True):
        """
        Resample a latent entity, given a predicate and links to other entities.
        Uses Metropolis-Hastings, potentially turning one component on and one off.
        :param old_ent: the current entity vector
        :param pred: the predicate for the entity 
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param in_labels: an iterable of link labels
        :param in_vectors: an iterable of entity vectors
        :param chosen: (default True) if we should condition on the pred being chosen, or just being true
        :return: a sampled entity vector (changed in place!)
        """
        # Get a new entity proposal
        new_ent, old_i, new_i = self.propose_ent(old_ent)
        
        # Calculate Metropolis-Hastings ratio
        # First, probability of each predicate being applicable:
        new_prob = self.prob(new_ent, pred)
        old_prob = self.prob(old_ent, pred)
        
        #print(new_prob, old_prob, end=' ')
        
        if old_prob == 0:
            print(pred)
            print(old_ent)
            for wei in self.pred_local_weights:
                print(wei[pred])
            raise Exception('prob 0!')  # TODO deal with this
        
        ratio = new_prob / old_prob
        # TODO Need to deal with one or both being nan...
        
        # Next, background energy of entities:
        negenergy = (self.ent_bias[old_i] - self.ent_bias[new_i]) * (len(out_labels) + len(in_labels))
        for n, label in enumerate(out_labels):
            negenergy += dot(self.link_wei[label, new_i, :], out_vectors[n])
            negenergy -= dot(self.link_wei[label, old_i, :], out_vectors[n])
        for n, label in enumerate(in_labels):
            negenergy += dot(self.link_wei[label, :, new_i], in_vectors[n])
            negenergy -= dot(self.link_wei[label, :, old_i], in_vectors[n])
        
        if chosen:
            # Finally, weighted number of other predicates that are true:
            # (Use an approximation, rather than summing over all predicates...)
            negenergy += 0.5 * (self.av_pred[old_i] - self.av_pred[new_i])  # TODO control extra param
        
        ratio *= exp(negenergy)  # TODO deal with overflow errors
        
        # Change the vector accordingly
        if self.metro_switch(ratio):
            #print('switch')
            old_ent[old_i] = 0
            old_ent[new_i] = 1
        #else:
        #    print('stay')
        
        return old_ent
    
    def calc_av_pred(self):
        """
        Recalculate the average predicate
        (used as an approximation in conditional sampling)
        """
        # TODO make this a shared array
        # (which also requires updating it with a separate process)
        raise NotImplementedError
    
    def propose_pred(self, shape=None):
        """
        Generate a random predicate
        (or an array of predicates, if shape is given)
        """
        return alias_sample(self.freq_U, self.freq_K, shape)
    
    def resample_pred(self, vector, old_pred):
        """
        Resample a predicate from an entity vector,
        using Metropolis-Hastings
        :param vector: the entity vector
        :param old_pred: the current latent predicate
        :return: the resampled predicate
        """
        # Propose new predicate
        new_pred = self.propose_pred()
        # Metropolis-Hastings ratio
        ratio = self.freq[new_pred] * self.prob(vector, new_pred) \
              /(self.freq[old_pred] * self.prob(vector, old_pred))
        # Return corresponding pred
        if self.metro_switch(ratio):
            return new_pred
        else:
            return old_pred
    
    # Gradients
    
    def observe_out_links(self, vector, out_labels, out_vectors, matrices=None, n_in_links=0):
        """
        Calculate link weight gradients for the outgoing links of a node
        (the gradients for incoming links will be found when considering the other node)
        :param vector: an entity vector
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param matrices: (optional) the matrices which gradients should be added to
        :param n_in_links: (default 0) number of incoming links (for bias gradients)
        :return: gradient matrices
        """
        # Initialise matrices if not given
        if matrices is None:
            matrices = [zeros_like(m) for m in self.link_weights]
        # Unpack matrices
        gradient_matrix, bias_grad = matrices
        # Calculate gradient for each link
        for i, label in enumerate(out_labels):
            gradient_matrix[label] += outer(vector, out_vectors[i])
        # Calculate gradient for bias terms
        bias_grad -= vector * (len(out_labels) + n_in_links)  # all biases assumed to be negative
        return matrices
    
    def observe_links(self, vector, out_labels, out_vectors, in_labels, in_vectors, matrices=None, link_counts=None):
        """
        Calculate link weight gradients for all links of a node
        :param vector: an entity vector
        :param out_labels: an iterable of link labels (outgoing)
        :param out_vectors: an iterable of entity vectors (outgoing)
        :param in_labels: an iterable of link labels (incoming)
        :param in_vectors: an iterable of entity vectors (incoming)
        :param matrices: (optional) the matrices which gradients should be added to
        :param link_counts: (optional) the vector of counts which should be added to
        :return: gradient matrices, number of times links observed
        """
        # Initialise matrices if not given
        if matrices is None:
            matrices = [zeros_like(m) for m in self.link_weights]
        if link_counts is None:
            link_counts = zeros(self.L)
        # Unpack matrices
        gradient_matrix, bias_grad = matrices
        # Calculate gradient for each link, and count links
        for i, label in enumerate(out_labels):
            gradient_matrix[label] += outer(vector, out_vectors[i])
            link_counts[label] += 1
        for i, label in enumerate(in_labels):
            gradient_matrix[label] += outer(in_vectors[i], vector)
            link_counts[label] += 1
        # Calculate gradient for bias terms
        bias_grad -= vector * (len(out_labels) + len(in_labels))  # all biases assumed to be negative
        return matrices, link_counts
    
    def observe_pred(self, vector, pred, matrices=None):
        """
        Calculate pred weight gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param matrices: (optional) the weight matrix which gradients should be added to
        :return: gradient matrices
        """
        raise NotImplementedError
    
    def observe_latent(self, vector, pred, neg_preds, out_labels, out_vectors, in_labels, in_vectors, link_matrices=None, pred_matrices=None, link_counts=None):
        """
        Calculate multiple gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param neg_preds: an iterable of predicates
        :param out_labels: an iterable of link labels (outgoing)
        :param out_vectors: an iterable of entity vectors (outgoing)
        :param in_labels: an iterable of link labels (incoming)
        :param in_vectors: an iterable of entity vectors (incoming)
        :param link_matrices: (optional) the matrices which link gradients should be added to
        :param pred_matrices: (optional) the matrices which pred gradients should be added to
        :param link_counts: (optional) the vector of counts which should be added to
        :return: link gradient matrices, pred gradient matrices, number of times links observed
        """
        # Initialise matrices if not given
        if link_matrices is None:
            link_matrices = [zeros_like(m) for m in self.link_weights]
        if pred_matrices is None:
            pred_matrices = [zeros_like(m) for m in self.pred_weights]
        if link_counts is None:
            link_counts = zeros(self.L)
        # Add gradients...
        # ...from links:
        self.observe_links(vector, out_labels, out_vectors, in_labels, in_vectors, link_matrices, link_counts)
        # ...from the pred:
        self.observe_pred(vector, pred, pred_matrices)
        # ...from the negative preds:
        
        # This currently only works if all matrices are SparseRows objects
        # We first add one row for each neg pred, and then make the rows negative
        num_preds = neg_preds.shape[0]
        current_index = pred_matrices[0].next
        for p in neg_preds:
            self.observe_pred(vector, p, pred_matrices)
        for grad in pred_matrices:
            grad.array[current_index:grad.next] /= - num_preds  # negative
        # Return gradient matrices
        return link_matrices, pred_matrices, link_counts
    
    def init_observe_latent_batch(self, batch, neg_preds):
        """
        Initialise gradient matrices for a batch of nodes
        :param batch: an iterable of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param neg_preds: a matrix of negative samples of preds
        :return: link gradient matrices, pred gradient matrices
        """
        # Initialise gradient matrices
        link_grads = [zeros_like(m) for m in self.link_weights]
        total_preds = len(batch) * (neg_preds.shape[1] + 1)
        pred_grads = [sparse_like(m, total_preds) for m in self.pred_local_weights]
        pred_grads += [zeros_like(m) for m in self.pred_global_weights]
        return link_grads, pred_grads
    
    # Testing functions
    
    def background_energy(self, links, ents):
        """
        Calculate the background energy of a DMRS graph with given entity vectors
        :param links: an iterable of links of the form (start, end, label) 
        :param ents: an array of entity vectors, indexed consistently with links
        :return: the energy
        """
        e = 0
        # Add energy from each link
        for l in links:
            # This also allows overloading of pydmrs Link objects
            start, end, label = l[:3]
            e -= dot(dot(self.link_wei[label],
                         ents[end]),
                     ents[start])
        return e
    
    def cosine_of_parameters(self, pred1, pred2):
        """
        Calculate the cosine similarity
        between the weights for a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :return: the distance
        """
        raise NotImplementedError
    
    def cosine_of_samples(self, pred1, pred2, **kwargs):
        """
        Calculate the average cosine similarity
        between sampled entity vectors conditioned on a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :param **kwargs: sampling options
        :return: the cosine distance
        """
        # As dot products are distributive over addition, and all samples have the same length,
        # we can take the dot product of the sums
        mean1 = self.mean_sample_from_pred(pred1, **kwargs)
        mean2 = self.mean_sample_from_pred(pred2, **kwargs)
        return cosine(mean1, mean2)
    
    def implies(self, pred1, pred2, **kwargs):
        """
        Calculate the probability that the truth of pred1 implies the truth of pred2
        :param pred1: a predicate
        :param pred2: a predicate
        :param **kwargs: sampling options
        :return: the probability pred1 implies pred2
        """
        total = 0
        ents = self.sample_from_pred(pred1, chosen=False, **kwargs)
        n = 0
        for v in ents:
            total += self.prob(v, pred2)
            n += 1
        return total/n
    
    def distance_matrix_from_parameters(self):
        """
        Find the distances between predicates, based on their parameters
        :return: distance matrix 
        """
        raise NotImplementedError
    
    def closest_pairs(self, number, metric, **kwargs):
        """
        Find the closest pairs of preds 
        :param number: how many pairs to return
        :param metric: the metric to use to find distances; a function, or one of the strings:
            'samples' - use cosine of samples
            'parameters' - use cosine of parameters
        :kwargs: will be passed to the metric, as appropriate
        :return: the names of the closest pairs of preds
        """
        # Get the distance matrix
        if metric == 'samples':
            matrix = self.distance_matrix_from_samples(**kwargs)
        elif metric == 'parameters':
            matrix = self.distance_matrix_from_parameters()
        else:
            matrix = self.distance_matrix(metric)
        # Ignore the diagonal and below it
        matrix += tril(zeros_like(matrix) + inf)
        # Find the indices of the smallest distances
        flat_indices = matrix.argpartition(number-1, None)[:number]  # None: flatten array
        indices = unravel_index(flat_indices, matrix.shape)  # a tuple of arrays
        values = matrix[indices]
        ind_of_val = values.argsort()[::-1]
        ind_tuples = list(zip(*indices))
        sorted_tuples = [ind_tuples[i] for i in ind_of_val]
        # Return these as preds
        preds = [(self.pred_name[i], self.pred_name[j]) for i,j in sorted_tuples]
        return preds
    
    def dot_product_of_samples(self, pred1, pred2, **kwargs):
        """
        Get the dot products of samples between two predicates
        :param pred1: a predicate index
        :param pred2: another predicate index
        :param **kwargs: sampling options
        """
        pred1_samples = array(list(self.sample_from_pred(pred1, **kwargs)), dtype=float)
        pred2_samples = array(list(self.sample_from_pred(pred2, **kwargs)), dtype=float)
        return tensordot(pred1_samples, pred2_samples, (1,1)) / self.C
    
    def probability_of_match(self, pred1, pred2, threshold, **kwargs):
        """
        Find the probability that entities conditioned on two predicates are very similar
        :param pred1: a predicate index
        :param pred2: another predicate index
        :param threshold: the similarity threshold (from 0 to 1)
        :param **kwargs: sampling options
        """
        dots = self.dot_product_of_samples(pred1, pred2, **kwargs)
        above = (dots > threshold).sum()
        return above / dots.size
    
    def mean_sd_dot(self, pred1, pred2, **kwargs):
        """
        Get the mean and standard deviation of the dot products of samples between two predicates
        :param pred1: a predicate index
        :param pred2: another predicate index
        :param **kwargs: sampling options
        """
        dots = self.dot_product_of_samples(pred1, pred2, **kwargs)
        return dots.mean(), dots.std()
    
    # Generation
    
    def sample_from_pred(self, pred, samples=100, burnin=500, interval=50, chosen=True, init='max'):
        """
        Sample entity vectors conditioned on a predicate
        :param pred: a predicate
        :param samples: (default 100) number of samples to average over
        :param burnin: (default 500) number of samples to skip before averaging starts
        :param interval: (default 50) number of samples to take between those used in the average
        :param chosen: (default True) whether to condition on the pred being chosen or being true
        :return: a generator of entity vectors
        """
        if init == 'old':
            v = self.init_vec_from_pred(pred)  # TODO control high and low limits
        elif init == 'max':
            v = self.max_vec_from_pred(pred)
        else:
            raise ValueError("init parameter must be 'max' or 'old'")
        
        for _ in range(burnin-interval):
            self.resample_conditional(v, pred, (),(),(),(), chosen=chosen)
        for _ in range(samples):
            for _ in range(interval):
                self.resample_conditional(v, pred, (),(),(),(), chosen=chosen)
            yield v.copy()
    
    def sample_from_graph(self, nodes, samples=100, burnin=500, interval=50, chosen=False, init='max', keep=None):
        """
        Sample entity vectors conditioned on a lexicalised graph
        :param nodes: a list of nodes of the form (pred, out_labs, out_ids, in_labs, in_ids), indexed by position in the list
        :param samples: (default 100) number of samples to average over
        :param burnin: (default 500) number of samples to skip before averaging starts
        :param interval: (default 50) number of samples to take between those used in the average
        :param chosen: (default True) whether to condition on the pred being chosen or being true
        :param init: (default max) how to initialise vectors - choices are 'old' (init_vec_from_pred) or 'max' (max_vec_from_pred)
        :param keep: array or list of which nodes' vectors should be returned (default everything)
        :return: a generator of entity vectors (as matrices)
        """
        # Initialise
        N = len(nodes)
        ents = zeros((N, self.D))
        if init == 'old':
            for i in range(N):
                ents[i] = self.init_vec_from_pred(nodes[i][0])
        elif init == 'max':
            for i in range(N):
                ents[i] = self.max_vec_from_pred(nodes[i][0])
        
        # Change indices to vectors
        nodes = [list(n) for n in nodes]
        for n in nodes:
            n[2] = [ents[i] for i in n[2]]
            n[4] = [ents[i] for i in n[4]]
        
        # Sample
        for _ in range(burnin - interval):
            for i, n in enumerate(nodes):
                self.resample_conditional(ents[i], *n, chosen=chosen)
        for _ in range(samples):
            for _ in range(interval):
                for i, n in enumerate(nodes):
                    self.resample_conditional(ents[i], *n, chosen=chosen)
            if keep:
                yield ents[keep].copy()
            else:
                yield ents.copy()
    
    def sample_from_svo(self, subj, verb, obj, **kwargs):
        """
        Sample entity vectors from an SVO triple
        :param subj: subject predicate index
        :param verb: verb predicate index
        :param obj: object predicate index
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [(subj, (),    (),    (0,), (1,)),
                 (verb, (0,1), (0,2), (),   ()  ),
                 (obj,  (),    (),    (1,), (1,))]
        return self.sample_from_graph(nodes, **kwargs)
    
    def sample_from_sv(self, subj, verb, **kwargs):
        """
        Sample entity vectors from an SV pair
        :param subj: subject predicate index
        :param verb: verb predicate index
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [(subj, (),   (),   (0,), (1,)),
                 (verb, (0,), (0,), (),   ()  )]
        return self.sample_from_graph(nodes, **kwargs)
    
    def sample_from_vo(self, verb, obj, **kwargs):
        """
        Sample entity vectors from a VO pair
        :param verb: verb predicate index
        :param obj: object predicate index
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [(verb, (1,), (1,), (),   ()  ),
                 (obj,  (),   (),   (1,), (0,))]
        return self.sample_from_graph(nodes, **kwargs)
    
    def mean_sample_from_pred(self, pred, **kwargs):
        """
        Sample entity vectors conditioned on a predicate, and average them
        :param pred: a predicate
        :param **kwargs: sampling options
        :return: the mean entity vector
        """
        samples = self.sample_from_pred(pred, **kwargs)
        return sum(samples) / len(samples)
    
    def sample_background_graph(self, nodes, samples=100, burnin=500, interval=50, keep=None):
        """
        Sample entity vectors from an unlexicalised graph
        :param nodes: a list of nodes of the form (out_labs, out_ids, in_labs, in_ids), indexed by position in the list
        :param samples: (default 100) number of samples to average over
        :param burnin: (default 500) number of samples to skip before averaging starts
        :param interval: (default 50) number of samples to take between those used in the average
        :param keep: array or list of which nodes' vectors should be returned (default everything)
        :return: a generator of entity vectors (as matrices)
        """
        # Initialise
        N = len(nodes)
        ents = zeros((N, self.D))
        
        # Change indices to vectors
        nodes = [list(n) for n in nodes]
        for n in nodes:
            n[1] = [ents[i] for i in n[1]]
            n[3] = [ents[i] for i in n[3]]
        
        # Sample
        for _ in range(burnin - interval):
            for i, n in enumerate(nodes):
                ents[i] = self.resample_background(*n)
        for _ in range(samples):
            for _ in range(interval):
                for i, n in enumerate(nodes):
                    ents[i] = self.resample_background(*n)
            if keep:
                yield ents[keep].copy()
            else:
                yield ents.copy()
    
    def sample_background_svo(self, **kwargs):
        """
        Sample entity vectors from a background SVO graph
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [((),    (),    (0,), (1,)),
                 ((0,1), (0,2), (),   ()  ),
                 ((),    (),    (1,), (1,))]
        return self.sample_background_graph(nodes, **kwargs)
    
    def sample_background_sv(self, **kwargs):
        """
        Sample entity vectors from a background SV graph
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [((),   (),   (0,), (1,)),
                 ((0,), (0,), (),   ()  )]
        return self.sample_background_graph(nodes, **kwargs)
    
    def sample_background_vo(self, **kwargs):
        """
        Sample entity vectors from a background VO graph
        :return: a generator of entity vectors (as matrices)
        """
        nodes = [((1,), (1,), (),   ()  ),
                 ((),   (),   (1,), (0,))]
        return self.sample_background_graph(nodes, **kwargs)
    
    def pred_dist(self, ent):
        """
        Calculate the probability distribution over predicates, for a given entity vector
        :param ent: entity vector
        :return: probability distribution over predicates
        """
        # Unnormalised probability of generating a predis the probability of truth, multiplied by the frequency 
        unnorm = self.prob_all(ent) * self.freq
        return unnorm / unnorm.sum()
    
    def sample_pred(self, ent):
        """
        Sample a predicate from an entity vector
        :param ent: entity vector
        :return: predicate index
        """
        return random.choice(self.V, p=self.pred_dist(ent))
    
    def generate_from_graph(self, nodes, **kwargs):
        """
        Generate predicates from an unlexicalised graph
        :param nodes: a list of nodes of the form (out_labs, out_ids, in_labs, in_ids), indexed by position in the list
        :return: a generator of predicates (as numpy arrays)
        """
        for ents in self.sample_background_graph(nodes, **kwargs):
            preds = [self.sample_pred(e) for e in ents]
            yield array(preds) 
    
    def generate_svo(self, **kwargs):
        """
        Generate predicates from a background SVO graph
        :return: a generator of predicates (as numpy arrays)
        """
        nodes = [((),    (),    (0,), (1,)),
                 ((0,1), (0,2), (),   ()  ),
                 ((),    (),    (1,), (1,))]
        return self.generate_from_graph(nodes, **kwargs)
    
    def generate_sv(self, **kwargs):
        """
        Generate predicatess from a background SV graph
        :return: a generator of predicates (as numpy arrays)
        """
        nodes = [((),   (),   (0,), (1,)),
                 ((0,), (0,), (),   ()  )]
        return self.generate_from_graph(nodes, **kwargs)
    
    def generate_vo(self, **kwargs):
        """
        Generate predicates from a background VO graph
        :return: a generator of predicates (as numpy arrays)
        """
        nodes = [((1,), (1,), (),   ()  ),
                 ((),   (),   (1,), (0,))]
        return self.generate_from_graph(nodes, **kwargs)
    
    # Util functions
    
    def init_vec_from_pred(self, pred, low=0.01, high=0.8):
        """
        Initialise an entity vector from a pred
        :param pred: a predicate
        :param low: (default 0.01) the minimum non-sparse probability of each component
        :param high: (default 0.8) the maximum non-sparse probability of each component
        :return: the vector
        """
        raise NotImplementedError
    
    def max_vec_from_pred(self, pred):
        """
        Return the most typical entity vector for a pred
        :param pred: a predicate
        :return: the vector
        """
        raise NotImplementedError
    
    def index(self, name):
        """
        Find the index of a predicate
        :param name: predicate string
        :return: predicate index
        """
        return index(self.pred_name, name)
    
    # Summary functions
    
    def get_histogram(self, matrix, bins):
        """
        Get a histogram to summarise the distribution of values in a weight matrix
        :param matrix: the weight matrix to be summarised
        :param bins: the histogram bin edges (0 and inf will be added to this)
        :return: the histogram, as probability mass (not density) in each bin
        """
        bin_edges = [0] + list(bins) + [inf]
        histo_no_zero, _ = histogram(matrix, bin_edges)
        num_zero = matrix.size - count_nonzero(matrix)
        histo_no_zero[0] -= num_zero
        histo = array([num_zero] + list(histo_no_zero)) / matrix.size
        return histo
    
    def get_histogram_bias(self, matrix, bins):
        """
        Get a histogram to summarise the distribution of values in a weight matrix
        :param matrix: the weight matrix to be summarised
        :param bins: the histogram bin edges (0 and inf will be added to this)
        :return: the histogram, as probability mass (not density) in each bin
        """
        bin_edges = [0] + list(bins) + [inf]
        histo, _ = histogram(matrix, bin_edges)
        return histo / matrix.size
    
    def get_all_histograms(self, histogram_bins, bias_histogram_bins):
        """
        Produce histograms of weights
        :param histogram_bins: edges of bins for non-bias weights (0 and inf will be added)
        :param bias_histogram_bins: edges of bins for bias weights (0 and inf will be added)
        :return: non-bias histogram, bias histogram
        """
        # Histogram bins of non-bias weights
        num_bins = len(histogram_bins) + 2
        histo = zeros((len(self.normal_weights) + 1, num_bins))
        histo[0,1:-1] = histogram_bins
        histo[0,0] = 0
        histo[0,-1] = inf
        # Histogram bins of bias weights
        num_bins_bias = len(bias_histogram_bins) + 1
        histo_bias = zeros((len(self.bias_weights) + 1, num_bins_bias))
        histo_bias[0, :-1] = bias_histogram_bins
        histo_bias[0,-1] = inf
        # Get histograms of weights
        for i in range(1, len(histo)):
            histo[i] = self.get_histogram(self.normal_weights[i-1], histo[0,1:-1])
        for i in range(1, len(histo_bias)):
            histo_bias[i] = self.get_histogram_bias(self.bias_weights[i-1], histo_bias[0, :-1])
        
        return histo, histo_bias


class SemFuncModel_IndependentPreds(SemFuncModel):
    """
    SemFuncModel with independent parameters for each predicate
    """
    def __init__(self, preds, links, freq, dims, card, init_bias=0, init_card=None, init_range=0, init_ent_bias=None, init_link_str=0, init_verb_prop=0.5, init_pat_prop=0.6, init_ag_prop=0.6, freq_alpha=0.75, verbose=True):
        """
        Initialise the model
        :param preds: names of predicates
        :param links: names of links
        :param freq: frequency of each predicate
        :param dims: dimension of latent entities
        :param card: cardinality of latent entities
        :param init_bias: (optional) initial bias for calculating semantic function values
        :param init_card: (optional) approximate cardinality for initialising pred weights
        :param init_range: (optional) range for initialising pred weights
        :param init_ent_bias: (optional) initial bias for entity components
        :param init_link_str: (optional) initial bias for links, setting some dimensions for verbs, agents, patients
        :param init_verb_prop: (default 0.5) proportion of dimensions to use for verbs
        :param init_pat_prop: (default 0.6) proportion of noun dimensions to use for patients
        :param init_ag_prop: (default 0.6) proportion of noun dimensions to use for agents
        :param freq_alpha: (default 0.75) exponent to raise frequencies to the power of
        :param verbose: (default True) whether to print messages
        """
        self.verbose = verbose
        if self.verbose:
            print("Initialising model")
        
        # Names for human readability
        self.pred_name = preds
        self.link_name = links
        
        # Fixed parameters
        if len(freq) != len(preds):
            raise ValueError('Number of predicate frequencies must match number of predicates')
        self.freq_alpha = freq_alpha
        self.freq = freq ** freq_alpha
        self.freq /= self.freq.sum()
        self.freq_U, self.freq_K = init_alias(freq)
        
        # Constants
        self.D = dims
        self.V = len(preds)
        self.L = len(links)
        self.C = card
        
        ### Trained weights ###
        
        # Initialise link weights
        self.link_wei = shared_zeros((self.L, self.D, self.D))  # link, from, to
        mid = int(dims * init_verb_prop)  # N/V proportion
        agent_high = mid + int((dims-mid) * init_ag_prop)  # agent proportion
        patient_low = mid + int((dims-mid) * (1-init_pat_prop))  # patient proportion
        self.link_wei[0, :mid, mid:agent_high] = init_link_str  # initial strength
        self.link_wei[1, :mid, patient_low:] = init_link_str
        self.ent_bias = shared_zeros(self.D)
        if init_ent_bias is None:
            init_ent_bias = log(self.D/self.C - 1)  # so that expit(bias) = C/D
        self.ent_bias += init_ent_bias
        
        # Initialise pred weights
        if init_card is None: init_card = dims/2
        self.pred_wei = make_shared(random.uniform(0, init_range, (self.V, self.D)))
        for i,p in enumerate(preds):
            if is_verb(p):
                self.pred_wei[i, mid:] = 0
                self.pred_wei[i, :mid] *= random.binomial(1, 2*init_card / dims, mid)
            else:
                self.pred_wei[i, :mid] = 0
                self.pred_wei[i, mid:] *= random.binomial(1, 2*init_card / dims, mid)
        
        self.pred_bias = shared_zeros((self.V,))
        self.pred_bias[:] = init_bias
        
        # Ignore preds that don't occur
        self.pred_wei[self.freq == 0] = 0
        
        # For sampling:
        self.calc_av_pred()  # average predicate
        
        # Package for training setup
        self.collect()
    
    def make_shared(self):
        """
        Convert to shared memory
        """
        self.link_wei = make_shared(self.link_wei)
        self.ent_bias = make_shared(self.ent_bias)
        self.pred_wei = make_shared(self.pred_wei)
        self.pred_bias = make_shared(self.pred_bias)
        # Update pointers in collected lists of weights
        self.collect()
    
    def collect(self):
        """
        Package the weights into lists for training setup
        """
        self.link_local_weights = [self.link_wei]
        self.link_global_weights = [self.ent_bias]
        self.link_weights = self.link_local_weights + self.link_global_weights
        self.pred_local_weights = [self.pred_wei,
                                   self.pred_bias]
        self.pred_global_weights= []
        self.pred_weights = self.pred_local_weights + self.pred_global_weights
        
        self.bias_weights = [self.ent_bias,
                             self.pred_bias]
        self.normal_weights=[self.link_wei,
                             self.pred_wei]
    
    def prob(self, ent, pred):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: a probability
        """
        return expit(dot(ent, self.pred_wei[pred]) - self.pred_bias[pred])
    
    def prob_all(self, ent):
        """
        Calculate the probabilities of all predicates being true of an entity
        :param ent: an entity vector
        :return: a probability for each predicate
        """
        return expit(dot(self.pred_wei, ent) - self.pred_bias)
    
    def calc_av_pred(self):
        """
        Recalculate the average predicate
        (used as an approximation in conditional sampling)
        """
        # Weighted sum of predicates
        self.av_pred = dot(self.freq, self.pred_wei)
    
    def observe_pred(self, vector, pred, matrices=None):
        """
        Calculate pred weight gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param matrices: (optional) the weight matrix which gradients should be added to
        :return: gradient matrices
        """
        # Initialise matrices if not given
        if matrices is None:
            matrices = [zeros_like(m) for m in self.pred_weights]
        # Unpack matrices
        gradient_matrix, bias_gradient_vector = matrices
        # Calculate gradients
        factor = 1 - self.prob(vector, pred)
        
        # we can just set the gradients, since we're storing each separately
        # This currently only works if all matrices are SparseRows objects 
        gradient_matrix[pred] = vector * factor
        bias_gradient_vector[pred] = - factor  # all biases assumed to be negative
        return matrices
    
    def cosine_of_parameters(self, pred1, pred2):
        """
        Calculate the cosine distance (1 - normalised dot product)
        between the weights for a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :return: the distance
        """
        return cosine(self.pred_wei[pred1],
                      self.pred_wei[pred2])
    
    def init_vec_from_pred(self, pred, low=0.01, high=0.8):
        """
        Initialise an entity vector from a pred
        :param pred: a predicate
        :param low: (default 0.01) the minimum non-sparse probability of each component
        :param high: (default 0.8) the maximum non-sparse probability of each component
        :return: the vector
        """
        prob = self.pred_wei[pred].clip(low, high)  # Use ent bias?  # Take expit?
        return self.sample_card_restr(prob)
    
    def max_vec_from_pred(self, pred):
        """
        Return the most typical entity vector for a pred
        :param pred: a predicate
        :return: the vector
        """
        vec = zeros(self.D)
        vec[argpartition(self.pred_wei[pred], -self.C)[-self.C:]] = 1
        return vec
    
    def closest_preds_by_ind(self, pred, number=50):
        """
        Find the nearest neighbour to a pred
        :param pred: a predicate name
        :param number: how many neighbours to return (default 1)
        :return: the nearest neighbours (closest last)
        """
        # Get the parameters for the predicate
        vec = self.pred_wei[pred]
        if not vec.any():  # if all entries are zero
            return None
        
        # Find the distance to other predicates
        dot_prod = dot(self.pred_wei, vec)
        dist = dot_prod / norm(self.pred_wei, axis=1)
        dist = nan_to_num(dist)
        # The closest pred will have the second largest dot product
        # (Largest is the pred itself)
        return dist.argpartition(tuple(range(-1-number,0)))[-1-number:-1]
        
    
    def closest_preds(self, name, number=50):
        """
        Find the nearest neighbour to a pred
        :param pred: a predicate name
        :param number: how many neighbours to return (default 1)
        :return: names of the nearest neighbours (closest last)
        """
        ind = self.index(name)
        closest_inds = self.closest_preds_by_ind(ind, number)
        if closest_inds is not None:
            return [self.pred_name[i] for i in closest_inds]
        else:
            return None

'''
class SemFuncModel_FactorisedPreds(SemFuncModel):
    """
    SemFuncModel with factorised pred parameters
    Deprecated... global pred weights will not be updated correctly...
    """
    def __init__(self, preds, links, freq, dims, card, embed_dims, init_bias=0, init_card=None, init_range=0):
        """
        Initialise the model
        :param preds: names of predicates
        :param links: names of links
        :param freq: frequency of each predicate, as an int
        :param dims: dimension of latent entities
        :param card: cardinality of latent entities
        :param embed_dims: dimension of pred embeddings
        :param init_bias: (optional) initial bias for calculating semantic function values
        :param init_card: (optional) approximate cardinality for initialising pred weights
        :param init_range: (optional) range for initialising pred weights
        """
        raise NotImplementedError
        
        print("Initialising model")
        
        # Names for human readability
        self.pred_name = preds
        self.link_name = links
        
        # Fixed parameters
        if isinstance(freq, list):
            freq = array(freq)
        self.freq = freq / sum(freq)
        assert len(freq) == len(preds)
        
        # Constants
        self.D = dims
        self.V = len(preds)
        self.L = len(links)
        self.C = card
        self.E = embed_dims
        
        # Trained weights
        self.link_wei = shared_zeros((self.L, self.D, self.D))  # link, from, to
        if init_card is None: init_card = dims
        self.pred_embed = random.uniform(0, init_range, (self.V, self.E)) \
                        * random.binomial(1, init_card / dims, (self.V, self.E))
        self.pred_factor = random.uniform(0, init_range, (self.E, self.D)) \
                         * random.binomial(1, init_card / dims, (self.E, self.D))
        self.pred_bias = shared_zeros((self.V,))
        self.pred_bias[:] = init_bias
        
        # Ignore preds that don't occur
        self.pred_embed[self.freq == 0] = 0
        
        # For sampling:
        self.calc_av_pred()  # average predicate
        self.get_pred_tokens(freq)  # pred tokens, for sampling preds
        
        print("Converting to shared memory")
        # Convert to shared memory
        self.freq = make_shared(self.freq)
        self.link_wei = make_shared(self.link_wei)
        self.pred_embed = make_shared(self.pred_embed)
        self.pred_factor = make_shared(self.pred_factor)
        self.pred_bias = make_shared(self.pred_bias)
        
        # Package for training setup
        self.link_weights = [self.link_wei]
        self.pred_local_weights = [self.pred_embed,
                                   self.pred_bias]
        self.pred_global_weights= [self.pred_factor]
        self.pred_weights = self.pred_local_weights + self.pred_global_weights
        
        self.bias_weights = [self.pred_bias]
        self.normal_weights=[self.link_wei,
                             self.pred_embed,
                             self.pred_factor]
        
        print("Finished initialising model")
    
    def prob(self, ent, pred):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: a probability
        """
        pred_wei = dot(self.pred_embed[pred], self.pred_factor)
        return expit(dot(ent, pred_wei) - self.pred_bias[pred])
    
    def calc_av_pred(self):
        """
        Recalculate the average predicate
        (used as an approximation in conditional sampling)
        """
        # Weighted sum of predicates
        av_embed = dot(self.freq, self.pred_embed)
        self.av_pred = dot(av_embed, self.pred_factor)
    
    def observe_pred(self, vector, pred, matrices=None):
        """
        Calculate pred weight gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param matrices: (optional) the weight matrix which gradients should be added to
        :return: gradient matrices
        """
        # Initialise matrices if not given
        if matrices is None:
            matrices = [zeros_like(m) for m in self.pred_weights]
        # Unpack matrices
        gradient_embed, gradient_bias, gradient_factor = matrices
        # Calculate gradients
        multiplier = 1 - self.prob(vector, pred)
        gradient_embed[pred] += multiplier * dot(self.pred_factor, vector)
        gradient_factor += multiplier * outer(self.pred_embed[pred], vector)
        gradient_bias[pred] += - multiplier  # all biases assumed to be negative
        return matrices
    
    def cosine_of_parameters(self, pred1, pred2):
        """
        Calculate the cosine distance (1 - normalised dot product)
        between the weights for a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :return: the distance
        """
        return cosine(dot(self.pred_embed[pred1], self.pred_factor),
                      dot(self.pred_embed[pred2], self.pred_factor))
    
    def init_vec_from_pred(self, pred, low=0.01, high=0.8):
        """
        Initialise an entity vector from a pred
        :param pred: a predicate
        :param low: (default 0.01) the minimum non-sparse probability of each component
        :param high: (default 0.8) the maximum non-sparse probability of each component
        :return: the vector
        """
        prob = dot(self.pred_embed[pred], self.pred_factor).clip(low, high)
        return self.sample_card_restr(prob)
    
    def distance_matrix_from_parameters(self):
        """
        Find the distances between predicates, based on their parameters
        :return: distance matrix 
        """
        parameters = dot(self.pred_embed, self.pred_factor)
        dot_prod = dot(parameters, parameters.T)
        lengths = norm(parameters, axis=1)
        denominator = outer(lengths, lengths)
        cosine_matrix = dot_prod / denominator
        distance_matrix = 1 - cosine_matrix
        return distance_matrix
    
    def closest_preds(self, preds, number=1):
        """
        Find the nearest neighbour to some preds
        :param preds: an iterable of predicates
        :param number: how many neighbours to return (default 1)
        :return: the nearest neighbours
        """
        parameters = dot(self.pred_embed, self.pred_factor)
        res = []
        for p in preds:
            vec = parameters[p]
            if not vec.any():  # if all entries are zero
                res.append(None)
                continue
            dot_prod = dot(parameters, vec)
            dist = dot_prod / norm(parameters, axis=1)
            dist = nan_to_num(dist)
            # The closest pred will have the second largest dot product
            # (Largest is the pred itself)
            indices = dist.argpartition(tuple(range(-1,-1-number,-1)))[-1-number:-1]
            res.append(indices)
        return res
'''


class MultiPredMixin(SemFuncModel):
    """
    Allow calculations based on multiple preds
    """
    def prob(self, ent, pred):
        """
        Calculate the probability that one or more predicates are true of an entity
        :param ent: entity vector
        :param pred: predicate index, or list of indices
        """
        if isinstance(pred, (int, integer)):
            return super().prob(ent, pred)
        
        # Get parent function outside of comprehension
        single_prob = super().prob
        return product(single_prob(ent, p) for p in pred)
    
    def observe_latent(self, vector, pred, neg_preds, out_labels, out_vectors, in_labels, in_vectors, link_matrices=None, pred_matrices=None, link_counts=None):
        """
        Calculate multiple gradients for a node
        :param vector: an entity vector
        :param pred: a predicate or list of predicates
        :param neg_preds: an iterable of predicates
        :param out_labels: an iterable of link labels (outgoing)
        :param out_vectors: an iterable of entity vectors (outgoing)
        :param in_labels: an iterable of link labels (incoming)
        :param in_vectors: an iterable of entity vectors (incoming)
        :param link_matrices: (optional) the matrices which link gradients should be added to
        :param pred_matrices: (optional) the matrices which pred gradients should be added to
        :param link_counts: (optional) the vector of counts which should be added to
        :return: link gradient matrices, pred gradient matrices, number of times links observed
        """
        if isinstance(pred, (int, integer)):
            return super().observe_latent(vector, pred, neg_preds, out_labels, out_vectors, in_labels, in_vectors, link_matrices, pred_matrices, link_counts)
        
        # Initialise matrices if not given
        if link_matrices is None:
            link_matrices = [zeros_like(m) for m in self.link_weights]
        if pred_matrices is None:
            pred_matrices = [zeros_like(m) for m in self.pred_weights]
        if link_counts is None:
            link_counts = zeros(self.L)
        # Add gradients...
        # ...from links:
        self.observe_links(vector, out_labels, out_vectors, in_labels, in_vectors, link_matrices, link_counts)
        
        # (Iterate through all preds for the node)
        for pos_p in pred:
            # ...from the pred:
            self.observe_pred(vector, pos_p, pred_matrices)
            # ...from the negative preds:
            
            # This currently only works if all matrices are SparseRows objects
            # We first add one row for each neg pred, and then make the rows negative
            num_preds = neg_preds.shape[0]
            current_index = pred_matrices[0].next
            for neg_p in neg_preds:
                self.observe_pred(vector, neg_p, pred_matrices)
            for grad in pred_matrices:
                grad.array[current_index:grad.next] /= - num_preds  # negative
        
        # Return gradient matrices
        return link_matrices, pred_matrices, link_counts
    
    def init_observe_latent_batch(self, batch, neg_preds):
        """
        Initialise gradient matrices for a batch of nodes
        :param batch: an iterable of (nodeid, preds, out_labs, out_ids, in_labs, in_ids) tuples
        :param neg_preds: a matrix of negative samples of preds
        :return: link gradient matrices, pred gradient matrices
        """
        link_grads = [zeros_like(m) for m in self.link_weights]
        total_pos_preds = sum(len(x[1]) for x in batch)  # Number of preds including multipred nodes
        total_preds = total_pos_preds * (neg_preds.shape[1] + 1)
        pred_grads = [sparse_like(m, total_preds) for m in self.pred_local_weights]
        pred_grads += [zeros_like(m) for m in self.pred_global_weights]
        return link_grads, pred_grads
    
    def init_vec_from_pred(self, pred, low=0.01, high=0.8):
        """
        Initialise an entity vector from a pred
        :param pred: a predicate or list or predicates
        :param low: (default 0.01) the minimum non-sparse probability of each component
        :param high: (default 0.8) the maximum non-sparse probability of each component
        :return: the vector
        """
        if isinstance(pred, (int, integer)):
            return super().init_vec_from_pred(pred, low, high)
        
        # Multiply the weights from each pred
        prob = self.pred_wei[pred].prod(0).clip(low, high)  # Use ent bias?  # Take expit?
        return self.sample_card_restr(prob)
    
    def max_vec_from_pred(self, pred):
        """
        Return the most typical entity vector for a pred
        :param pred: a predicate
        :return: the vector
        """
        if isinstance(pred, (int, integer)):
            return super().max_vec_from_pred(pred)
        
        vec = zeros(self.D)
        # Sum the weights from each pred, and find the max vector for that
        sum_pred = self.pred_wei[pred].sum(0)
        vec[sum_pred.argpartition(-self.C)[-self.C:]] = 1
        return vec


class SemFuncModel_MultiIndependentPreds(MultiPredMixin, SemFuncModel_IndependentPreds):
    """
    SemFuncModel with independent parameters for each predicate,
    and allowing calculations based on multiple preds
    """