from math import exp, log
from numpy import array, random, dot, zeros, zeros_like, outer, unravel_index, bool_, empty, histogram, count_nonzero, inf, tril, nan_to_num
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.special import expit

from utils import make_shared, is_verb


class SemFuncModel():
    """
    The core semantic function model, including the background distribution
    """
    def __init__(self, *arg, **kwargs):
        raise NotImplementedError
    
    def get_pred_tokens(self, freq):
        pred_toks = []  # fill with pred tokens, for sampling preds
        for i, f in enumerate(freq):  # The original ints, not the normalised values
            pred_toks.extend([i]*f)
        self.pred_tokens = make_shared(array(pred_toks))
    
    # Semantic functions
    
    def prob(self, ent, pred):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: a probability
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
        negenergy = - self.ent_bias
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
        for p in prob:
            if p == 1:
                print(prob)
                raise Exception("prob 1!")
        
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
        first_message = zeros(self.C+1)
        first_message[0] = 1-prob[0]
        first_message[1] = prob[0]
        intermed = [first_message]
        # Note! This loop is a bottleneck, because it's done in Python rather than NumPy.
        # (We call this function often enough that it matters)
        # It would be faster for this loop to be done in C...
        for p in prob[1:-1]:  # [1, 2, ..., D-2]
            message = self.convolve(intermed[-1], p)
            intermed.append(message)
        return intermed
    
    @staticmethod
    def convolve(prev, p):
        """
        Convolve prev with (p,1-p), truncated to length self.C+1
        (Faster than numpy.convolve for this case)
        :param prev: probabilities [P(0),...,P(C)]
        :param p: probability of the next component
        :return: the next probabilities
        """
        result = prev * (1-p)
        result[1:] += prev[:-1] * p
        return result
    
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
                        raise Exception('div zero!')
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
        # Pick an on and an off unit to switch
        old_jth = random.randint(self.C)
        new_jth = random.randint(self.D-self.C)
        # Find these units
        on = 0
        for i, val in enumerate(ent):
            if val:
                if on == old_jth:
                    old_i = i
                    break
                else:
                    on += 1
        off = 0
        for i, val in enumerate(ent):
            if not val:
                if off == new_jth:
                    new_i = i
                    break
                else:
                    off += 1
        # Propose new entity
        new_ent = array(ent)  # returns a copy
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
        
        if old_prob == 0:
            print(pred)
            print(old_ent)
            for wei in self.pred_local_weights:
                print(wei[pred])
            raise Exception('prob 0!')
        
        ratio = new_prob / old_prob
        
        # Next, background energy of entities:
        negenergy = self.ent_bias[old_i] - self.ent_bias[new_i]
        for n, label in enumerate(out_labels):
            negenergy += dot(self.link_wei[label, new_i, :], out_vectors[n])
            negenergy -= dot(self.link_wei[label, old_i, :], out_vectors[n])
        for n, label in enumerate(in_labels):
            negenergy += dot(self.link_wei[label, :, new_i], in_vectors[n])
            negenergy -= dot(self.link_wei[label, :, old_i], in_vectors[n])
        
        if chosen:
            # Finally, weighted number of other predicates that are true:
            # (Use an approximation, rather than summing over all predicates...)
            negenergy += 0.5 * (self.av_pred[old_i] - self.av_pred[new_i])  #!# extra param
        
        ratio *= exp(negenergy)
        
        # Change the vector accordingly
        if self.metro_switch(ratio):
            old_ent[old_i] = 0
            old_ent[new_i] = 1
        
        return old_ent
    
    def calc_av_pred(self):
        """
        Recalculate the average predicate
        (used as an approximation in conditional sampling)
        """
        raise NotImplementedError
    
    def resample_pred(self, vector, old_pred):
        """
        Resample a predicate from an entity vector,
        using Metropolis-Hastings
        :param vector: the entity vector
        :param old_pred: the current latent predicate
        :return: the resampled predicate
        """
        # Propose new predicate
        new_pred = random.choice(self.pred_tokens)
        # Metropolis-Hastings ratio
        ratio = self.freq[new_pred] * self.prob(vector, new_pred) \
              /(self.freq[old_pred] * self.prob(vector, old_pred))
        # Return corresponding pred
        if self.metro_switch(ratio):
            return new_pred
        else:
            return old_pred
    
    # Gradients
    
    def observe_out_links(self, vector, out_labels, out_vectors, matrices=None):
        """
        Calculate link weight gradients for the outgoing links of a node
        (the gradients for incoming links will be found when considering the other node)
        :param vector: an entity vector
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param matrices: (optional) the matrices which gradients should be added to
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
        bias_grad -= vector  # all biases assumed to be negative
        return matrices
    
    def observe_links(self, vector, out_labels, out_vectors, in_labels, in_vectors, matrices=None, link_counts=None):
        """
        Calculate link weight gradients for the outgoing links of a node
        (the gradients for incoming links will be found when considering the other node)
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
        bias_grad -= vector  # all biases assumed to be negative
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
    
    def observe_latent_out_links(self, vector, pred, neg_preds, out_labels, out_vectors, link_matrices=None, pred_matrices=None):
        """
        Calculate multiple gradients for a node
        (the gradients for incoming links will be found when considering the other node)
        :param vector: an entity vector
        :param pred: a predicate
        :param neg_preds: an iterable of predicates
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param link_matrices: (optional) the matrices which link gradients should be added to
        :param pred_matrices: (optional) the matrices which pred gradients should be added to
        :return: link gradient matrices, pred gradient matrices
        """
        # Initialise matrices if not given
        if link_matrices is None:
            link_matrices = [zeros_like(m) for m in self.link_weights]
        if pred_matrices is None:
            pred_matrices = [zeros_like(m) for m in self.pred_weights]
        # Add gradients...
        # ...from links:
        self.observe_out_links(vector, out_labels, out_vectors, link_matrices)
        # ...from the pred:
        self.observe_pred(vector, pred, pred_matrices)
        # ...from the negative preds:
        num_preds = neg_preds.shape[0]
        neg_pred_matrices = [zeros_like(m) for m in self.pred_weights]
        for p in neg_preds:
            self.observe_pred(vector, p, neg_pred_matrices)
        for i, grad in enumerate(neg_pred_matrices):
            grad /= num_preds
            pred_matrices[i] -= grad
        # Return gradient matrices
        return link_matrices, pred_matrices
    
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
        num_preds = neg_preds.shape[0]
        current_index = pred_matrices[0].next
        for p in neg_preds:
            self.observe_pred(vector, p, pred_matrices)
        for grad in pred_matrices:
            grad.array[current_index:grad.next] /= - num_preds  # negative
        # Return gradient matrices
        return link_matrices, pred_matrices, link_counts
    
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
        Calculate the cosine distance (1 - normalised dot product)
        between the weights for a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :return: the distance
        """
        raise NotImplementedError
    
    def sample_from_pred(self, pred, samples=5, burnin=5, interval=2, chosen=True):
        """
        Sample entity vectors conditioned on a predicate
        :param pred: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :param chosen: (default True) whether to condition on the pred being chosen or being true
        :return: a generator of entity vectors
        """
        v = self.init_vec_from_pred(pred)  #!# Not currently controlling high and low limits
        for _ in range(max(0, burnin-interval)):
            self.resample_conditional(v, pred, (),(),(),(), chosen=chosen)
        for _ in range(samples):
            for _ in range(interval):
                self.resample_conditional(v, pred, (),(),(),(), chosen=chosen)
            yield v.copy()
    
    def mean_sample_from_pred(self, pred, samples=5, burnin=5, interval=2, chosen=True):
        """
        Sample entity vectors conditioned on a predicate, and average them
        :param pred: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :param chosen: (default True) whether to condition on the pred being chosen or being true
        :return: the mean entity vector
        """
        return sum(self.sample_from_pred(pred, samples, burnin, interval, chosen)) / samples
    
    def cosine_of_samples(self, pred1, pred2, samples=5, burnin=5, interval=2, chosen=True):
        """
        Calculate the average cosine distance (1 - normalised dot product)
        between sampled entity vectors conditioned on a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :param chosen: (default True) whether to condition on the pred being chosen or being true
        :return: the cosine distance
        """
        # As dot products are distributive over addition, and all samples have the same length,
        # we can take the dot product of the sums
        mean1 = self.mean_sample_from_pred(pred1, samples, burnin, interval, chosen)
        mean2 = self.mean_sample_from_pred(pred2, samples, burnin, interval, chosen)
        return cosine(mean1, mean2)
    
    def implies(self, pred1, pred2, samples=5, burnin=5, interval=2):
        """
        Calculate the probability that the truth of pred1 implies the truth of pred2
        :param pred1: a predicate
        :param pred2: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :return: the probability pred1 implies pred2
        """
        total = 0
        ents = self.sample_from_pred(pred1, samples=samples, burnin=burnin, interval=interval, chosen=False)
        for v in ents:
            total += self.prob(v, pred2)
        return total/samples
    
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


class SemFuncModel_IndependentPreds(SemFuncModel):
    """
    SemFuncModel with independent parameters for each predicate
    """
    def __init__(self, preds, links, freq, dims, card, init_bias=0, init_card=None, init_range=0):
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
        """
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
        
        # Trained weights
        # Initialise link weights
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        #!# Init settings for links are not yet controllable with parameters
        mid = int(dims/2) #!# N/V proportion
        agent_high = int(dims * 0.8) #!# agent proportion
        patient_low = int(dims * 0.7) #!# patient proportion
        self.link_wei[0, :mid, mid:agent_high] = 0.3 #!# initial strength
        self.link_wei[1, :mid, patient_low:] = 0.3
        self.ent_bias = zeros(self.D)
        self.ent_bias += log(self.D/self.C - 1) #!# initial bias (currently so that expit(bias) = C/D)
        # Initialise pred weights
        if init_card is None: init_card = dims/2
        self.pred_wei = random.uniform(0, init_range, (self.V, self.D))
        for i,p in enumerate(preds):
            if is_verb(p):
                self.pred_wei[i, mid:] = 0
                self.pred_wei[i, :mid] *= random.binomial(1, 2*init_card / dims, mid)
            else:
                self.pred_wei[i, :mid] = 0
                self.pred_wei[i, mid:] *= random.binomial(1, 2*init_card / dims, mid)
        
        self.pred_bias = empty((self.V,))
        self.pred_bias[:] = init_bias
        
        # Ignore preds that don't occur
        self.pred_wei[self.freq == 0] = 0
        
        # For sampling:
        self.calc_av_pred()  # average predicate
        self.get_pred_tokens(freq)  # pred tokens, for sampling preds
        
        print("Converting to shared memory")
        # Convert to shared memory
        self.freq = make_shared(self.freq)
        self.link_wei = make_shared(self.link_wei)
        self.ent_bias = make_shared(self.ent_bias)
        self.pred_wei = make_shared(self.pred_wei)
        self.pred_bias = make_shared(self.pred_bias)
        
        # Package for training setup
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
        prob = self.pred_wei[pred].clip(low, high)  # Use ent bias?
        return self.sample_card_restr(prob)
    
    def closest_preds(self, preds, number=1):
        """
        Find the nearest neighbour to some preds
        :param preds: an iterable of predicates
        :param number: how many neighbours to return (default 1)
        :return: the nearest neighbours
        """
        res = []
        for p in preds:
            vec = self.pred_wei[p]
            if not vec.any():  # if all entries are zero
                res.append(None)
                continue
            dot_prod = dot(self.pred_wei, vec)
            dist = dot_prod / norm(self.pred_wei, axis=1)
            dist = nan_to_num(dist)
            # The closest pred will have the second largest dot product
            # (Largest is the pred itself)
            indices = dist.argpartition(tuple(range(-1-number,0)))[-1-number:-1]
            res.append(indices)
        return res


class SemFuncModel_FactorisedPreds(SemFuncModel):
    """
    SemFuncModel with factorised pred parameters
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
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        if init_card is None: init_card = dims
        self.pred_embed = random.uniform(0, init_range, (self.V, self.E)) \
                        * random.binomial(1, init_card / dims, (self.V, self.E))
        self.pred_factor = random.uniform(0, init_range, (self.E, self.D)) \
                         * random.binomial(1, init_card / dims, (self.E, self.D))
        self.pred_bias = empty((self.V,))
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
    

# Converting pydmrs data to form required by SemFuncModel

def reform_links(self, node, ents):
    """
    Get the links from a PointerNode,
    and convert them to the form required by SemFuncModel
    :param node: a PointerNode
    :param ents: a matrix of entity vectors (indexed by nodeid)
    """
    out_labs = [l.rargname for l in node.get_out(itr=True)]
    out_vecs = [ents[l.end] for l in node.get_out(itr=True)]
    in_labs = [l.rargname for l in node.get_in(itr=True)]
    in_vecs = [ents[l.start] for l in node.get_in(itr=True)]
    return out_labs, out_vecs, in_labs, in_vecs

def reform_out_links(self, node, ents):
    """
    Get the outgoing links from a PointerNode,
    and convert them to the form required by SemFuncModel
    :param node: a PointerNode
    :param ents: a matrix of entity vectors (indexed by nodeid)
    """
    out_labs = [l.rargname for l in node.get_out(itr=True)]
    out_vecs = [ents[l.end] for l in node.get_out(itr=True)]
    return out_labs, out_vecs
