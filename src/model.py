from math import sqrt, exp
from numpy import array, random, tensordot, dot, zeros, zeros_like, outer, arange, amax, convolve, bool_, empty, histogram, count_nonzero, inf
import pickle
from scipy.spatial.distance import cosine
from scipy.special import expit


class SemFuncModel():
    """
    The core semantic function model, including the background distribution
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
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        if init_card is None: init_card = dims
        self.pred_wei = random.uniform(0, init_range, (self.V, self.D)) \
                      * random.binomial(1, init_card / dims, (self.V, self.D))
        self.pred_bias = empty((self.V,))
        self.pred_bias[:] = init_bias
        # For sampling:
        self.calc_av_pred()  # average predicate
        pred_toks = []  # fill with pred tokens, for sampling preds
        for i, f in enumerate(freq):
            pred_toks.extend([i]*f)
        self.pred_tokens = array(pred_toks)
    
    # Semantic functions
    
    def prob(self, ent, pred):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: a probability
        """
        return expit(dot(ent, self.pred_wei[pred]) + self.pred_bias[pred])
    
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
        negenergy = zeros(self.D)
        for i, label in enumerate(out_labels):
            negenergy += tensordot(self.link_wei[label], out_vectors[i], (1,0))
        for i, label in enumerate(in_labels):
            negenergy += tensordot(self.link_wei[label], in_vectors[i], (0,0))
        # Expit gives the probability of each component if there are no sparsity constraints
        p = expit(negenergy) # Warning! If the negenergy is above 710, expit returns nan
        return self.sample_card_restr(p)
        
    def sample_card_restr(self, prob):
        """
        Sample a vector from component probabilities,
        restricting the total cardinality.
        :param prob: the probability of each component being on
        """
        minp = 1 - prob
        
        for p in prob:
            if p == 1:
                print(prob)
                raise Exception("prob 1!")
        
        # Sparsity constraints can be enforced using belief propagation (sum-product algorithm)
        # We introduce intermediate nodes which count how many components have been turned on so far
        # Pass messages up
        intermed = [array([minp[0], prob[0]])]
        for i in range(1,self.D-1):  # [1, 2, ..., D-2]
            message = convolve(intermed[-1], [minp[i], prob[i]])[:self.C+1]
            intermed.append(message)
        
        # Fix total number of components, and pass messages down
        vec = zeros(self.D, dtype=bool_)  # Output vector
        aux = self.C  # Number of components still to come
        # Iteratively sample
        for i in range(self.D-1, -1, -1):  # [D-1, D-2, ..., 0] 
            if aux == i+1:  # All remaining components are on
                vec[:i+1] = 1
                break
            elif aux == 0:  # All remaining components are off
                vec[:i+1] = 0
            else:
                # Unnormalised probabilities of being on or off:
                ein = prob[i] * intermed[i-1][aux-1]
                aus = minp[i] * intermed[i-1][aux]
                if ein == 0 and aus == 0:
                    print(prob)
                    print(minp)
                    print(intermed)
                    print(i, aux)
                    raise Exception('div zero!')
                # Probability of being on:
                on = ein/(ein+aus)
                if on < 0 or on > 1:
                    raise Exception('not prob')
                # Random sample:
                if random.binomial(1, on):
                    # Update vector and count
                    vec[i] = 1
                    aux -= 1
        return vec
    
    def resample_conditional(self, old_ent, pred, out_labels, out_vectors, in_labels, in_vectors):
        """
        Resample a latent entity, given a predicate and links to other entities.
        Uses Metropolis-Hastings, potentially turning one component on and one off.
        :param old_ent: the current entity vector
        :param pred: the predicate for the entity 
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param in_labels: an iterable of link labels
        :param in_vectors: an iterable of entity vectors
        :return: a sampled entity vector (changed in place!)
        """
        # Pick an on and an off unit to switch
        old_jth = random.randint(self.C)
        new_jth = random.randint(self.D-self.C)
        # Find these units
        on = 0
        for i, val in enumerate(old_ent):
            if val:
                if on == old_jth:
                    old_i = i
                    break
                else:
                    on += 1
        off = 0
        for i, val in enumerate(old_ent):
            if not val:
                if off == new_jth:
                    new_i = i
                    break
                else:
                    off += 1
        # Propose new entity
        new_ent = array(old_ent)
        new_ent[old_i] = 0
        new_ent[new_i] = 1
        
        # Calculate Metropolis-Hastings ratio
        # First, probability of each predicate being applicable:
        ratio = self.prob(new_ent, pred) / self.prob(old_ent, pred)
        
        # Next, background energy of entities:
        negenergy = 0
        for n, label in enumerate(out_labels):
            negenergy += dot(self.link_wei[label, new_i, :], out_vectors[n])
            negenergy -= dot(self.link_wei[label, old_i, :], out_vectors[n])
        for n, label in enumerate(in_labels):
            negenergy += dot(self.link_wei[label, :, new_i], in_vectors[n])
            negenergy -= dot(self.link_wei[label, :, old_i], in_vectors[n])
        
        # Finally, weighted number of other predicates that are true:
        # (Use an approximation...)
        negenergy += 0.5 * (self.av_pred[old_i] - self.av_pred[new_i])
        
        ratio *= exp(negenergy)
        
        """
        # Exact number of other predicates... slow!
        ratio /= sum(self.freq[i] * self.prob(new_ent, i) for i in range(self.V))
        ratio *= sum(self.freq[i] * self.prob(old_ent, i) for i in range(self.V))
        """
        
        # Decide whether to accept or reject the new entity
        if ratio > 1:
            switch = True
        else:
            switch = random.binomial(1, ratio)
        
        # Change the vector accordingly
        if switch:
            old_ent[old_i] = 0
            old_ent[new_i] = 1
        return old_ent
    
    def calc_av_pred(self):
        """
        Recalculate the average predicate
        (used as an approximation in conditional sampling)
        """
        # Weighted sum of predicates
        self.av_pred = dot(self.freq, self.pred_wei)
    
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
        # Decide whether to switch
        if ratio > 1:
            switch = True
        else:
            switch = random.binomial(1, ratio)
        # Return corresponding pred
        if switch:
            return new_pred
        else:
            return old_pred
    
    # Gradients
    
    def observe_out_links(self, vector, out_labels, out_vectors, gradient_matrix=None):
        """
        Calculate link weight gradients for the outgoing links of a node
        (the gradients for incoming links will be found when considering the other node)
        :param vector: an entity vector
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param gradient_matrix: (optional) the matrix which gradients should be added to
        :return: a matrix of gradients
        """
        # Initialise a matrix if not given one
        if gradient_matrix is None:
            gradient_matrix = zeros_like(self.link_wei)
        # Calculate gradient for each link
        for i, label in enumerate(out_labels):
            gradient_matrix[label] += outer(vector, out_vectors[i])
        return gradient_matrix
    
    def observe_pred(self, vector, pred, gradient_matrix=None, bias_gradient_vector=None):
        """
        Calculate pred weight gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param gradient_matrix: (optional) the matrix which gradients should be added to
        :param bias_gradient_vector: (optional) the vector which bias gradients should be added to
        :return: a vector of gradients (not the whole matrix!), and the bias gradient
        """
        factor = 1 - self.prob(vector, pred)
        grad_vector = vector * factor
        bias_grad = factor
        if gradient_matrix is not None:
            gradient_matrix[pred] += grad_vector
        if bias_gradient_vector is not None:
            bias_gradient_vector[pred] += bias_grad
        return grad_vector, bias_grad
    
    def observe_latent(self, vector, pred, neg_preds, out_labels, out_vectors, link_grad_matrix=None, pred_grad_matrix=None, pred_bias_grad_vector=None):
        """
        Calculate multiple gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param neg_preds: an iterable of predicates
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param link_grad_matrix: (optional) the matrix which link weight gradients should be added to
        :param pred_grad_matrix: (optional) the matrix which pred weight gradients should be added to
        :param pred_bias_grad_vector: (optional) the vector which pred bias gradients should be added to
        :return: link gradient matrix, pred gradient matrix, pred bias gradient vector
        """
        # Initialise matrices if not given
        if link_grad_matrix is None:
            link_grad_matrix = zeros_like(self.link_wei)
        if pred_grad_matrix is None:
            pred_grad_matrix = zeros_like(self.pred_wei)
        if pred_bias_grad_vector is None:
            pred_bias_grad_vector = zeros_like(self.pred_bias)
        # Add gradients...
        # ...from links:
        self.observe_out_links(vector, out_labels, out_vectors, link_grad_matrix)
        # ...from the pred:
        self.observe_pred(vector, pred, pred_grad_matrix, pred_bias_grad_vector)
        # ...from the negative preds:
        num_preds = neg_preds.shape[0]
        for p in neg_preds:
            grad_vec, bias_grad = self.observe_pred(vector, p)
            grad_vec /= num_preds
            bias_grad /= num_preds
            pred_grad_matrix[p] -= grad_vec
            pred_bias_grad_vector[p] -= bias_grad
        # Return gradient matrices
        return link_grad_matrix, pred_grad_matrix, pred_bias_grad_vector
    
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
            start = l[0]
            end = l[1]
            label = l[2]
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
        return cosine(self.pred_wei[pred1],
                      self.pred_wei[pred2])
    
    def sample_from_pred(self, pred, samples=5, burnin=5, interval=2):
        """
        Sample entity vectors conditioned on a predicate
        :param pred: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :return: a generator of entity vectors
        """
        v = self.init_vec_from_pred(pred)
        for _ in range(max(0, burnin-interval)):
            self.resample_conditional(v, pred, (),(),(),())
        for _ in range(samples):
            for _ in range(interval):
                self.resample_conditional(v, pred, (),(),(),())
            yield v.copy()
    
    def mean_sample_from_pred(self, pred, samples=5, burnin=5, interval=2):
        """
        Sample entity vectors conditioned on a predicate, and average them
        :param pred: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :return: the mean entity vector
        """
        return sum(self.sample_from_pred(pred, samples=samples, burnin=burnin, interval=interval)) / samples
    
    def cosine_samples(self, pred1, pred2, samples=5, burnin=5, interval=2):
        """
        Calculate the average cosine distance (1 - normalised dot product)
        between sampled entity vectors conditioned on a pair of predicates
        :param pred1: a predicate
        :param pred2: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :return: the cosine distance
        """
        # As dot products are distributive over addition, we can take the dot product of the sums
        mean1 = self.mean_sample_from_pred(pred1, samples=samples, burnin=burnin, interval=interval)
        mean2 = self.mean_sample_from_pred(pred2, samples=samples, burnin=burnin, interval=interval)
        return cosine(mean1, mean2)
    
    def implies(self, pred1, pred2, samples=5, burnin=5, interval=2):
        """
        Calculate the probability that pred1 implies pred2
        :param pred1: a predicate
        :param pred2: a predicate
        :param samples: (default 5) number of samples to average over
        :param burnin: (default 5) number of samples to skip before averaging starts
        :param interval: (default 2) number of samples to take between those used in the average
        :return: the probability pred1 implies pred2
        """
        total = 0
        ents = self.sample_from_pred(pred1, samples=samples, burnin=burnin, interval=interval)
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
        prob = self.pred_wei[pred].clip(low, high)
        return self.sample_card_restr(prob)
    

class WrappedVectors():
    """
    Access vectors according to different indices.
    """
    def __init__(self, matrix, index):
        """
        Initialise the object
        :param matrix: a numpy matrix
        :param index: a dict mapping from desired keys to numpy indices
        """
        self.matrix = matrix
        self.index = index
    
    def __getitem__(self, key):
        """
        Get an item
        :param key: 2-tuple (vector index, numpy index)
        :return: what numpy would return 
        """
        return self.matrix[self.index[key[0]], key[1]]
    
    def __setitem__(self, key, value):
        """
        Set an item
        :param key: 2-tuple (vector index, numpy index)
        :param value: the new value
        """
        self.matrix[self.index[key[0]], key[1]] = value



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


# Convert a triple
def resample_background_triple(self, triple, ents):
    """
    Resample the latent entities for a triple,
    using the model's background distribution.
    :param triple: (event, agent, patient) nodeids
    :param ents: a matrix of entity vectors (indexed by nodeid)
    """
    event, agent, patient = triple
    labs = []
    nids = [] 
    if agent:
        labs.append(0)
        nids.append(agent)
        ents[agent] = self.model.resample_background((), (), [0], [ents[event]])
    if patient:
        labs.append(1)
        nids.append(patient)
        ents[patient] = self.model.resample_background((), (), [1], [ents[event]])
    ents[event] = self.model.resample_background(labs, nids, (), ())


class DirectTrainingSetup():
    """
    A semantic function model with a training regime.
    Expects preprocessed data during training.
    """
    def __init__(self, model, rate, rate_ratio, l1, l1_ratio, l2, l2_ratio, ent_steps, pred_steps):
        """
        Initialise the training setup
        :param model: the semantic function model
        :param rate: overall training rate
        :param rate_ratio: ratio between pred and link training rates
        :param l1: overall L1 regularisation strength
        :param l1_ratio: ratio between pred and link L1 regularisation strengths
        :param l2: overall L2 regularisation strength
        :param l2_ratio: ratio between pred and link L2 regularisation strengths
        :param ent_steps: (default 1) number of Metropolis-Hastings steps to make when resampling latent entities
        :param pred_steps: (default 1) number of Metropolis-Hastings steps to make when resampling negative predicates
        """
        # Semantic function model
        self.model = model
        self.link_wei = model.link_wei
        self.pred_wei = model.pred_wei
        self.pred_bias = model.pred_bias
        # Hyperparameters
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        self.L2_link = 1 - 2 * self.rate_link * l2 / sqrt(l2_ratio)
        self.L2_pred = 1 - 2 * self.rate_pred * l2 * sqrt(l2_ratio)
        self.L1_link = self.rate_link * l1 / sqrt(l1_ratio)
        self.L1_pred = self.rate_pred * l1 * sqrt(l1_ratio)
        # Metropolis-Hasting steps
        self.ent_steps = ent_steps
        self.pred_steps = pred_steps
        '''
        # Moving average of squared gradients...
        self.link_sumsq = zeros_like(self.link_wei)
        self.pred_sumsq = zeros_like(self.pred_wei)
        '''
    
    # Batch resampling
    
    def resample_background_batch(self, batch, ents):
        """
        Resample the latent entities for a batch of nodes,
        using the model's background distribution.
        :param batch: an iterable of (nodeid, out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: a matrix of entity vectors (indexed by nodeid) 
        """
        for nodeid, out_labs, out_ids, in_labs, in_ids in batch:
            out_vecs = [ents[i] for i in out_ids]
            in_vecs = [ents[i] for i in in_ids]
            ents[nodeid] = self.model.resample_background(out_labs, out_vecs, in_labs, in_vecs)
    
    def resample_conditional_batch(self, batch, ents):
        """
        Resample the latent entities for a batch of nodes,
        conditioning on the nodes' preds.
        :param batch: an iterable of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: a matrix of entity vectors (indexed by nodeid)
        """
        for nodeid, pred, out_labs, out_ids, in_labs, in_ids in batch:
            vec = ents[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            in_vecs = [ents[i] for i in in_ids]
            self.model.resample_conditional(vec, pred, out_labs, out_vecs, in_labs, in_vecs)
    
    def resample_pred_batch(self, batch, ents, neg_preds):
        """
        Resample the negative preds for a batch of nodes,
        conditioning on the latent entity vectors.
        :param batch: iterable of tuples (nodeid first element)
        :param ents: matrix of entity vectors
        :param neg_preds: matrix of negative preds
        """
        for x in batch:
            nid = x[0]
            old_preds = neg_preds[nid]
            vec = ents[nid]
            for i, pred in enumerate(old_preds):
                old_preds[i] = self.model.resample_pred(vec, pred)
    
    # Batch gradients
    
    def observe_particle_batch(self, batch, ents):
        """
        Calculate gradients for link weights, for a fantasy particle
        :param batch: an iterable of (nodeid, out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: a matrix of particle entity vectors  
        :return: a gradient matrix
        """
        gradient_matrix = zeros_like(self.link_wei)
        for nodeid, out_labs, out_ids, _, _ in batch:
            # For each node, add gradients from outgoing links
            vec = ents[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            self.model.observe_out_links(vec, out_labs, out_vecs, gradient_matrix)
        return gradient_matrix
    
    def observe_latent_batch(self, batch, ents, neg_preds):
        """
        Calculate gradients for a batch of nodes
        :param batch: an iterable of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: a matrix of latent entity vectors
        :param neg_preds: a matrix of negative samples of preds
        :return: link gradient matrix, pred gradient matrix, pred bias gradient vector
        """
        link_grad = zeros_like(self.link_wei)
        pred_grad = zeros_like(self.pred_wei)
        pred_bias_grad = zeros_like(self.pred_bias)
        for nodeid, pred, out_labs, out_ids, _, _ in batch:
            # For each node, add gradients
            vec = ents[nodeid]
            npreds = neg_preds[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            self.model.observe_latent(vec, pred, npreds, out_labs, out_vecs, link_grad, pred_grad, pred_bias_grad)
        return link_grad, pred_grad, pred_bias_grad
    
    # Gradient descent
    
    def descend(self, link_gradient, pred_gradient, pred_bias_gradient, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradient: gradient for link weights
        :param pred_gradient: gradient for pred weights
        :param pred_bias_gradient: gradient for pred biases
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        # Update from the gradient
        self.link_wei += link_gradient
        self.pred_wei += pred_gradient
        self.pred_bias += pred_bias_gradient
        # Apply regularisation
        self.link_wei *= self.L2_link
        self.link_wei -= self.L1_link
        if pred_list:
            for p in pred_list:
                self.pred_wei[p] *= self.L2_pred
                self.pred_wei[p] -= self.L1_pred
                self.pred_bias[p] *= self.L2_pred
                self.pred_bias[p] += self.L1_pred
        else:
            self.pred_wei *= self.L2_pred
            self.pred_wei -= self.L1_pred
            self.pred_bias *= self.L2_pred
            self.pred_bias += self.L1_pred
        # Remove negative weights
        self.link_wei.clip(0, out=self.link_wei)
        self.pred_wei.clip(0, out=self.pred_wei)
        self.pred_bias.clip(-inf, 0, out=self.pred_bias)
        # Recalculate average predicate
        self.model.calc_av_pred()
    
    # Batch training
    
    def train_batch(self, pos_batch, pos_ents, neg_preds, neg_batch, neg_ents):
        """
        Train the model on a minibatch
        :param pos_batch: list (from data) of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param pos_ents: matrix of latent entity vectors
        :param neg_preds: matrix of sampled negative predicates
        :param neg_batch: list (from fantasy particle) of (nodeid, out_labs, out_ids, in_labs, in_ids) tuples
        :param neg_ents: matrix of particle entity vectors
        """
        # Resample latent variables
        for _ in range(self.ent_steps):
            self.resample_conditional_batch(pos_batch, pos_ents)
        for _ in range(self.pred_steps):
            self.resample_pred_batch(pos_batch, pos_ents, neg_preds)
        self.resample_background_batch(neg_batch, neg_ents)
        
        # Observe gradients
        link_del, pred_del, pred_bias_del = self.observe_latent_batch(pos_batch, pos_ents, neg_preds)
        neg_link_del = self.observe_particle_batch(neg_batch, neg_ents)
        
        # Average gradients by batch size
        # (Note that this assumes positive and negative links are balanced)
        pred_bias_del /= len(pos_batch)
        pred_del /= len(pos_batch)
        link_del /= len(pos_batch)
        link_del -= neg_link_del / len(neg_batch)
        
        # Descend
        preds = [x[1] for x in pos_batch]  # Only regularise the preds we've just seen
        self.descend(link_del, pred_del, pred_bias_del, preds)
    
    # Testing functions
    
    def graph_background_energy(self, nodes, ents):
        """
        Find the energy of a DMRS graph, given entity vectors
        :param nodes: iterable of (nodeid, (pred,) out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: the entity vectors, indexed by nodeid
        :return: the energy
        """
        links = []
        for x in nodes:
            start = x[0]
            out_labs = x[-4]
            out_ids = x[-3]
            for i, lab in enumerate(out_labs):
                links.append([start, out_ids[i], lab])
        return self.model.background_energy(links, ents)


class DirectTrainer():
    """
    A semantic function model with a training regime and data
    """
    def __init__(self, setup, data, particle, neg_samples):
        """
        Initialise the trainer
        :param setup: semantic function model with training setup
        :param data: observed data of the form (nodeid, pred, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        :param particle: fantasy particle of the form (nodeid, out_labs, out_ids, in_labs, in_ids), with increasing nodeids 
        :param neg_samples: number of negative pred samples to draw for each node
        """
        # Training setup
        self.setup = setup
        self.model = setup.model
        # Dicts for graphs, nodes, and pred frequencies
        self.nodes = data
        for i, n in enumerate(self.nodes): assert i == n[0]
        self.N = len(self.nodes)
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            self.ents[i] = self.model.init_vec_from_pred(n[1])
        # Particles for negative samples
        self.neg_nodes = particle
        for i, n in enumerate(self.neg_nodes): assert i == n[0]
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, self.model.C/self.model.D, (self.K, self.model.D))
        # Negative pred samples
        self.NEG = neg_samples
        self.neg_preds = empty((self.N, neg_samples))
        for n in self.nodes:
            self.neg_preds[n[0], :] = n[1]  # Initialise all pred samples as the nodes' preds
    
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
    
    def train(self, epochs, minibatch, print_every, histogram_bins=(0.05,0.2,1), bias_histogram_bins=(4,5,6,10), dump_file=None):
        """
        Train the model on the data
        :param epochs: number of passes over the data
        :param minibatch: size of a minibatch (as a number of graphs)
        :param print_every: how many epochs should pass before printing
        :param histogram_bins: edges of bins to summarise distribution of weights
            (default: 0.05, 0.2, 1)
        :param bias_histogram_bins: edges of bins to summarise distribution of biases
        :param dump_file: (optional) file to save the trained model
        """
        # Record training in the setup
        self.setup.minibatch = minibatch
        if not hasattr(self.setup, 'epochs'):
            self.setup.epochs = 0
        
        # Histogram bins, for printing
        num_bins = len(histogram_bins) + 2
        histo = zeros((3, num_bins))
        histo[0,1:-1] = histogram_bins
        histo[0,0] = 0
        histo[0,-1] = inf
        
        num_bins_bias = len(bias_histogram_bins) + 1
        histo_bias = zeros((2, num_bins_bias))
        histo_bias[0, :-1] = bias_histogram_bins
        histo_bias[0,-1] = inf
        
        # Indices of nodes, to be randomised
        indices = arange(self.N)
        for e in range(epochs):
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            # Take batches
            for i in range(0, self.N, minibatch):
                # Get the nodes for this batch
                batch = [self.nodes[i] for i in indices[i : i+minibatch]]
                # Train on this batch
                self.setup.train_batch(batch, self.ents, self.neg_preds, self.neg_nodes, self.neg_ents)
                
            # Print regularly
            if (e+1) % print_every == 0:
                # Record training in the setup
                self.setup.epochs += print_every
                # Get histogram of weights
                histo[1] = self.get_histogram(self.model.link_wei, histo[0,1:-1])
                histo[2] = self.get_histogram(self.model.pred_wei, histo[0,1:-1])
                histo_bias[1] = self.get_histogram_bias(-self.model.pred_bias, histo_bias[0, :-1])
                # Print to console
                print('Epoch {} complete!'.format(self.setup.epochs))
                print('Weight histogram (link, then pred):')
                print(histo)
                print('Bias histogram (pred):')
                print(histo_bias)
                print('max link weight:', amax(self.model.link_wei))
                print('max pred weight:', amax(self.model.pred_wei))
                print('max  -pred bias:', amax(-self.model.pred_bias))
                print('avg data back E:', self.setup.graph_background_energy(self.nodes, self.ents) / self.N)
                print('avg part back E:', self.setup.graph_background_energy(self.neg_nodes, self.neg_ents) / self.K)
                print('avg data pred t:', sum(self.model.prob(self.ents[n[0]], n[1]) for n in self.nodes) / self.N)
                print('avg part pred t:', sum(self.model.prob(self.ents[n[0]], p) for n in self.nodes for p in self.neg_preds[n[0]]) / self.N / self.NEG)
                # Save to file
                if dump_file:
                    with open(dump_file, 'wb') as f:
                        pickle.dump(self.setup, f)
