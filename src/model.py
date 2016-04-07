from math import sqrt, exp
from numpy import array, random, dot, zeros, zeros_like, outer, arange, unravel_index, bool_, empty, histogram, count_nonzero, inf, tril, nan_to_num
from numpy.linalg import norm
import pickle, numpy as np
from multiprocessing import Array, Pool
from scipy.spatial.distance import cosine
from scipy.special import expit

def make_shared(array):
    """
    Convert a numpy array to a multiprocessing array with numpy access
    """
    # Get the C type for the array
    ctype = np.ctypeslib.as_ctypes(array)
    while not isinstance(ctype, str): ctype = ctype._type_
    # Create a shared array
    shared = Array(ctype, array.flatten())
    # Create a new numpy array from the shared array
    flat_array = np.frombuffer(shared._obj, #._wrapper.create_memoryview(),
                               dtype = array.dtype)
    # Reshape the new array
    return flat_array.reshape(array.shape)

def is_verb(string):
    """
    Check if a predstring is for a verb or a noun
    """
    return string.split('_')[-2] == 'v'


class SemFuncModel():
    """
    The core semantic function model, including the background distribution
    """
    def __init__(self, *arg, **kwargs):
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
                    print(prob)
                    print(intermed)
                    print(i, aux)
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
        ratio = self.prob(new_ent, pred) / self.prob(old_ent, pred)
        
        # Next, background energy of entities:
        negenergy = 0
        for n, label in enumerate(out_labels):
            negenergy += dot(self.link_wei[label, new_i, :], out_vectors[n])
            negenergy -= dot(self.link_wei[label, old_i, :], out_vectors[n])
        for n, label in enumerate(in_labels):
            negenergy += dot(self.link_wei[label, :, new_i], in_vectors[n])
            negenergy -= dot(self.link_wei[label, :, old_i], in_vectors[n])
        
        if chosen:
            # Finally, weighted number of other predicates that are true:
            # (Use an approximation, rather than summing over all predicates...)
            negenergy += 0.5 * (self.av_pred[old_i] - self.av_pred[new_i])
        
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
        gradient_matrix, = matrices
        # Calculate gradient for each link
        for i, label in enumerate(out_labels):
            gradient_matrix[label] += outer(vector, out_vectors[i])
        return gradient_matrix,
    
    def observe_links(self, vector, out_labels, out_vectors, in_labels, in_vectors, matrices=None):
        """
        Calculate link weight gradients for the outgoing links of a node
        (the gradients for incoming links will be found when considering the other node)
        :param vector: an entity vector
        :param out_labels: an iterable of link labels (outgoing)
        :param out_vectors: an iterable of entity vectors (outgoing)
        :param in_labels: an iterable of link labels (incoming)
        :param in_vectors: an iterable of entity vectors (incoming)
        :param matrices: (optional) the matrices which gradients should be added to
        :return: gradient matrices
        """
        # Initialise matrices if not given
        if matrices is None:
            matrices = [zeros_like(m) for m in self.link_weights]
        # Unpack matrices
        gradient_matrix, = matrices
        # Calculate gradient for each link
        for i, label in enumerate(out_labels):
            gradient_matrix[label] += outer(vector, out_vectors[i])
        for i, label in enumerate(in_labels):
            gradient_matrix[label] += outer(in_vectors[i], vector)
        return gradient_matrix,
    
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
    
    def observe_latent(self, vector, pred, neg_preds, out_labels, out_vectors, in_labels, in_vectors, link_matrices=None, pred_matrices=None):
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
        :return: link gradient matrices, pred gradient matrices
        """
        # Initialise matrices if not given
        if link_matrices is None:
            link_matrices = [zeros_like(m) for m in self.link_weights]
        if pred_matrices is None:
            pred_matrices = [zeros_like(m) for m in self.pred_weights]
        # Add gradients...
        # ...from links:
        self.observe_links(vector, out_labels, out_vectors, in_labels, in_vectors, link_matrices)
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
        v = self.init_vec_from_pred(pred)
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
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        mid = int(dims/2)
        agent_high = int(dims * 0.8)
        patient_low = int(dims * 0.7)
        self.link_wei[0, :mid, mid:agent_high] = 0.3
        self.link_wei[1, :mid, patient_low:] = 0.3
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
        pred_toks = []  # fill with pred tokens, for sampling preds
        for i, f in enumerate(freq):
            pred_toks.extend([i]*f)
        self.pred_tokens = array(pred_toks)
        
        print("Converting to shared memory")
        # Convert to shared memory
        self.freq = make_shared(self.freq)
        self.link_wei = make_shared(self.link_wei)
        self.pred_wei = make_shared(self.pred_wei)
        self.pred_bias = make_shared(self.pred_bias)
        self.pred_tokens = make_shared(self.pred_tokens)
        
        # Package for training setup
        self.link_weights = [self.link_wei]
        self.pred_local_weights = [self.pred_wei,
                                   self.pred_bias]
        self.pred_global_weights= []
        self.pred_weights = self.pred_local_weights + self.pred_global_weights
        
        self.bias_weights = [self.pred_bias]
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
        gradient_matrix[pred] += vector * factor
        bias_gradient_vector[pred] += - factor  # all biases assumed to be negative
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
        prob = self.pred_wei[pred].clip(low, high)
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
            indices = dist.argpartition(tuple(range(-1,-1-number,-1)))[-1-number:-1]
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
        pred_toks = []  # fill with pred tokens, for sampling preds
        for i, f in enumerate(freq):  # The original ints, not the normalised values
            pred_toks.extend([i]*f)
        self.pred_tokens = array(pred_toks)
        
        print("Converting to shared memory")
        # Convert to shared memory
        self.freq = make_shared(self.freq)
        self.link_wei = make_shared(self.link_wei)
        self.pred_embed = make_shared(self.pred_embed)
        self.pred_factor = make_shared(self.pred_factor)
        self.pred_bias = make_shared(self.pred_bias)
        self.pred_tokens = make_shared(self.pred_tokens)
        
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
        self.link_weights = model.link_weights  # list of link weight tensors
        self.pred_weights = model.pred_weights  # list of pred weight tensors
        self.pred_local_weights = model.pred_local_weights
        self.pred_global_weights = model.pred_global_weights
        self.all_weights = self.link_weights + self.pred_weights  # all weights
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
        :return: gradient matrices
        """
        gradient_matrices = [zeros_like(m) for m in self.link_weights]
        for nodeid, out_labs, out_ids, _, _ in batch:
            # For each node, add gradients from outgoing links
            # (Incoming links are not included - this assumes we have all the relevant nodes)
            vec = ents[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            self.model.observe_out_links(vec, out_labs, out_vecs, gradient_matrices)
        return gradient_matrices
    
    def observe_latent_batch(self, batch, ents, neg_preds):
        """
        Calculate gradients for a batch of nodes
        :param batch: an iterable of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param ents: a matrix of latent entity vectors
        :param neg_preds: a matrix of negative samples of preds
        :return: link gradient matrices, pred gradient matrices
        """
        # Initialise gradient matrices
        link_grads = [zeros_like(m) for m in self.link_weights]
        pred_grads = [zeros_like(m) for m in self.pred_weights]
        for nodeid, pred, out_labs, out_ids, in_labs, in_ids in batch:
            # For each node, add gradients
            # Look up the vector, neg preds, and linked vectors
            vec = ents[nodeid]
            npreds = neg_preds[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            in_vecs = [ents[i] for i in in_ids]
            # Observe the gradient
            self.model.observe_latent(vec, pred, npreds, out_labs, out_vecs, in_labs, in_vecs, link_grads, pred_grads)
        # Each link will be observed twice in an epoch
        for m in link_grads:
            m /= 2
        return link_grads, pred_grads
    
    # Gradient descent
    
    def descend(self, link_gradients, pred_gradients, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradients: gradients for link weights
        :param pred_gradients: gradients for pred weights
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        # Update from the gradient
        for i, grad in enumerate(link_gradients):
            self.link_weights[i] += grad
        for i, grad in enumerate(pred_gradients):
            self.pred_weights[i] += grad
        
        # Apply regularisation
        for wei in self.link_weights:
            wei *= self.L2_link
            wei -= self.L1_link
        for wei in self.pred_global_weights:
            wei *= self.L2_pred
            wei -= self.L1_pred
        if pred_list:
            for wei in self.pred_local_weights:
                for p in pred_list:
                    wei[p] *= self.L2_pred
                    wei[p] -= self.L1_pred
        else:
            for wei in self.pred_local_weights:
                wei *= self.L2_pred
                wei -= self.L1_pred
        
        # Remove negative weights
        for wei in self.all_weights:
            wei.clip(0, out=wei)
        
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
        link_dels, pred_dels, = self.observe_latent_batch(pos_batch, pos_ents, neg_preds)
        neg_link_dels = self.observe_particle_batch(neg_batch, neg_ents)
        
        # Average gradients by batch size
        # (Note that this assumes positive and negative links are balanced)
        for delta in link_dels + pred_dels:
            delta /= len(pos_batch)
        for i, delta in enumerate(neg_link_dels):
            link_dels[i] -= delta / len(neg_batch)
        
        # Descend
        preds = [x[1] for x in pos_batch]  # Only regularise the preds we've just seen
        self.descend(link_dels, pred_dels, preds)
    
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
        # Negative pred samples
        self.NEG = neg_samples
        # Data
        self.filename = None
        self.load_data(data)
        # Fantasy particles
        self.neg_nodes = particle
        for i, n in enumerate(self.neg_nodes): assert i == n[0]
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, self.model.C/self.model.D, (self.K, self.model.D))
    
    def load_data(self, data):
        """
        Load data from a list
        :param data: observed data of the form (nodeid, pred, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        """
        # Dicts for graphs, nodes, and pred frequencies
        self.nodes = data
        for i, n in enumerate(self.nodes): assert i == n[0]
        self.N = len(self.nodes)
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            self.ents[i] = self.model.init_vec_from_pred(n[1])
        # Negative pred samples
        self.neg_preds = empty((self.N, self.NEG), dtype=int)
        for n in self.nodes:
            self.neg_preds[n[0], :] = n[1]  # Initialise all pred samples as the nodes' preds
    
    def load_file(self, filehandle):
        """
        Load data from a file
        :param filehandle: pickled data
        """
        data = pickle.load(filehandle)
        self.load_data(data)
        self.filename = filehandle.name
        self.epochs = 0
    
    def report(self, histogram_bins, bias_histogram_bins, num_preds=5):
        """
        Print a summary of the current state of training the model
        :param histogram_bins: edges of bins for non-bias weights (0 and inf will be added)
        :param bias_histogram_bins: edges of bins for bias weights (0 and inf will be added)
        :param num_preds: number of preds to print the nearest neighbours of
        """
        # Get histogram
        histo, histo_bias = self.model.get_all_histograms(histogram_bins, bias_histogram_bins)
        # Print to console
        print()
        print('File {} epoch {} complete!'.format(self.filename, self.epochs))
        print('Weight histogram (link, then pred):')
        print(histo)
        print('Bias histogram (pred):')
        print(histo_bias)
        print('max link weights:')
        for m in self.model.link_weights:
            print('\t', m.max())
        print('max global pred weights:')
        for m in self.model.pred_global_weights:
            print('\t', m.max())
        print('max local pred weights:')
        for m in self.model.pred_local_weights:
            i_max = unravel_index(m.argmax(), m.shape)
            print('\t', m[i_max], '\t', self.model.pred_name[i_max[0]])
        print('avg data background E:', self.setup.graph_background_energy(self.nodes, self.ents) / self.N)
        print('avg part background E:', self.setup.graph_background_energy(self.neg_nodes, self.neg_ents) / self.K)  # Check for new samples?
        print('avg data pred t:', sum(self.model.prob(self.ents[n[0]], n[1]) for n in self.nodes) / self.N)
        print('avg part pred t:', sum(self.model.prob(self.ents[n[0]], p) for n in self.nodes for p in self.neg_preds[n[0]]) / self.N / self.NEG)  # Just check different preds?
        #print('closest preds:')
        #for p, q in self.model.closest_pairs(12, 'parameters'):
        #    print(p,q)
        print('nearest neighbours:')
        # Get frequent preds
        if not hasattr(self, 'pred_list'):
            self.pred_list = list(self.model.freq.argpartition(tuple(range(-1,-1-num_preds,-1)))[-num_preds:])
        # Get the first few preds in the current file
        self.pred_list[num_preds:] = [n[1] for n in self.nodes[:num_preds]]
        nearest = self.model.closest_preds(self.pred_list, 3)
        for i, p in enumerate(self.pred_list):
            if nearest[i] is not None:
                neighbours = ', '.join(self.model.pred_name[x] for x in nearest[i])
            else:
                neighbours = ''
            print('{}: {}'.format(self.model.pred_name[p], neighbours))
    
    def train(self, epochs, minibatch, print_every=inf, histogram_bins=(0.05,0.2,1), bias_histogram_bins=(4,5,6,10), dump_file=None):
        """
        Train the model on the data
        :param epochs: number of passes over the data
        :param minibatch: size of a minibatch (as a number of graphs)
        :param print_every: how many epochs should pass before printing
            (default: don't print)
        :param histogram_bins: edges of bins to summarise distribution of weights
            (default: 0.05, 0.2, 1)
        :param bias_histogram_bins: edges of bins to summarise distribution of biases
            (default: 4, 5, 6, 10)
        :param dump_file: (optional) file to save the trained model (dumps after printing)
        """
        # Indices of nodes, to be randomised
        indices = arange(self.N)
        for e in range(epochs):
            # Record that another epoch has passed
            self.epochs += 1
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
                # Print a summary
                self.report(histogram_bins, bias_histogram_bins)
                # Save to file
                if dump_file:
                    with open(dump_file, 'wb') as f:
                        pickle.dump(self.setup, f)
