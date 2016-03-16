from numpy import array, random, tensordot, dot, zeros, zeros_like, outer, arange, absolute, sign, minimum, amax, convolve, bool_, empty
from scipy.special import expit
from scipy.spatial.distance import cosine
from math import sqrt, exp
from collections import Counter
import pickle

class SemFuncModel():
    """
    The core semantic function model, including the background distribution
    """
    def __init__(self, preds, links, freq, dims, card, bias, init_range=0):
        """
        Initialise the model
        :param preds: names of predicates
        :param links: names of links
        :param freq: frequency of each predicate
        :param dims: dimension of latent entities
        :param card: cardinality of latent entities
        :param bias: bias for calculating semantic function values
        :param init_range: (optional) range for initialising pred weights
        """
        # Names for human readability
        self.pred_name = preds
        self.link_name = links
        # Parameters
        self.bias = bias
        if isinstance(freq, list):
            freq = array(freq)
        self.freq = freq / sum(freq)
        assert len(freq) == len(preds)
        # Constants
        self.D = dims
        self.V = len(preds)
        self.L = len(links)
        self.C = card
        # Weight matrices
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        self.pred_wei = random.uniform(0, init_range, (self.V, self.D))
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
        return expit(dot(ent, self.pred_wei[pred]) + self.bias)
    
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
    
    def observe_pred(self, vector, pred, gradient_matrix=None):
        """
        Calculate pred weight gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param gradient_matrix: (optional) the matrix which gradients should be added to
        :return: a vector of gradients (not the whole matrix!)
        """
        grad_vector = vector * (1 - self.prob(vector, pred)) 
        if gradient_matrix is not None:
            gradient_matrix[pred] += grad_vector
        return grad_vector
    
    def observe_latent(self, vector, pred, neg_preds, out_labels, out_vectors, link_grad_matrix=None, pred_grad_matrix=None):
        """
        Calculate multiple gradients for a node
        :param vector: an entity vector
        :param pred: a predicate
        :param neg_preds: an iterable of predicates
        :param out_labels: an iterable of link labels
        :param out_vectors: an iterable of entity vectors
        :param link_grad_matrix: (optional) the matrix which link weight gradients should be added to
        :param pred_grad_matrix: (optional) the matrix which pred weight gradients should be added to
        :return: link gradient matrix, pred gradient matrix
        """
        # Initialise matrices if not given
        if link_grad_matrix is None:
            link_grad_matrix = zeros_like(self.link_wei)
        if pred_grad_matrix is None:
            pred_grad_matrix = zeros_like(self.pred_wei)
        # Add gradients...
        # ...from links:
        self.observe_out_links(vector, out_labels, out_vectors, link_grad_matrix)
        # ...from the pred:
        self.observe_pred(vector, pred, pred_grad_matrix)
        # ...from the negative preds:
        num_preds = neg_preds.shape[0]
        for p in neg_preds:
            pred_grad_matrix[p] -= self.observe_pred(vector, p) / num_preds
        # Return gradient matrices
        return link_grad_matrix, pred_grad_matrix
    
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
    
    def pred_energy(self, ent, pred):
        """
        Calculate the energy of a predicate with a given entity
        :param ent: an entity vector
        :param pred: a predicate
        :return: the energy
        """
        return -( dot(self.pred_wei[pred], ent) + self.bias )
    
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
    def __init__(self, model, rate, rate_ratio, l1, l1_ratio, l2, l2_ratio):
        """
        Initialise the training setup
        :param model: the semantic function model
        :param rate: overall training rate
        :param rate_ratio: ratio between pred and link training rates
        :param l1: overall L1 regularisation strength
        :param l1_ratio: ratio between pred and link L1 regularisation strengths
        :param l2: overall L2 regularisation strength
        :param l2_ratio: ratio between pred and link L2 regularisation strengths
        """
        # Semantic function model
        self.model = model
        self.link_wei = model.link_wei
        self.pred_wei = model.pred_wei
        # Hyperparameters
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        self.L2_link = 1 - 2 * self.rate_link * l2 / sqrt(l2_ratio)
        self.L2_pred = 1 - 2 * self.rate_pred * l2 * sqrt(l2_ratio)
        self.L1_link = self.rate_link * l1 / sqrt(l1_ratio)
        self.L1_pred = self.rate_pred * l1 * sqrt(l1_ratio)
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
        :return: link gradient matrix, pred gradient matrix
        """
        link_grad = zeros_like(self.link_wei)
        pred_grad = zeros_like(self.pred_wei)
        for nodeid, pred, out_labs, out_ids, _, _ in batch:
            # For each node, add gradients
            vec = ents[nodeid]
            npreds = neg_preds[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            self.model.observe_latent(vec, pred, npreds, out_labs, out_vecs, link_grad, pred_grad)
        return link_grad, pred_grad
    
    # Gradient descent
    
    def descend(self, link_gradient, pred_gradient, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradient: gradient for link weights
        :param pred_gradient: gradient for pred weights
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        # Update from the gradient
        self.link_wei += link_gradient
        self.pred_wei += pred_gradient
        # Apply regularisation
        self.link_wei *= self.L2_link
        self.link_wei -= self.L1_link
        if pred_list:
            for p in pred_list:
                self.pred_wei[p] *= self.L2_pred
                self.pred_wei[p] -= self.L1_pred
        else:
            self.pred_wei *= self.L2_pred
            self.pred_wei -= self.L1_pred
        # Remove negative weights
        self.link_wei.clip(0, out=self.link_wei)
        self.pred_wei.clip(0, out=self.pred_wei)
        # Recalculate average predicate
        self.model.calc_av_pred()
    
    # Batch training
    
    def train_batch(self, pos_batch, pos_ents, neg_preds, neg_batch, neg_ents):
        """
        Train the model on a minibatch
        :param pos_batch: iterable (from data) of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param pos_ents: matrix of latent entity vectors
        :param neg_preds: matrix of sampled negative predicates
        :param neg_batch: iterable (from fantasy particle) of (nodeid, out_labs, out_ids, in_labs, in_ids) tuples
        :param neg_ents: matrix of particle entity vectors
        """
        # Resample latent variables
        self.resample_conditional_batch(pos_batch, pos_ents)
        self.resample_pred_batch(pos_batch, pos_ents, neg_preds)
        self.resample_background_batch(neg_batch, neg_ents)
        
        # Ratio in size between positive and negative samples
        # (Note that this assumes positive and negative links are balanced)
        neg_ratio = len(pos_batch) / len(neg_batch)
        
        # Observe gradients
        link_del, pred_del = self.observe_latent_batch(pos_batch, pos_ents, neg_preds)
        link_del -= neg_ratio * self.observe_particle_batch(neg_batch, neg_ents)
        
        # Descend
        preds = [x[1] for x in pos_batch]  # Only regularise the preds we've just seen
        self.descend(link_del, pred_del, preds)
    
    # Testing functions
    
    def graph_energy(self, nodes, ents):
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


class CoreTrainer():
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
    
    def train(self, epochs, minibatch, print_every):
        """
        Train the model on the data
        :param epochs: number of passes over the data
        :param minibatch: size of a minibatch (as a number of graphs)
        :param print_every: how many epochs should pass before printing
        """
        M = minibatch
        indices = arange(self.N)
        for e in range(epochs):
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            for i in range(0, self.N, M):
                # Get the nodes for this batch
                batch = indices[i : i+M]
                # Train on this batch
                self.setup.train_batch(batch, self.ents, self.neg_preds, self.neg_nodes, self.neg_ents)
                
            # Print regularly
            if e % print_every == 0:
                print(self.model.link_wei)
                print(self.model.pred_wei)
                print(amax(absolute(self.model.link_wei)))
                print(amax(absolute(self.model.pred_wei)))
                #print(self.average_energy())
                with open('../data/out.pkl','wb') as f:
                    pickle.dump(self, f)



class ToyTrainingSetup():
    """
    A semantic function model with a training regime.
    Expects pydmrs data during training.
    """
    def __init__(self, model, rate, rate_ratio, l1, l1_ratio, l2, l2_ratio):
        """
        Initialise the training setup
        :param model: the semantic function model
        :param rate: overall training rate
        :param rate_ratio: ratio between pred and link training rates
        :param l1: overall L1 regularisation strength
        :param l1_ratio: ratio between pred and link L1 regularisation strengths
        :param l2: overall L2 regularisation strength
        :param l2_ratio: ratio between pred and link L2 regularisation strengths
        """
        # Semantic function model
        self.model = model
        self.link_wei = model.link_wei
        self.pred_wei = model.pred_wei
        # Hyperparameters
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        self.L2_link = 1 - 2 * self.rate_link * l2 / sqrt(l2_ratio)
        self.L2_pred = 1 - 2 * self.rate_pred * l2 * sqrt(l2_ratio)
        self.L1_link = self.rate_link * l1 / sqrt(l1_ratio)
        self.L1_pred = self.rate_pred * l1 * sqrt(l1_ratio)
        '''
        # Moving average of squared gradients...
        self.link_sumsq = zeros_like(self.link_wei)
        self.pred_sumsq = zeros_like(self.pred_wei)
        '''
    
    # Batch resampling
    
    def resample_background_batch(self, nodes, ents):
        """
        Resample the latent entities for a batch of nodes,
        using the model's background distribution.
        :param nodes: an iterable of PointerNodes
        :param ents: a matrix of entity vectors (indexed by nodeid) 
        """
        for n in nodes:
            link_info = reform_links(n, ents)
            ents[n.nodeid] = self.model.resample_background(*link_info)
    
    def resample_conditional_batch(self, nodes, ents):
        """
        Resample the latent entities for a batch of nodes,
        conditioning on the nodes' preds.
        :param nodes: an iterable of PointerNodes
        :param ents: a matrix of entity vectors (indexed by nodeid)
        """
        for n in nodes:
            vec = ents[n.nodeid]
            pred = n.pred
            link_info = reform_links(n, ents)
            self.model.resample_conditional(vec, pred, *link_info)
    
    def resample_pred_batch(self, nodes, ents, neg_preds):
        """
        Resample the negative preds for a batch of nodes,
        conditioning on the latent entity vectors.
        :param nodes: iterable of nodes
        :param ents: matrix of entity vectors
        :param neg_preds: matrix of negative preds
        """
        for n in nodes:
            nid = n.nodeid
            old_preds = neg_preds[nid]
            vec = ents[nid]
            for i, pred in enumerate(old_preds):
                old_preds[i] = self.model.resample_pred(vec, pred)
    
    # Batch gradients
    
    def observe_particle_batch(self, nodes, ents):
        """
        Calculate gradients for link weights, for a fantasy particle
        :param nodes: an iterable of PointerNodes
        :param ents: a matrix of particle entity vectors  
        :return: a gradient matrix
        """
        gradient_matrix = zeros_like(self.link_wei)
        for n in nodes:
            # For each node, add gradients from outgoing links
            vec = ents[n.nodeid]
            out_labs, out_vecs = reform_out_links(n, ents)
            self.model.observe_out_links(vec, out_labs, out_vecs, gradient_matrix)
        return gradient_matrix
    
    def observe_latent_batch(self, nodes, ents, neg_preds):
        """
        Calculate gradients for a batch of nodes
        :param nodes: an iterable of PointerNodes
        :param ents: a matrix of latent entity vectors
        :param neg_preds: a matrix of negative samples of preds
        :return: link gradient matrix, pred gradient matrix
        """
        link_grad = zeros_like(self.link_wei)
        pred_grad = zeros_like(self.pred_wei)
        for n in nodes:
            # For each node, add gradients
            nid = n.nodeid
            out_labs, out_vecs = reform_out_links(n, ents)
            self.model.observe_latent(ents[nid], n.pred, neg_preds[nid], out_labs, out_vecs, link_grad, pred_grad)
        return link_grad, pred_grad
    
    # Gradient descent
    
    def descend(self, link_gradient, pred_gradient, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradient: gradient for link weights
        :param pred_gradient: gradient for pred weights
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        # Update from the gradient
        self.link_wei += link_gradient
        self.pred_wei += pred_gradient
        # Apply regularisation
        self.link_wei *= self.L2_link
        self.link_wei -= self.L1_link
        if pred_list:
            for p in pred_list:
                self.pred_wei[p] *= self.L2_pred
                self.pred_wei[p] -= self.L1_pred
        else:
            self.pred_wei *= self.L2_pred
            self.pred_wei -= self.L1_pred
        # Remove negative weights
        self.link_wei.clip(0, out=self.link_wei)
        self.pred_wei.clip(0, out=self.pred_wei)
        # Recalculate average predicate
        self.model.calc_av_pred()
    
    # Batch training
    
    def train_batch(self, pos_nodes, pos_ents, neg_preds, neg_nodes, neg_ents):
        """
        Train the model on a minibatch
        :param pos_nodes: iterable of PointerNodes (from data)
        :param pos_ents: matrix of latent entity vectors
        :param neg_preds: matrix of sampled negative predicates
        :param neg_nodes: iterable of PointerNodes (from fantasy particle)
        :param neg_ents: matrix of particle entity vectors
        """
        # Resample latent variables
        self.resample_conditional_batch(pos_nodes, pos_ents)
        self.resample_pred_batch(pos_nodes, pos_ents, neg_preds)
        self.resample_background_batch(neg_nodes, neg_ents)
        
        # Ratio in size between positive and negative samples
        # (Note that this assumes positive and negative links are balanced)
        neg_ratio = len(pos_nodes) / len(neg_nodes)
        
        # Observe gradients
        link_del, pred_del = self.observe_latent_batch(pos_nodes, pos_ents, neg_preds)
        link_del -= neg_ratio * self.observe_particle_batch(neg_nodes, neg_ents)
        
        # Descend
        preds = [n.pred for n in pos_nodes]
        self.descend(link_del, pred_del, preds)
    
    # Testing functions
    
    def graph_energy(self, graph, ents):
        """
        Find the energy of a DMRS graph, given entity vectors
        :param graph: a pydmrs Dmrs object
        :param ents: the entity vectors, indexed by nodeid
        :return: the energy
        """
        return self.model.background_energy(graph.iter_links(), ents)


class ToyTrainer():
    """
    A semantic function model with a training regime and data
    """
    def __init__(self, setup, data, neg_graphs, neg_samples):
        """
        Initialise the trainer
        :param setup: semantic function model with training setup
        :param data: list of DictDmrs graphs with increasing nodeids (observed data)
        :param neg_graphs: list of DictDmrs graphs with increasing nodeids (fantasy particles)
        :param neg_samples: number of negative pred samples to draw for each node
        """
        # Training setup
        self.setup = setup
        self.model = setup.model
        # Dicts for graphs, nodes, and pred frequencies
        self.graphs = data
        self.nodes = [n for g in data for n in g.nodes]
        for i, n in enumerate(self.nodes): assert i == n.nodeid
        self.N = len(self.nodes)
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            self.ents[i] = self.model.init_vec_from_pred(n.pred)
        # Particles for negative samples
        self.neg_graphs = neg_graphs
        self.neg_nodes = [n for g in neg_graphs for n in g.nodes]
        for i, n in enumerate(self.neg_nodes): assert i == n.nodeid
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, self.model.C/self.model.D, (self.K, self.model.D))
        # Negative pred samples
        self.NEG = neg_samples
        self.neg_preds = empty((self.N, neg_samples))
        for n in self.nodes:
            self.neg_preds[n.nodeid, :] = n.pred  # Initialise all pred samples as the nodes' preds
    
    def train(self, epochs, minibatch, print_every):
        """
        Train the model on the data
        :param epochs: number of passes over the data
        :param minibatch: size of a minibatch (as a number of graphs)
        :param print_every: how many epochs should pass before printing
        """
        G = len(self.graphs)
        M = minibatch
        indices = arange(G)
        for e in range(epochs):
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            for i in range(0, G, M):
                # Get the nodes for this batch
                batch_graphs = [self.graphs[j] for j in indices[i : i+M]]
                batch_nodes = [n for g in batch_graphs for n in g.iter_nodes()]
                # Train on this batch
                self.setup.train_batch(batch_nodes, self.ents, self.neg_preds, self.neg_nodes, self.neg_ents)
                
            # Print regularly
            if e % print_every == 0:
                print(self.model.link_wei)
                print(self.model.pred_wei)
                print(amax(absolute(self.model.link_wei)))
                print(amax(absolute(self.model.pred_wei)))
                #print(self.average_energy())
                with open('../data/out.pkl','wb') as f:
                    pickle.dump(self, f)
