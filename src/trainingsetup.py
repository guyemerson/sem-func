import pickle, os
from numpy import zeros, zeros_like, array, sqrt
from multiprocessing import Queue

from utils import make_shared, sparse_like

class TrainingSetup():
    """
    A semantic function model with a training regime.
    Expects preprocessed data during training.
    """
    def __init__(self, model, rate, rate_ratio, l1, l1_ratio, l1_ent, l2, l2_ratio, l2_ent, ent_steps, pred_steps):
        """
        Initialise the training setup
        :param model: the semantic function model
        :param rate: overall training rate
        :param rate_ratio: ratio between pred and link training rates
        :param l1: overall L1 regularisation strength
        :param l1_ratio: ratio between pred and link L1 regularisation strengths
        :param l1_ent: L1 regularisation strength for entity biases
        :param l2: overall L2 regularisation strength
        :param l2_ratio: ratio between pred and link L2 regularisation strengths
        :param l2_ent: L2 regularisation strength for entity biases
        :param ent_steps: (default 1) number of Metropolis-Hastings steps to make when resampling latent entities
        :param pred_steps: (default 1) number of Metropolis-Hastings steps to make when resampling negative predicates
        """
        # Semantic function model
        self.model = model
        self.link_weights = model.link_weights  # list of link weight tensors
        self.link_local_weights = model.link_local_weights
        self.link_global_weights = model.link_global_weights
        self.pred_weights = model.pred_weights  # list of pred weight tensors
        self.pred_local_weights = model.pred_local_weights
        self.pred_global_weights = model.pred_global_weights
        self.all_weights = self.link_weights + self.pred_weights  # all weights
        # Hyperparameters
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        self.L2_link = 2 * l2 / sqrt(l2_ratio)
        self.L2_pred = 2 * l2 * sqrt(l2_ratio)
        self.L1_link = l1 / sqrt(l1_ratio)
        self.L1_pred = l1 * sqrt(l1_ratio)
        self.L1_ent = l1_ent
        self.L2_ent = l2_ent
        # Metropolis-Hasting steps
        self.ent_steps = ent_steps
        self.pred_steps = pred_steps
        # Queues for weight updates
        self.link_update_queues = [Queue() for _ in self.link_weights]
        self.pred_update_queues = [Queue() for _ in self.pred_weights]
    
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
        total_preds = len(batch) * (neg_preds.shape[1] + 1)
        pred_grads = [sparse_like(m, total_preds) for m in self.pred_local_weights]
        pred_grads += [zeros_like(m) for m in self.pred_global_weights]
        link_counts = zeros(self.model.L)
        for nodeid, pred, out_labs, out_ids, in_labs, in_ids in batch:
            # For each node, add gradients
            # Look up the vector, neg preds, and linked vectors
            vec = ents[nodeid]
            npreds = neg_preds[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            in_vecs = [ents[i] for i in in_ids]
            # Observe the gradient
            self.model.observe_latent(vec, pred, npreds, out_labs, out_vecs, in_labs, in_vecs, link_grads, pred_grads, link_counts)
        return link_grads, pred_grads, link_counts
    
    # Gradient descent
    
    def descend(self, link_gradients, pred_gradients, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradients: gradients for link weights
        :param pred_gradients: gradients for pred weights
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        raise NotImplementedError
    
    # Batch training
    
    def train_batch(self, pos_batch, pos_ents, neg_preds, neg_batch, neg_ents, neg_link_counts):
        """
        Train the model on a minibatch
        :param pos_batch: list (from data) of (nodeid, pred, out_labs, out_ids, in_labs, in_ids) tuples
        :param pos_ents: matrix of latent entity vectors
        :param neg_preds: matrix of sampled negative predicates
        :param neg_batch: list (from fantasy particle) of (nodeid, out_labs, out_ids, in_labs, in_ids) tuples
        :param neg_ents: matrix of particle entity vectors
        :param neg_link_counts: how many times each link is observed
        """
        # Resample latent variables
        for _ in range(self.ent_steps):
            self.resample_conditional_batch(pos_batch, pos_ents)
        for _ in range(self.pred_steps):
            self.resample_pred_batch(pos_batch, pos_ents, neg_preds)
        self.resample_background_batch(neg_batch, neg_ents)
        
        # Observe gradients
        link_dels, pred_dels, link_counts = self.observe_latent_batch(pos_batch, pos_ents, neg_preds)
        neg_link_dels = self.observe_particle_batch(neg_batch, neg_ents)
        
        # Average gradients by batch size
        # (this is a constant factor, so can be ignored...)
        #for delta in link_dels + pred_dels:
        #    delta /= len(pos_batch)
        
        # Balance particle links with observed links
        # Reshaping is necessary for broadcasting...
        # If link matrices are of different shapes, this needs to be handled differently
        link_ratio = (link_counts / neg_link_counts).reshape(-1,1,1)
        node_ratio = len(pos_batch) / len(neg_batch)
        for i, delta in enumerate(neg_link_dels):
            if i < len(self.link_local_weights):
                link_dels[i] -= delta * link_ratio
            else:
                link_dels[i] -= delta * node_ratio
        
        # Descend
        #preds = [x[1] for x in pos_batch]  # Only regularise the preds we've just seen
        self.descend(link_dels, pred_dels, neg_preds.shape[1]+1)
    
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
    
    @staticmethod
    def load(fname, directory='/anfs/bigdisc/gete2/wikiwoods/sem-func', with_tokens=False):
        """
        Load a TrainingSetup instance from a file
        """
        with open(os.path.join(directory, fname)+'.pkl', 'rb') as f:
            setup = pickle.load(f)
        if isinstance(setup.model.pred_name, str):
            with open(setup.model.pred_name, 'rb') as f:
                setup.model.pred_name = pickle.load(f)
        if isinstance(setup.model.freq, str):
            with open(setup.model.freq, 'rb') as f:
                freq = pickle.load(f)
            setup.model.freq = array(freq) / sum(freq)
            if with_tokens:
                setup.model.get_pred_tokens(freq)
        if setup.link_update_queues is None:
            setup.link_update_queues = [Queue() for _ in setup.link_weights]
        if setup.pred_update_queues is None:
            setup.pred_update_queues = [Queue() for _ in setup.pred_weights]
        
        with open(os.path.join(directory, fname)+'.aux.pkl', 'rb') as f:
            aux_info = pickle.load(f)
        
        return setup, aux_info


class DirectTrainingSetup(TrainingSetup):
    """
    Use the gradients directly
    """
    def __init__(self, *args, **kwargs):
        """
        L1 and L2 will be used directly
        """
        raise NotImplementedError
        
        super().__init__(*args, **kwargs)
        
        self.L1_link *= self.rate_link
        self.L1_pred *= self.rate_pred
        self.L2_link = 1 - self.L2_link * self.rate_link
        self.L2_pred = 1 - self.L2_pred * self.rate_pred
    
    def descend(self, link_gradients, pred_gradients, pred_list=None):
        """
        Descend the gradient and apply regularisation
        :param link_gradients: gradients for link weights
        :param pred_gradients: gradients for pred weights
        :param pred_list: (optional) restrict regularisation to these predicates
        """
        # Update from the gradient
        for i, grad in enumerate(link_gradients):
            self.link_weights[i] += grad * self.rate_link
        for i, grad in enumerate(pred_gradients):
            self.pred_weights[i] += grad * self.rate_pred
        
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


class AdaGradTrainingSetup(TrainingSetup):
    """
    Use AdaGrad
    """
    def __init__(self, *args, ada_decay=1, **kwargs):
        """
        Initialise as TrainingSetup, but also initialise shared square gradient matrices
        """
        super().__init__(*args, **kwargs)
        
        # Squared gradients
        assert ada_decay > 0
        assert ada_decay < 1
        self.ada_decay = ada_decay
        self.link_sqsum = [make_shared(zeros_like(m)) for m in self.link_weights]
        self.pred_sqsum = [make_shared(zeros_like(m)) for m in self.pred_weights]
        
    def descend(self, link_gradients, pred_gradients, n_pred):
        """
        Divide step lengths by the sum of the square gradients
        """
        for i, grad in enumerate(link_gradients):
            # Add regularisation
            if i < len(self.model.link_local_weights): #!# Specific to current model...
                L1 = self.L1_link
                L2 = self.L2_link
            else:
                L1 = self.L1_ent
                L2 = self.L2_ent
            grad -= L1
            grad -= self.link_weights[i] * L2
            # Calculate square
            sq = grad ** 2
            # Divide by root sum square
            grad /= sqrt((self.link_sqsum[i] + sq).clip(10**-12))  # Prevent zero-division errors
            # Multiply by learning rate
            grad *= self.rate_link
            # Descend (or rather, add to queue)
            self.link_update_queues[i].put((sq, grad))
        
        for i, grad in enumerate(pred_gradients):
            if n_pred and i < len(self.pred_local_weights):
                # Add regularisation
                grad.array[::n_pred] -= self.L1_pred
                grad.array[::n_pred] -= self.pred_weights[i][grad.indices[::n_pred]] * self.L2_pred
                # Increase square sums (or rather, add to queue)
                sq = grad.array ** 2
                # Divide by root sum square
                grad.array /= sqrt((self.pred_sqsum[i][grad.indices] + sq).clip(10**-12))  # Prevent zero-division errors
                # Multiply by learning rate
                grad.array *= self.rate_pred
            else:
                raise NotImplementedError
            # Descend (or rather, add to queue)
            self.pred_update_queues[i].put((sq, grad))
        
        # Recalculate average predicate
        self.model.calc_av_pred()
