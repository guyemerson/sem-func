import pickle, os
from numpy import zeros, zeros_like, sqrt
from multiprocessing import Manager, Process

from __config__.filepath import INIT_DIR
from utils import make_shared, sparse_like, shared_zeros_like

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
        self.inherit()
        # Hyperparameters:
        # Learning rate
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        # Regularisation
        self.L2_link = 2 * l2 / sqrt(l2_ratio)  # TODO make regularisation strength a list, like the weights (and change 'descend' functions appropriately)
        self.L2_pred = 2 * l2 * sqrt(l2_ratio)
        self.L1_link = l1 / sqrt(l1_ratio)
        self.L1_pred = l1 * sqrt(l1_ratio)
        self.L1_ent = l1_ent
        self.L2_ent = 2 * l2_ent
        # Metropolis-Hasting steps
        self.ent_steps = ent_steps
        self.pred_steps = pred_steps
    
    def inherit(self):
        """
        Assign attributes of self.model to self
        """
        self.link_weights = self.model.link_weights  # list of link weight tensors
        self.link_local_weights = self.model.link_local_weights
        self.link_global_weights = self.model.link_global_weights
        self.pred_weights = self.model.pred_weights  # list of pred weight tensors
        self.pred_local_weights = self.model.pred_local_weights
        self.pred_global_weights = self.model.pred_global_weights
        self.all_weights = self.link_weights + self.pred_weights  # all weights
    
    def make_shared(self):
        """
        Make sure that tensors can be shared across processes
        """
        self.model.make_shared()
        self.inherit()
    
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
            out_vecs = ents[out_ids]
            in_vecs = ents[in_ids]
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
        for nodeid, out_labs, out_ids, in_labs, _ in batch:
            # For each node, add gradients from outgoing links
            # (Incoming links are not included - this assumes we have all the relevant nodes)
            vec = ents[nodeid]
            out_vecs = [ents[i] for i in out_ids]
            self.model.observe_out_links(vec, out_labs, out_vecs, gradient_matrices, len(in_labs))
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
        link_grads, pred_grads = self.model.init_observe_latent_batch(batch, neg_preds)
        link_counts = zeros(self.model.L)
        # For each node, add gradients
        for nodeid, pred, out_labs, out_ids, in_labs, in_ids in batch:
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
    
    def link_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
        """
        raise NotImplementedError
    
    def pred_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
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
        :param neg_link_counts: how many times each link is present in the fantasy particle
        """
        # Resample latent variables
        for _ in range(self.ent_steps):
            self.resample_conditional_batch(pos_batch, pos_ents)
        for _ in range(self.pred_steps):
            self.resample_pred_batch(pos_batch, pos_ents, neg_preds)
        self.resample_background_batch(neg_batch, neg_ents)  # TODO control how many particle steps are taken
        
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
    
    # Loading from file
    
    @staticmethod
    def load(fname, directory=INIT_DIR, with_workers=False):
        """
        Load a TrainingSetup instance from a file
        """
        with open(os.path.join(directory, fname)+'.pkl', 'rb') as f:
            setup = pickle.load(f)
        
        # Make weights shared
        setup.make_shared()
        
        # (Optionally) start workers for weight updates:
        if with_workers:
            setup.start_update_workers() 
        
        # Load aux info
        with open(os.path.join(directory, fname)+'.aux.pkl', 'rb') as f:
            aux_info = pickle.load(f)
        
        return setup, aux_info
    
    # Multiprocessing
    
    def start_update_workers(self):
        """
        Start worker processes to manage updates
        """
        # Queues for weight updates
        manager = Manager()
        self.link_update_queues = [manager.Queue() for _ in self.link_weights]
        self.pred_update_queues = [manager.Queue() for _ in self.pred_weights]
        # Processes to consume the queued weight updates
        for i in range(len(self.link_weights)):
            worker = Process(target=self.link_update_constructor(i))
            worker.start()
        for i in range(len(self.pred_weights)):
            worker = Process(target=self.pred_update_constructor(i))
            worker.start()


class AdaGradTrainingSetup(TrainingSetup):
    """
    Use AdaGrad (with decay of the sum of the squared gradients, as in RMSprop)
    """
    def __init__(self, *args, ada_decay=1, **kwargs):
        """
        Initialise as TrainingSetup, but also initialise shared square gradient matrices
        """
        super().__init__(*args, **kwargs)
        
        # Decay hyperparameter
        if not (ada_decay > 0 and ada_decay <= 1):
            raise ValueError('ada_decay must be in the range (0,1]')
        self.ada_decay = ada_decay
        
        # Squared gradients
        self.link_sqsum = [shared_zeros_like(m) for m in self.link_weights]
        self.pred_sqsum = [shared_zeros_like(m) for m in self.pred_weights]
    
    def make_shared(self):
        """
        Make sure that tensors can be shared across processes
        """
        super().make_shared()
        self.link_sqsum = [make_shared(m) for m in self.link_sqsum]
        self.pred_sqsum = [make_shared(m) for m in self.pred_sqsum]
        
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
            # Add regularisation
            grad.array[::n_pred] -= self.L1_pred
            grad.array[::n_pred] -= self.pred_weights[i][grad.indices[::n_pred]] * self.L2_pred
            # Calculate square
            sq = grad.array ** 2
            # Divide by root sum square
            grad.array /= sqrt((self.pred_sqsum[i][grad.indices] + sq).clip(10**-12))  # Prevent zero-division errors
            # Multiply by learning rate
            grad.array *= self.rate_pred
            # Descend (or rather, add to queue)
            self.pred_update_queues[i].put((sq, grad))
        
        # Recalculate average predicate
        self.model.calc_av_pred()
    
    def link_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
        """
        def update():
            """
            Update a link weight matrix
            """
            weight = self.link_weights[i]
            sqsum = self.link_sqsum[i]
            queue = self.link_update_queues[i]
            while True:
                item = queue.get()
                if item is not None:
                    sqgrad, step = item
                    sqsum *= self.ada_decay
                    sqsum += sqgrad
                    weight += step.clip(-weight)
                else:
                    break
        return update
    
    def pred_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        Updates expected as SparseRows objects.
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
        """
        def update():
            """
            Update a pred weight matrix
            """
            weight = self.pred_weights[i]
            sqsum = self.pred_sqsum[i]
            queue = self.pred_update_queues[i]
            while True:
                item = queue.get()
                if item is not None:
                    sqgrad, step = item
                    assert step.next == step.indices.shape[0]
                    sqsum[step.indices] *= self.ada_decay
                    sqsum[step.indices] += sqgrad
                    weight[step.indices] += step.array.clip(-weight[step.indices])
                else:
                    break
        return update


class AdamTrainingSetup(TrainingSetup):
    """
    Use Adaptive Moment Estimation (Adam), which keeps a moving average of gradients
    """
    def __init__(self, *args, mean_decay=0.9, var_decay=0.999, **kwargs):
        """
        Initialise as TrainingSetup, but also initialise shared square gradient matrices
        """
        super().__init__(*args, **kwargs)
        
        # Decay hyperparameters
        if not (mean_decay > 0 and mean_decay < 1):
            raise ValueError('mean_decay must be in the range (0,1)')
        self.mean_decay = mean_decay
        if not (var_decay > 0 and var_decay < 1):
            raise ValueError('var_decay must be in the range (0,1)')
        self.var_decay = var_decay
        
        # Moving averages of gradient and squared gradient
        self.link_mean = [shared_zeros_like(m) for m in self.link_weights]
        self.pred_mean = [shared_zeros_like(m) for m in self.pred_weights]
        self.link_var = [shared_zeros_like(m) for m in self.link_weights]
        self.pred_var = [shared_zeros_like(m) for m in self.pred_weights]
    
    def make_shared(self):
        """
        Make sure that tensors can be shared across processes
        """
        super().make_shared()
        self.link_mean = [make_shared(m) for m in self.link_mean]
        self.pred_mean = [make_shared(m) for m in self.pred_mean]
        self.link_var = [make_shared(m) for m in self.link_var]
        self.pred_var = [make_shared(m) for m in self.pred_var]
        
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
            # Scale
            grad *= (1 - self.mean_decay)
            sq *= (1 - self.var_decay)
            # Calculate step
            mean = self.mean_decay * self.link_mean[i] + grad  #!# Use bias-corrected estimate?
            var = self.var_decay * self.link_var[i] + sq  #!# Use bias-corrected estimate?
            step = self.rate_link * mean / sqrt(var.clip(10**-12))  # Prevent zero-division errors
            # Descend (or rather, add to queue)
            self.link_update_queues[i].put((grad, sq, step))
        
        for i, grad in enumerate(pred_gradients):
            # Add regularisation
            grad.array[::n_pred] -= self.L1_pred
            grad.array[::n_pred] -= self.pred_weights[i][grad.indices[::n_pred]] * self.L2_pred
            # Calculate square
            sq = grad.array ** 2
            # Scale
            grad.array *= (1 - self.mean_decay)
            sq *= (1 - self.var_decay)
            # Calculate step
            mean = self.mean_decay * self.pred_mean[i][grad.indices] + grad.array  #!# Use bias-corrected estimate?
            var = self.var_decay * self.pred_var[i][grad.indices] + sq  #!# Use bias-corrected estimate?
            step = self.rate_pred * mean / sqrt(var.clip(10**-12))  # Prevent zero-division errors
            # Descend (or rather, add to queue)
            self.pred_update_queues[i].put((grad, sq, step))
        
        # Recalculate average predicate
        self.model.calc_av_pred()
    
    def link_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
        """
        def update():
            """
            Update a link weight matrix
            """
            weight = self.link_weights[i]
            mean = self.link_mean[i]
            var = self.link_var[i]
            queue = self.link_update_queues[i]
            while True:
                item = queue.get()
                if item is not None:
                    grad, sq, step = item
                    mean *= self.mean_decay
                    mean += grad
                    var *= self.var_decay
                    var += sq
                    weight += step.clip(-weight)
                else:
                    break
        return update
    
    def pred_update_constructor(self, i):
        """
        Construct an update function for a link weight matrix,
        which will consume the updates generated by self.descend
        Updates expected as SparseRows objects.
        :param i: index of the weight matrix
        :return: function to be used by a multiprocessing.Process
        """
        def update():
            """
            Update a pred weight matrix
            """
            weight = self.pred_weights[i]
            mean = self.pred_mean[i]
            var = self.pred_var[i]
            queue = self.pred_update_queues[i]
            while True:
                item = queue.get()
                if item is not None:
                    grad, sq, step = item
                    assert grad.next == grad.indices.shape[0]
                    mean[grad.indices] *= self.mean_decay
                    mean[grad.indices] += grad.array
                    var[grad.indices] *= self.var_decay
                    var[grad.indices] += sq
                    weight[grad.indices] += step.clip(-weight[grad.indices])
                else:
                    break
        return update


class AdaDeltaTrainingSetup(TrainingSetup):
    pass

class AdameltaTrainingSetup(TrainingSetup):
    pass
