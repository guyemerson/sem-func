from math import sqrt
from numpy import zeros_like, empty, random, arange, amax, absolute
import pickle

from model import reform_links, reform_out_links


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
    
    def graph_background_energy(self, graph, ents):
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
