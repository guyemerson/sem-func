from numpy import array, random, tensordot, zeros, outer, arange, absolute, sign, minimum
from scipy.special import expit
from scipy.spatial.distance import cosine
import pickle

class SemFuncModel():
    def __init__(self, corpus, neg_graphs, dims, rate_link, rate_pred, l2_link, l2_pred, l1_link, l1_pred, init_range):
        """
        Corpus and neg_graphs should each have distinct nodeids for all nodes
        """
        # Hyperparameters
        self.rate_link = rate_link
        self.rate_pred = rate_pred
        self.L2_link = l2_link
        self.L2_pred = l2_pred
        self.L1_link = l1_link
        self.L1_pred = l1_pred
        # Indices mapping to node tokens, predicate types, and link types
        self.graphs = dict(enumerate(corpus))
        self.nodes = {n.nodeid:n for x in corpus for n in x.iter_nodes()}
        # Dimensions of matrices
        self.D = dims
        self.N = len(self.nodes)
        self.V = 1 + max(n.pred for n in self.nodes.values())
        self.L = 1 + max(x.rargname for n in self.nodes.values() for x in n.outgoing)
        # Latent entities, link weights, and pred weights
        self.ents = random.binomial(1, 0.5, (self.N, self.D))
        self.link_wei = random.uniform(-init_range, init_range, (self.L, self.D, self.D)) # link, from, to
        self.pred_wei = random.uniform(-init_range, init_range, (self.V, self.D))
        self.link_sumsq = zeros((self.L, self.D, self.D))
        self.pred_sumsq = zeros((self.V, self.D))
        # Particles for negative samples
        self.neg_graphs = dict(enumerate(neg_graphs))
        self.neg_nodes = {n.nodeid:n for x in neg_graphs for n in x.iter_nodes()}
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, 0.5, (self.K, self.D))
        # Weight for negative samples
        pos_links = sum(len(n.outgoing) for n in self.nodes.values())
        neg_links = sum(len(n.outgoing) for n in self.neg_nodes.values())
        self.neg_link_weight = pos_links / neg_links
        #self.neg_pred_weight = len(self.nodes) / len(self.neg_nodes)
    
    # Training functions
    
    def resample(self, nodes, ents, pred=True):
        for n in nodes:
            if pred:
                negenergy = array(self.pred_wei[n.pred, :], copy=True)
            else:
                negenergy = zeros(self.D)
            for link in n.outgoing:
                negenergy += tensordot(self.link_wei[link.rargname, :, :],
                                       ents[link.end, :],
                                       (1,0))
            for link in n.incoming:
                negenergy += tensordot(self.link_wei[link.rargname, :, :],
                                       ents[link.start, :],
                                       (0,0))
            prob = expit(negenergy)
            # Warning!
            # If the negenergy is above 710, expit returns nan
            ents[n.nodeid, :] = random.binomial(1, prob)
    
    def sample_latent(self):
        self.resample(self.nodes.values(), self.ents)
    
    def sample_latent_batch(self, nodes):
        self.resample(nodes, self.ents)
    
    def sample_particle(self):
        self.resample(self.neg_nodes.values(), self.neg_ents, pred=False)
    
    def sample_particle_batch(self, nodes):
        self.resample(nodes, self.neg_ents, pred=False)
    
    def contrast(self):
        # Corpus
        link_obs = zeros((self.L, self.D, self.D))
        link_neg = zeros((self.L, self.D, self.D))
        pred_obs = zeros((self.V, self.D))
        pred_neg = zeros((self.V, self.D))
        for i,n in self.nodes.items():
            # Reinforce observed links
            for link in n.outgoing:
                link_obs[link.rargname, :,:] += outer(self.ents[i,:], self.ents[link.end, :])
            # Reinforce observed preds
            pred_obs[n.pred, :] += self.ents[i,:]
            # Negatively sample another entity
            m = random.randint(self.N)
            pred_neg[n.pred, :] += self.ents[m,:]
        # Particle
        for i,n in self.neg_nodes.items():
            for link in n.outgoing:
                link_neg[link.rargname, :,:] += outer(self.neg_ents[i,:], self.neg_ents[link.end, :])
        # Return steps for link weights and pred weights
        return (link_obs - self.neg_link_weight * link_neg,
                pred_obs - pred_neg) #self.neg_pred_weight * pred_neg)
    
    def observe_links(self, nodes, ents):
        link_obs = zeros((self.L, self.D, self.D))
        for n in nodes:
            for link in n.outgoing:
                link_obs[link.rargname, :,:] += outer(ents[n.nodeid, :], ents[link.end, :])
        return link_obs
    
    def observe_preds(self, nodes):
        pred_obs = zeros((self.V, self.D))
        for n in nodes:
            pred_obs[n.pred, :] += self.ents[n.nodeid, :]
        return pred_obs
    
    def neg_sample_preds(self, nodes):
        pred_neg = zeros((self.V, self.D))
        for n in nodes:
            m = random.randint(self.N)
            pred_neg[n.pred, :] += self.ents[m,:]
        return pred_neg
    
    def descend(self):
        link_del, pred_del = self.contrast()
        self.link_wei *= self.L2_link
        self.pred_wei *= self.L2_pred
        self.link_wei += self.rate_link * link_del
        self.pred_wei += self.rate_pred * pred_del
    
    def train(self, epochs, minibatch=10, print_every=10):
        G = len(self.graphs)
        indices = arange(G)
        for e in range(epochs):
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            for i in range(0, G, minibatch):
                batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                batch = [n for g in batch_graphs for n in g.iter_nodes()]
                # Resample latent variables
                self.sample_latent_batch(batch)
                self.sample_particle_batch(self.neg_nodes.values())
                neg_link_ratio = len(batch) / self.K
                # Observe latent variables, particles, and negative samples
                link_del = self.observe_links(batch, self.ents)
                link_del -= neg_link_ratio * self.observe_links(self.neg_nodes.values(), self.neg_ents)
                pred_del = self.observe_preds(batch)
                pred_del -= self.neg_sample_preds(batch)
                # Descend
                self.link_wei += self.rate_link * link_del
                self.pred_wei += self.rate_pred * pred_del
                #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                # Regularise
                self.link_wei *= self.L2_link
                l1_reg(self.link_wei, self.L1_link)
                for n in batch:
                    self.pred_wei[n.pred, :] *= self.L2_pred
                    l1_reg(self.pred_wei[n.pred, :], self.L1_pred)
            # Print regularly
            if e % print_every == 0:
                print(self.link_wei)
                print(self.pred_wei)
                with open('../data/out.pkl','wb') as f:
                    pickle.dump(self, f)
        
    def train_alternate(self, epochs, minibatch=10, print_every=3, streak=3):
        G = len(self.graphs)
        indices = arange(G)
        for e in range(epochs):
            # Optimise links
            for _ in range(streak):
                # Randomise batches
                # (At the moment, just one batch of particles)
                random.shuffle(indices)
                for i in range(0, G, minibatch):
                    batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                    batch = [n for g in batch_graphs for n in g.iter_nodes()]
                    # Resample latent variables
                    self.sample_latent_batch(batch)
                    self.sample_particle_batch(self.neg_nodes.values())
                    neg_link_ratio = len(batch) / self.K
                    # Observe latent variables, particles, and negative samples
                    link_del = self.observe_links(batch, self.ents)
                    link_del -= neg_link_ratio * self.observe_links(self.neg_nodes.values(), self.neg_ents)
                    # Descend
                    self.link_wei += self.rate_link * link_del
                    #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                    # Regularise
                    self.link_wei *= self.L2_link
                    l1_reg(self.link_wei, self.L1_link)
            # Optimise preds
            for _ in range(streak):
                random.shuffle(indices)
                for i in range(0, G, minibatch):
                    batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                    batch = [n for g in batch_graphs for n in g.iter_nodes()]
                    # Resample latent variables
                    self.sample_latent_batch(batch)
                    neg_link_ratio = len(batch) / self.K
                    # Observe latent variables, particles, and negative samples
                    pred_del = self.observe_preds(batch)
                    pred_del -= self.neg_sample_preds(batch)
                    # Descend
                    self.pred_wei += self.rate_pred * pred_del
                    #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                    # Regularise
                    for n in batch:
                        self.pred_wei[n.pred, :] *= self.L2_pred
                        l1_reg(self.pred_wei[n.pred, :], self.L1_pred)
            # Print regularly
            if e % print_every == 0:
                print(self.link_wei)
                print(self.pred_wei)
    
    # Testing functions
    
    def energy(self, graph, entities, pred=True):
        e = 0
        for n in graph.nodes:
            for link in n.outgoing:
                e -= tensordot(tensordot(self.link_wei[link.rargname, :,:],
                                         entities[n.nodeid, :], (0,0)),
                               entities[link.end, :], (0,0))
            if pred:
                e -= tensordot(entities[n.nodeid, :],
                               self.pred_wei[n.pred, :], (0,0))
        return e
    
    def sample_energy(self, graph, samples=5, burnin=5, interval=2, pred=True):
        e = 0
        raw_ents = random.binomial(1, 0.5, (len(graph), self.D))
        index = {n.nodeid:i for i,n in enumerate(graph.iter_nodes())}
        ents = WrappedVectors(raw_ents, index)
        for i in range(-burnin, 1+(samples-1)*interval):
            self.resample(graph.iter_nodes, ents, pred=pred)
            if i >= 0 and i % interval == 0:
                e -= self.energy(graph, ents, pred=pred)
        return e/samples
    
    def cosine_of_parameters(self, pred1, pred2):
        return cosine(self.pred_wei[pred1, :],
                      self.pred_wei[pred2, :])
    
    def cosine_samples(self, pred1, pred2, samples=5):
        total = 0
        for _ in range(samples):
            ent1 = random.binomial(1, self.pred_wei[pred1, :])
            ent2 = random.binomial(1, self.pred_wei[pred2, :])
            total += cosine(ent1, ent2)
        return total/samples 
    
    def implies(self, pred1, pred2, samples=5):
        total = 0
        for _ in range(samples):
            ent = random.binomial(1, self.pred_wei[pred1, :])
            total += self.prob(ent, pred2)
        return total/samples
    
    def prob(self, ent, pred):
        return expit(tensordot(self.pred_wei[pred, :],
                               ent, (0,0)))

class WrappedVectors():
    """
    Access vectors according to different indices
    """
    def __init__(self, matrix, index):
        self.matrix = matrix
        self.index = index
    def __getitem__(self, key):
        return self.matrix[self.index[key[0]], key[1]]
    def __setitem__(self, key, value):
        self.matrix[self.index[key[0]], key[1]] = value

def l1_reg(array, penalty):
    """
    Apply L1 regularisation
    """
    array -= minimum(absolute(array), penalty) * sign(array)