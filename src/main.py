from numpy import array, random, tensordot, zeros, outer, float, arange, average, absolute, sign, minimum
from scipy.special import expit
from scipy.spatial.distance import cosine
from dmrs.core import DictPointDmrs as Dmrs
from dmrs.core import PointerNode as Node
from dmrs.core import RealPred, GPred, Link, LinkLabel
from copy import copy
import pickle
import warnings


# Model

class SemFuncModel():
    def __init__(self, corpus, neg_graphs, dims, rate, l2_link, l2_pred, l1_link, l1_pred, init_range):
        """
        Corpus and neg_graphs should each have distinct nodeids for all nodes
        """
        # Hyperparameters
        self.rate = rate
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
        self.link_wei = random.uniform(-init_range, init_range, (self.L, self.D, self.D))
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
        self.link_wei *= self.L2
        self.pred_wei *= self.L2
        self.link_wei += self.rate * link_del
        self.pred_wei += self.rate * pred_del
    
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
                self.link_wei += self.rate * link_del
                self.pred_wei += self.rate * pred_del
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
                    self.link_wei += self.rate * link_del
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
                    self.pred_wei += self.rate * pred_del
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


if __name__ == "__main__":
    
    # Positive graphs
    
    with open('../data/toy.pkl', 'rb') as f:
        intr_dmrs, tran_dmrs, ditr_dmrs = pickle.load(f)
    dmrs = intr_dmrs + tran_dmrs + ditr_dmrs
    freq = [len(intr_dmrs), len(tran_dmrs), len(ditr_dmrs)]
    
    # Negative graphs
    
    def empty_dmrs(i=0, num=1):
        assert num in [1,2,3]
        nodes = [Node(i+j, None) for j in range(num+1)]
        links = [Link(i, i+j+1, j, None) for j in range(num)]
        return Dmrs(nodes, links)
    
    neg_freq = [2,10,2]
    
    neg_dmrs = []
    i = 0
    for j in [1,2,3]:
        for _ in range(neg_freq[j-1]):
            neg_dmrs.append(empty_dmrs(i, j))
            i += j+1
    
    # Set up model
    
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 25,
                         rate = 10**-2,
                         l2_link = 0.999,
                         l2_pred = 0.999,
                         l1_link = 0.003,
                         l1_pred = 0.003,
                         init_range = 0.001)
    """
    # Stabilises to a local optimum distinguishing nouns and verbs:
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 25,
                         rate = 10**-3,
                         l2_link = 0.999,
                         l2_pred = 0.999,
                         init_range = 0.001)
    # Reducing the rate or increasing L2 seems to push the model into pure noise
    """
    
    """
    import toy
    for i in range(model.V):
        model.pred_wei[i,i] += 100
    for i in range(len(toy.noun)):
        model.pred_wei[i, 19] += 100
    for i in range(len(toy.intr)):
        model.pred_wei[i+toy.I, 20] += 100
        model.pred_wei[i+toy.I, 21] += 100
    for i in range(len(toy.tran)):
        model.pred_wei[i+toy.T, 20] += 100
        model.pred_wei[i+toy.T, 22] += 100
    model.pred_wei[18,20] += 100
    """
    # Try alternating training pred_wei and link_wei, each for some time?
    
    
    # Use higher negative sampling instead of L2 regularisation?
    # -> this seems to make all the weights negative... 
    # (Regularising common preds more is a good idea, right?
        
    
    #model.train(5000)
    model.train_alternate(5000)
    
    
    # Perhaps the space is too crowded...
    # There are 3 weights necessary to encode a predicate-argument pair
    # If we randomly sample a new/rare component for both,
    # then we can immediately get all three of these weights
    # -> look into sparse RBMs?
    # e.g. Cardinality-RBMs:
    # http://papers.nips.cc/paper/4668-cardinality-restricted-boltzmann-machines
    
    # Note that this is sparsity in the latent vectors, not sparsity in the weights
    
    # At the moment, everything that's "not useful" for a pred becomes a negative weight
    # Instead, perhaps only allow positive weights, and introduce a negative bias?
    
    # Try with more data?
    
    # Cheat the weights?
    
    # !!!
    # Training is optimising the situation RBM at the expense of the predicates
    # If all the features are nouny or verby, then we've collapsed the space to two states
    # This means we can predict the entity vectors extremely well (even if we can't predict the preds) 
    
    # Try training on preds more than on links?
    # Try contrastive divergence instead of fantasy particles? 