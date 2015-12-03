from numpy import array, random, tensordot, zeros, outer, float
from scipy.special import expit
from dmrs.core import DictPointDmrs as Dmrs
from dmrs.core import PointerNode as Node
from dmrs.core import RealPred, GPred, Link, LinkLabel
from copy import copy

# Toy Corpus

triples = [('dog','chase','cat'),
           ('dog','see','cat'),
           ('cat','see','dog'),
           ('dog','chase','dog'),
           ('cat','see','cat'),
           ('dog','see','dog')]
"""
triples = [('dog','chase','cat'),
           ('cat','chase','mouse'),
           ('mouse','eat','rice'),
           ('cat','eat','mouse'),
           ('dog','eat','mouse'),
           ('dog','eat','rice'),
           ('dog','chase','tail')]
"""

def simp(ident, subj, verb, obj):
    return Dmrs([Node(1,RealPred(verb,'v')),
              Node(2,RealPred(subj,'n')),
              Node(3,RealPred(obj,'n'))],
             [Link(1,2,'ARG1','NEQ'),
              Link(1,3,'ARG2','NEQ')],
             top=1,
             ident=ident)

dmrs = []
for i,t in enumerate(triples):
    dmrs.append(simp(i, *t))

# Negative graphs

def empty_dmrs():
    return Dmrs([Node(1,None),
                 Node(2,None),
                 Node(3,None)],
                [Link(1,2,'ARG1','NEQ'),
                 Link(1,3,'ARG2','NEQ')],
                top=1)

neg_dmrs = []
for _ in range(len(triples)):
    neg_dmrs.append(empty_dmrs()) 

# Utils

def map_unzip(func, iterable):
    return map(func,*list(zip(*iterable)))
def dictify(iterable):
    return dict(enumerate(iterable))

class SemFuncModel():
    def __init__(self, corpus, neg_graphs, dims, rate, l2, init_range):
        # Hyperparameters
        self.rate = rate
        self.L2 = l2
        # Indices mapping to node tokens, predicate types, and link types
        self.nodes = dictify(n for x in corpus for n in x.iter_nodes())
        self.preds = dictify({n.pred for n in self.nodes.values()})
        self.links = dictify(map_unzip(LinkLabel, [('ARG1','NEQ'),('ARG2','NEQ')]))
        # Dimensions of matrices
        self.D = dims
        self.N = len(self.nodes)
        self.V = len(self.preds)
        self.L = len(self.links)
        # Relabel nodes, preds, and links to be indices
        for i,n in self.nodes.items():
            if i in n.graph:
                n.graph.renumber_node(i, self.N+i)
            n.renumber(i)
        pred_index = {p:i for i,p in self.preds.items()}
        link_index = {l:i for i,l in self.links.items()}
        for n in self.nodes.values():
            n.pred = pred_index[n.pred]
            for link in copy(n.outgoing):
                n.graph.remove_link(link)
                n.graph.add_link(Link(link.start, link.end, link_index[link.label], None))
        # Latent entities, link weights, and pred weights
        self.ents = random.binomial(1, 0.5, (self.N, self.D))
        self.link_wei = random.uniform(-init_range, init_range, (self.L, self.D, self.D))
        self.pred_wei = random.uniform(-init_range, init_range, (self.V, self.D))
        self.link_sumsq = zeros((self.L, self.D, self.D))
        self.pred_sumsq = zeros((self.V, self.D))
        # Particles for negative samples
        self.neg_nodes = dictify(n for x in neg_graphs for n in x.iter_nodes())
        self.K = len(self.neg_nodes)
        for i,n in self.neg_nodes.items():
            if i in n.graph:
                n.graph.renumber_node(i, self.K+i)
            n.renumber(i)
        for n in self.neg_nodes.values():
            for link in copy(n.outgoing):
                n.graph.remove_link(link)
                n.graph.add_link(Link(link.start, link.end, link_index[link.label], None))
        self.neg_ents = random.binomial(1, 0.5, (self.K, self.D))
        # Weight for negative samples
        pos_links = sum(len(n.outgoing) for n in self.nodes.values())
        neg_links = sum(len(n.outgoing) for n in self.neg_nodes.values())
        self.neg_link_weight = pos_links / neg_links
        self.neg_pred_weight = len(self.nodes) / len(self.neg_nodes)
    
    # Training functions
    
    def resample(self, nodes, ents, pred=True):
        for n in nodes.values():
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
            ents[n.nodeid, :] = random.binomial(1, prob)
    
    def sample_latent(self):
        self.resample(self.nodes, self.ents)
    
    def sample_particle(self):
        self.resample(self.neg_nodes, self.neg_ents, pred=False)
    
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
            # Negatively sample another pred
            m = random.randint(self.N)
            pred_neg[self.nodes[m].pred, :] += self.ents[i,:]
        # Particle
        for i,n in self.neg_nodes.items():
            for link in n.outgoing:
                link_neg[link.rargname, :,:] += outer(self.neg_ents[i,:], self.neg_ents[link.end, :])
        # Return steps for link weights and pred weights
        return (link_obs - self.neg_link_weight * link_neg,
                pred_obs - self.neg_pred_weight * pred_neg)
    
    def descend(self):
        link_del, pred_del = self.contrast()
        self.link_wei *= self.L2
        self.pred_wei *= self.L2
        self.link_wei += self.rate * link_del
        self.pred_wei += self.rate * pred_del
    
    def train(self, epochs):
        for e in range(epochs):
            self.sample_latent()
            self.sample_particle()
            self.descend()
            if e % 100 == 0:
                print(self.link_wei)
                print(self.pred_wei)
                print(self.preds)
    
    # Testing functions
    
    def energy(self, graph, entities, pred=True):
        e = float(0)
        for n in graph.nodes:
            for link in n.outgoing:
                e -= tensordot(tensordot(self.link_wei[link.rargname, :,:],
                                         entities[n.nodeid, :], (0,0)),
                               entities[link.end, :], (0,0))
            if pred:
                e -= tensordot(entities[n.nodeid, :],
                               self.pred_wei[n.pred, :], (0,0))
        return e
    
    def sample_energy(self, graph, samples=1, burnin=5, interval=1, pred=True):
        pass
    
    def cosine(self, pred1, pred2):
        pass
    
    def implies(self, pred1, pred2):
        pass
    
    def prob(self, ent, pred):
        pass
