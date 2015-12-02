from numpy import array, random, tensordot, zeros, outer
from scipy.special import expit
from dmrs.core import DictPointDmrs as Dmrs
from dmrs.core import PointerNode as Node
from dmrs.core import RealPred, GPred, Link, LinkLabel

# Hyperparameters

DIMS = 2
RATE = 0.1
EPOCHS = 10

# Corpus

dmrs = [Dmrs([Node(1,RealPred('chase','v')),
              Node(2,RealPred('dog','n')),
              Node(3,RealPred('cat','n'))],
             [Link(1,2,'ARG1','NEQ'),
              Link(1,3,'ARG2','NEQ')],
             top=1,
             ident=1),
        Dmrs([Node(1,RealPred('chase','v')),
              Node(2,RealPred('cat','n')),
              Node(3,RealPred('mouse','n'))],
             [Link(1,2,'ARG1','NEQ'),
              Link(1,3,'ARG2','NEQ')],
             top=1,
             ident=2),
        Dmrs([Node(1,RealPred('chase','v')),
              Node(2,RealPred('dog','n')),
              Node(3,RealPred('mouse','n'))],
             [Link(1,2,'ARG1','NEQ'),
              Link(1,3,'ARG2','NEQ')],
             top=1,
             ident=3)]

# Latent variables and weights

def mapzip(func, iterable):
    return list(map(func,*list(zip(*iterable))))
def dictify(iterable):
    return {i:x for i,x in enumerate(iterable)}

nodes = dictify(n for x in dmrs for n in x.iter_nodes())
preds = dictify({n.pred for x in dmrs for n in x.iter_nodes()})
links = dictify(mapzip(LinkLabel,[('ARG1','NEQ'),('ARG2','NEQ')]))

i_node = {(n.graph.ident,n.nodeid):i for i,n in nodes.items()}
i_pred = {p:i for i,p in preds.items()}
i_link = {l:i for i,l in links.items()}

N = DIMS
K = len(nodes)
V = len(preds)
L = len(links)

entities = random.binomial(1, 0.5, (K,N))
link_wei = random.uniform(-0.1, 0.1, (L,N,N))
pred_wei = random.uniform(-0.1, 0.1, (V,N))

# Particle for negative samples

par_dmrs = [Dmrs([Node(1,GPred(None)),
                  Node(2,GPred(None)),
                  Node(3,GPred(None))],
                 [Link(1,2,'ARG1','NEQ'),
                  Link(1,3,'ARG2','NEQ')],
                  top=1,
                  ident=0)]

par_nodes = dictify(n for x in par_dmrs for n in x.iter_nodes())
P = len(par_nodes)
i_par_node = {(n.graph.ident,n.nodeid):i for i,n in par_nodes.items()}
par_ents = random.binomial(1, 0.5, (P,N))

def resample():
    # Latent entities
    for i,n in nodes.items():
        p = i_pred[n.pred]
        g = n.graph.ident
        negenergy = array(pred_wei[p,:], copy=True)
        for link in n.outgoing:
            j = i_node[g, link.end]
            k = i_link[link.label]
            negenergy += tensordot(link_wei[k,:,:],
                                   entities[j,:],
                                   (1,0))
        for link in n.incoming:
            j = i_node[g, link.start]
            k = i_link[link.label]
            negenergy += tensordot(link_wei[k,:,:],
                                   entities[j,:],
                                   (0,0))
        prob = expit(negenergy)
        entities[i,:] = random.binomial(1, prob)
    # Particle
    for i,n in par_nodes.items():
        g = n.graph.ident
        negenergy = zeros(N)
        for link in n.outgoing:
            j = i_par_node[g, link.end]
            k = i_link[link.label]
            negenergy += tensordot(link_wei[k,:,:],
                                   par_ents[j,:],
                                   (1,0))
        for link in n.incoming:
            j = i_par_node[g, link.start]
            k = i_link[link.label]
            negenergy += tensordot(link_wei[k,:,:],
                                   par_ents[j,:],
                                   (0,0))
        prob = expit(negenergy)
        par_ents[i,:] = random.binomial(1, prob)

def contrast():
    # Corpus
    link_obs = zeros((L,N,N))
    link_neg = zeros((L,N,N))
    pred_obs = zeros((V,N))
    pred_neg = zeros((V,N))
    for i,n in nodes.items():
        g = n.graph.ident
        for link in n.outgoing:
            j = i_node[g, link.end]
            k = i_link[link.label]
            link_obs[k,:,:] += outer(entities[i,:],entities[j,:])
        p = i_pred[n.pred]
        q = random.randint(V)
        pred_obs[p,:] += entities[i,:]
        pred_neg[q,:] += entities[i,:]
    # Particle
    for i,n in par_nodes.items():
        g = n.graph.ident
        for link in n.outgoing:
            j = i_par_node[g, link.end]
            k = i_link[link.label]
            link_neg[k,:,:] += outer(par_ents[i,:],par_ents[j,:])
    return link_obs-3*link_neg, pred_obs-pred_neg

def descend(link_weights, pred_weights):
    link_grad, pred_grad = contrast()
    link_weights += RATE * link_grad
    pred_weights += RATE * pred_grad


for e in range(EPOCHS):
    resample()
    descend(link_wei, pred_wei)
    print(entities)