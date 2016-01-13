from numpy import array, random, tensordot, zeros, outer, float, arange, average, absolute, sign, minimum
from scipy.special import expit
from scipy.spatial.distance import cosine
from dmrs.core import DictPointDmrs as Dmrs
from dmrs.core import PointerNode as Node
from dmrs.core import RealPred, GPred, Link, LinkLabel
from copy import copy
import pickle
import warnings
from model import SemFuncModel


if __name__ == "__main__":
    
    # Positive graphs
    """
    with open('../data/toy.pkl', 'rb') as f:
        intr_dmrs, tran_dmrs, ditr_dmrs = pickle.load(f)
    dmrs = intr_dmrs + tran_dmrs + ditr_dmrs
    freq = [len(intr_dmrs), len(tran_dmrs), len(ditr_dmrs)]
    """
    with open('../data/tinytoy.pkl', 'rb') as f:
        dmrs = pickle.load(f)
    
    # Negative graphs
    
    def empty_dmrs(i=0, num=1):
        assert num in [1,2,3]
        nodes = [Node(i+j, None) for j in range(num+1)]
        links = [Link(i, i+j+1, j, None) for j in range(num)]
        return Dmrs(nodes, links)
    
    #neg_freq = [2,10,2]
    neg_freq = [3,0,0]
    
    neg_dmrs = []
    i = 0
    for j in [1,2,3]:
        for _ in range(neg_freq[j-1]):
            neg_dmrs.append(empty_dmrs(i, j))
            i += j+1
    
    # Set up model
    
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 20,
                         card = 2,
                         rate_link = 10**-5,
                         rate_pred = 10**-3,
                         l2_link = 1-10**-6,
                         l2_pred = 1-10**-5,
                         l1_link = 10**-9,
                         l1_pred = 10**-8,
                         init_range = 0,
                         print_every = 100,
                         minibatch = 10)
    
    model.train(50000)
    #model.train_alternate(5000)
    
    # Even with 100 times the pred rate,
    # which leads to ~50 times larger weights,
    # the model collapses to two points...
    
    # To see if the model is training:
    # links: entropy of each latent variable
    # preds: discriminative accuracy
    
    # Occasionally restart particles?
    
    """
    # Cheating:
    # If we assign one dimension to each pred,
    # choose all the weights and initialise the latent vectors accordingly,
    # then this is stable
    # Note that the pred weights must be stronger (more than 3 times, for ditransitives)
    # The fantasy particles end up having all 0s for verbs and random vectors for nouns
    a = 5
    b = 20
    model.link_wei -= a
    model.pred_wei -= b
    import toy
    for i in range(model.V):
        model.pred_wei[i,i] = b
    for i, row in enumerate(toy.intr_sent):
        for j, value in enumerate(row):
            if value:
                model.link_wei[0, toy.I+i, j] = a
    for i, mat in enumerate(toy.tran_sent):
        for j, row in enumerate(mat):
            for k, value in enumerate(row):
                if value:
                    model.link_wei[0, toy.T+i, j] = a
                    model.link_wei[1, toy.T+i, k] = a
    for i, ten in enumerate(toy.ditr_sent):
        for j, mat in enumerate(ten):
            for k, row in enumerate(mat):
                for l, value in enumerate(row):
                    if value:
                        model.link_wei[0, toy.D+i, j] = a
                        model.link_wei[1, toy.D+i, k] = a
                        model.link_wei[2, toy.D+i, l] = a
    model.ents = zeros((model.N, model.D))
    for i in range(model.N):
        p = model.nodes[i].pred
        model.ents[i,p] = 1
    """
    
    
    """
    # Even setting the pred rate 10 times higher than the link rate
    # doesn't seem to force the model to distinguish preds:
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 25,
                         rate_link = 10**-5,
                         rate_pred = 10**-3,
                         l2_link = 1-10**-5,
                         l2_pred = 1-10**-4,
                         l1_link = 10**-7,
                         l1_pred = 10**-6,
                         init_range = 0.001)
    
    # Setting the pred rate 100 times higher seems to do something:
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 25,
                         rate_link = 10**-5,
                         rate_pred = 10**-3,
                         l2_link = 1-10**-5,
                         l2_pred = 1-10**-4,
                         l1_link = 10**-7,
                         l1_pred = 10**-6,
                         init_range = 0)
    
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
    
    # Currently, preds are trained discriminatively and links generatively
    # Use another set of weights for generation, and introduce a term for KL divergence or whatever? (cf Eng talk)