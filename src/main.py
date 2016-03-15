from numpy import array, random, tensordot, zeros, outer, float, arange, average, absolute, sign, minimum
from scipy.special import expit
from scipy.spatial.distance import cosine
from pydmrs.core import DictPointDmrs as Dmrs
from pydmrs.core import PointerNode as Node
from pydmrs.core import Link, LinkLabel
from pydmrs.components import RealPred, GPred
from copy import copy
from collections import Counter
import pickle
import warnings
from model import SemFuncModel, ToyTrainingSetup, ToyTrainer


if __name__ == "__main__":
    
    # Positive graphs
    """
    with open('../data/cathbaz/dmrs.pkl', 'rb') as f:
        dmrs, preds, links = pickle.load(f)
    """
    
    with open('../data/toy.pkl', 'rb') as f:
        intr_dmrs, tran_dmrs, ditr_dmrs = pickle.load(f)
    dmrs = intr_dmrs + tran_dmrs + ditr_dmrs
    freq = [len(intr_dmrs), len(tran_dmrs), len(ditr_dmrs)]
    neg_freq = [2,10,2]
    
    """
    with open('../data/tinytoy.pkl', 'rb') as f:
        dmrs = pickle.load(f)
    neg_freq = [3,0,0]
    """
    """
    with open('../data/trantoy.pkl', 'rb') as f:
        dmrs = pickle.load(f)
    neg_freq = [0,10,0]
    """
    
    # Negative graphs
    
    def empty_dmrs(i=0, num=1):
        assert num in [1,2,3]
        nodes = [Node(i+j, None) for j in range(num+1)]
        links = [Link(i, i+j+1, j, None) for j in range(num)]
        return Dmrs(nodes, links)
    
    neg_dmrs = []
    i = 0
    for j in [1,2,3]:
        for _ in range(neg_freq[j-1]):
            neg_dmrs.append(empty_dmrs(i, j))
            i += j+1
    
    # Preds
    
    max_pred = max(n.pred for g in dmrs for n in g.iter_nodes())
    pred_freq = zeros(max_pred+1)
    for g in dmrs:
        for n in g.iter_nodes():
            pred_freq[n.pred] += 1
    preds = list(range(max_pred+1))
    
    # Links
    
    max_link = max(l.rargname for g in dmrs for l in g.iter_links())
    links = list(range(max_link+1))
    
    # Set up model
    model = SemFuncModel(preds, links, pred_freq,
                         dims = 20,
                         card = 3,
                         bias = -5,
                         init_range = 1)
    setup = ToyTrainingSetup(model,
                         rate = 0.001,
                         rate_ratio = 1,
                         l2 = 100,
                         l2_ratio = 0.1,
                         l1 = 0.0001,
                         l1_ratio = 1)
    trainer = ToyTrainer(setup, dmrs, neg_dmrs,
                         neg_samples = 5)
    
    trainer.train(500,
                  minibatch = 10,
                  print_every = 1)
    """
    import cProfile
    cProfile.runctx('model.train(100)',globals(),locals())
    """