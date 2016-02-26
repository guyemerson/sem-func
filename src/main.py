from numpy import array, random, tensordot, zeros, outer, float, arange, average, absolute, sign, minimum
from scipy.special import expit
from scipy.spatial.distance import cosine
from pydmrs.core import DictPointDmrs as Dmrs
from pydmrs.core import PointerNode as Node
from pydmrs.core import RealPred, GPred, Link, LinkLabel
from copy import copy
import pickle
import warnings
from model import SemFuncModel


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
    
    # Set up model
    model = SemFuncModel(dmrs, neg_dmrs,
                         dims = 20,
                         card = 3,
                         rate = 0.01,
                         rate_ratio = 1,
                         l2 = 0.0001,
                         l2_ratio = 10,
                         l1 = 0.000001,
                         l1_ratio = 1,
                         init_range = 1,
                         print_every = 10,
                         minibatch = 10,
                         bias = -5,
                         neg_samples = 5)
    
    model.train(50000)
    """
    import cProfile
    cProfile.runctx('model.train(100)',globals(),locals())
    """