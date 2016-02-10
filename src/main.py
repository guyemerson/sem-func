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
    with open('../data/cathbaz/dmrs.pkl', 'rb') as f:
        dmrs, preds, links = pickle.load(f)
    """
    with open('../data/toy.pkl', 'rb') as f:
        intr_dmrs, tran_dmrs, ditr_dmrs = pickle.load(f)
    dmrs = intr_dmrs + tran_dmrs + ditr_dmrs
    freq = [len(intr_dmrs), len(tran_dmrs), len(ditr_dmrs)]
    """
    """
    with open('../data/tinytoy.pkl', 'rb') as f:
        dmrs = pickle.load(f)
    """
    
    # Negative graphs
    
    def empty_dmrs(i=0, num=1):
        assert num in [1,2,3]
        nodes = [Node(i+j, None) for j in range(num+1)]
        links = [Link(i, i+j+1, j, None) for j in range(num)]
        return Dmrs(nodes, links)
    
    neg_freq = [2,10,2]
    #neg_freq = [3,0,0]
    
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
                         rate_link = 1*10**-3,
                         rate_pred = 2*10**-2,
                         l2_link = 1-1*10**-3,
                         l2_pred = 1-3*10**-3,
                         l1_link = 10**-7,
                         l1_pred = 10**-7,
                         init_range = 0,
                         print_every = 1,
                         minibatch = 10)
    
    model.train(50000)