import os, pickle
from multiprocessing import Pool
from scipy.special import expit
from numpy import outer, zeros_like, zeros, array
from math import log

from pydmrs.components import RealPred
from utils import make_shared

D = 300
C = 30

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    preds = pickle.load(f)

pred_wei = make_shared(zeros((len(preds), D)))
with open('/anfs/bigdisc/gete2/wikiwoods/word2vec/matrix300', 'r') as f:
    for i, line in enumerate(f):
        pred, vecstr = line.strip().split(maxsplit=1)
        assert pred == preds[i]
        vec = array(vecstr.split())
        pred_wei[i] = vec
pred_wei *= 8

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-5'

pred_index = {RealPred.from_string(p):i for i,p in enumerate(preds)}

bias = log(D/C - 1)

def mean_vec(pred):
    vec = expit(pred_wei[pred_index[pred]] - bias)
    vec *= (30 / vec.sum())
    return vec

def summarise_triple(triple, matrix):
    v, a, p = triple
    v_mean = mean_vec(v)
    if a is not None:
        a_mean = mean_vec(a)
        matrix[0] += outer(v_mean, a_mean)
    if p is not None:
        p_mean = mean_vec(p)
        matrix[1] += outer(v_mean, p_mean)

def summarise(fname):
    print(fname)
    matrix = zeros((2, D, D))
    with open(os.path.join(DATA, fname), 'rb') as f:
        triples = pickle.load(f)
    for t in triples:
        summarise_triple(t, matrix)
    return matrix

all_files = os.listdir(DATA)
with Pool(50) as p:
    link_mat = zeros((2, D, D))
    for mat in p.imap_unordered(summarise, all_files):
        link_mat += mat

with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_0.pkl', 'wb') as f:
    pickle.dump(link_mat, f)

# Next step: given link weights,
# iterate to find mean field for each situation
# initialise as Above
# add in link weights, recalculate mean field - repeat till convergence
# also count total for each dimension