import os, pickle
from multiprocessing import Pool
from scipy.special import expit
from numpy import outer, zeros_like, zeros, array
from math import log

from pydmrs.components import RealPred
from utils import make_shared, is_verb

D = 800
C = 40

half = int(D/2)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    preds = pickle.load(f)
ind = {p:i for i,p in enumerate(preds)}
pred_index = {RealPred.from_string(p):i for p,i in ind.items()}

pred_wei = make_shared(zeros((len(preds), D)))
for filename, offset in [('/anfs/bigdisc/gete2/wikiwoods/word2vec/matrix_nouns400', 0),
                         ('/anfs/bigdisc/gete2/wikiwoods/word2vec/matrix_verbs400', half)]:
    with open(filename, 'r') as f:
        for line in f:
            pred, vecstr = line.strip().split(maxsplit=1)
            vec = array(vecstr.split())
            pred_wei[ind[pred], offset:offset+half] = vec
# Make vectors longer (av. sum 1.138 over av. 44.9 nonzero entries)
# An average entry is then 0.2, so a predicate is expit(0.2*30 - 3) = 0.95 true
pred_wei *= 8

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-5'

bias = log(D/C - 1)

def mean_vec(pred):
    vec = expit(pred_wei[pred_index[pred]] - bias)
    if pred.pos == 'v':
        vec[:half] = 0
    else:
        vec[half:] = 0
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

if __name__ == "__main__":
    all_files = os.listdir(DATA)
    with Pool(20) as p:
        link_mat = zeros((2, D, D))
        for mat in p.imap_unordered(summarise, all_files):
            link_mat += mat
    
    with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_400.pkl', 'wb') as f:
        pickle.dump(link_mat, f)
    
    # Next step: given link weights,
    # iterate to find mean field for each situation
    # initialise as Above
    # add in link weights, recalculate mean field - repeat till convergence
    # also count total for each dimension