import pickle
import numpy as np

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    pred_name = pickle.load(f)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl', 'rb') as f:
    pred_freq = pickle.load(f)

ind = {p:i for i,p in enumerate(pred_name)}
freq = {p:pred_freq[i] for i,p in enumerate(pred_name)}

v_lookup = {}
n_lookup = {}
lookup = {'v':v_lookup, 'n':n_lookup}
for x in pred_name:
    lemma, pos, sense = x[1:].rsplit('_', 2)
    lookup[pos].setdefault(lemma, set()).add(x)

n_flookup = {w:max((freq[x], x) for x in n_lookup[w])[1] for w in n_lookup}
v_flookup = {w:max((freq[x], x) for x in v_lookup[w])[1] for w in v_lookup}
flookup = {'v':v_flookup, 'n':n_flookup}

simlex = []
with open('../data/SimLex-999/SimLex-999.txt') as f:
    f.readline()  # first line is headings
    for line in f:
        simlex.append(line.strip().split('\t'))

simlex_vocab = {x[i] for x in simlex for i in (0,1)}
simlex_nouns = {x[i] for x in simlex for i in (0,1) if x[2]=='N'}
simlex_verbs = {x[i] for x in simlex for i in (0,1) if x[2]=='V'}

verbs = [i for i,x in enumerate(pred_name) if x[1:].rsplit('_', 2)[1] == 'v']
nouns = [i for i,x in enumerate(pred_name) if x[1:].rsplit('_', 2)[1] == 'n']

from scipy.stats import spearmanr
n_simlex = [x for x in simlex if x[2] == 'N']
v_simlex = [x for x in simlex if x[2] == 'V']
a_simlex = [x for x in simlex if x[2] == 'A']
n_scores = [float(x[3]) for x in n_simlex]
v_scores = [float(x[3]) for x in v_simlex]
a_scores = [float(x[3]) for x in a_simlex]

from math import sqrt
def sim(u,v):
    return np.dot(u,v) / sqrt(np.dot(u,u) * np.dot(v,v)) 

def sparsim_all(a, b, pos, sparse_mat):
    scores = []
    freqs = []
    try:
        for p in lookup[pos][a]:
            for q in lookup[pos][b]:
                scores.append(sim(sparse_mat[ind[p]], sparse_mat[ind[q]]))
                freqs.append(freq[p]*freq[q])
        totalscore = sum(scores[i] * freqs[i] for i in range(len(scores)))
        totalfreq = sum(freqs)
        return totalscore / totalfreq
    except KeyError:
        return 0

def test(setup, verbose=True):
    n_mysim_all = np.nan_to_num([sparsim_all(x[0], x[1], 'n', sparse_mat=setup.model.pred_wei) for x in n_simlex])
    v_mysim_all = np.nan_to_num([sparsim_all(x[0], x[1], 'v', sparse_mat=setup.model.pred_wei) for x in v_simlex])
    scores = (spearmanr(n_scores, n_mysim_all), spearmanr(v_scores, v_mysim_all))
    if verbose:
        print(*scores, sep='\n')
    return scores

common_simlex = [x for x in n_simlex if freq.get(n_flookup.get(x[0]),0)>1000 and freq.get(n_flookup.get(x[1]),0)>1000]
common_scores = [float(x[3]) for x in common_simlex]
def test_common(setup, verbose=True):
    mysim_all = np.nan_to_num([sparsim_all(x[0], x[1], 'n', sparse_mat=setup.model.pred_wei) for x in common_simlex])
    scores = spearmanr(common_scores, mysim_all)
    if verbose:
        print(scores)
    return scores

ws_sim = []
with open('/homes/gete2/git/sem-func/data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt') as f:
    for line in f:
        ws_sim.append(line.strip().split('\t'))

ws_rel = []
with open('/homes/gete2/git/sem-func/data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt') as f:
    for line in f:
        ws_rel.append(line.strip().split('\t'))

ws_sim_scores = [x[2] for x in ws_sim]
ws_rel_scores = [x[2] for x in ws_rel]

def test_ws(setup, verbose=True):
    my_ws_sim = np.nan_to_num([sparsim_all(x[0], x[1], 'n', sparse_mat=setup.model.pred_wei) for x in ws_sim])
    my_ws_rel = np.nan_to_num([sparsim_all(x[0], x[1], 'n', sparse_mat=setup.model.pred_wei) for x in ws_rel])
    scores = (spearmanr(ws_sim_scores, my_ws_sim), spearmanr(ws_rel_scores, my_ws_rel))
    if verbose:
        print(*scores, sep='\n')
    return scores

def test_all(setup, verbose=True):
    s1,s2 = test(setup, verbose=verbose)
    s3,s4 = test(setup, verbose=verbose)
    return (s1,s2,s3,s4)


def scores_pos(old, new, pos):
    if pos == 'n':
        w_simlex = n_simlex
    elif pos == 'v':
        w_simlex = v_simlex
    else:
        raise Exception
    old_scores = [sparsim_all(x[0], x[1], pos, sparse_mat=old.model.pred_wei) for x in w_simlex]
    new_scores = [sparsim_all(x[0], x[1], pos, sparse_mat=new.model.pred_wei) for x in w_simlex]
    return np.array(old_scores), np.array(new_scores), w_simlex

def compare(old, new, n=20, which='nv', direction='both'):
    old_scores = np.empty(0)
    new_scores = np.empty(0)
    w_simlex = []
    for pos in which:
        extra_old, extra_new, extra_simlex = scores_pos(old, new, pos)
        old_scores = np.concatenate([old_scores, extra_old])
        new_scores = np.concatenate([new_scores, extra_new])
        w_simlex.extend(extra_simlex)
    diffs = new_scores - old_scores
    if direction == 'pos':
        np.clip(diffs, 0, np.inf, out=diffs)
    elif direction == 'neg':
        np.clip(diffs, -np.inf, 0, out=diffs)
    diffs = np.abs(diffs)
    for i in diffs.argsort()[-n:]:
        print(w_simlex[i][0], w_simlex[i][1])
        print(old_scores[i])
        print(new_scores[i])


def mat_sim(a,b):
    return (a * b).sum() / sqrt((a ** 2).sum() * (b ** 2).sum())

def similarity(a, b, wv, pos):
    try:
        return wv.similarity(flookup[pos][a], flookup[pos][b])
    except KeyError:
        return 0

def similarity_all(a, b, wv, pos):
    scores = []
    freqs = []
    try:
        for p in lookup[pos][a]:
            for q in lookup[pos][b]:
                scores.append(wv.similarity(p,q))
                freqs.append(freq[p]*freq[q])
        totalscore = sum(scores[i] * freqs[i] for i in range(len(scores)))
        totalfreq = sum(freqs)
        return totalscore / totalfreq
    except KeyError:
        return 0

def test_w2v(w2v_model, verbose=True):
    n_mysim = np.nan_to_num([similarity_all(x[0], x[1], w2v_model, 'n') for x in n_simlex])
    v_mysim = np.nan_to_num([similarity_all(x[0], x[1], w2v_model, 'n') for x in v_simlex])
    scores = (spearmanr(n_scores, n_mysim), spearmanr(v_scores, v_mysim))
    if verbose:
        print(scores)
    return scores

def test_common_w2v(w2v_model, verbose=True):
    mysim = np.nan_to_num([similarity_all(x[0], x[1], w2v_model, 'n') for x in common_simlex])
    scores = spearmanr(common_scores, mysim)
    if verbose:
        print(scores)
    return scores
