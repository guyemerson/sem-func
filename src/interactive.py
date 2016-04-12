import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=np.nan)

import pickle
with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/core-10000-1.pkl', 'rb') as f:
    setup = pickle.load(f)
model = setup.model
with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    pred_name = pickle.load(f)
ind = {p:i for i,p in enumerate(pred_name)}
def closest(pred, n=5):
    return [pred_name[x] for x in model.closest_preds([ind[pred]], n)[0]]

v_lookup = {}
n_lookup = {}
lookup = {'v':v_lookup, 'n':n_lookup}
for x in pred_name:
    lemma, pos, sense = x[1:].rsplit('_', 2)
    lookup[pos].setdefault(lemma, set()).add(x)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl', 'rb') as f:
    pred_freq = pickle.load(f)
freq_of = {pred_name[i]:n for i,n in enumerate(pred_freq)}

n_flookup = {w:max((freq_of[x], x) for x in n_lookup[w])[1] for w in n_lookup}
v_flookup = {w:max((freq_of[x], x) for x in v_lookup[w])[1] for w in v_lookup}
flookup = {'v':v_flookup, 'n':n_flookup}

simlex = []
with open('../data/SimLex-999/SimLex-999.txt') as f:
    f.readline()  # first line is headings
    for line in f:
        simlex.append(line.strip().split('\t'))
simlex_vocab = {x[i] for x in simlex for i in (0,1)}
simlex_nouns = {x[i] for x in simlex for i in (0,1) if x[2]=='N'}
simlex_verbs = {x[i] for x in simlex for i in (0,1) if x[2]=='V'}

with open('/anfs/bigdisc/gete2/wikiwoods/word2vec/model', 'rb') as f:
    wv = pickle.load(f)

def similarity_all(a, b, pos):
    scores = []
    freqs = []
    try:
        for p in lookup[pos][a]:
            for q in lookup[pos][b]:
                scores.append(wv.similarity(p,q))
                freqs.append(freq_of[p]*freq_of[q])
        totalscore = sum(scores[i] * freqs[i] for i in range(len(scores)))
        totalfreq = sum(freqs)
        return totalscore / totalfreq
    except KeyError:
        return 0

def similarity(a, b, pos):
    try:
        return wv.similarity(flookup[pos][a], flookup[pos][b])
    except KeyError:
        return 0

from scipy.stats import spearmanr
n_simlex = [x for x in simlex if x[2] == 'N']
v_simlex = [x for x in simlex if x[2] == 'V']
a_simlex = [x for x in simlex if x[2] == 'A']
n_scores = [x[3] for x in n_simlex]
v_scores = [x[3] for x in v_simlex]
a_scores = [x[3] for x in a_simlex]
n_sim = [similarity(x[0], x[1], 'n') for x in n_simlex]
v_sim = [similarity(x[0], x[1], 'v') for x in v_simlex]
n_sim_all = [similarity_all(x[0], x[1], 'n') for x in n_simlex]
v_sim_all = [similarity_all(x[0], x[1], 'v') for x in v_simlex]
print(spearmanr(n_scores, n_sim))
print(spearmanr(v_scores, v_sim))
print(spearmanr(n_scores, n_sim_all))
print(spearmanr(v_scores, v_sim_all))

for point in simlex:
    a, b, pos, score = point[:4]
    if pos == 'V':
        print(a, b, score, similarity(a,b,'v'))

counter_score = {}
with open('../data/SimLex-999/counter_ranking2.txt', 'r') as f:
    for line in f:
        n, w, c = line.split(':')
        a, b = w.strip().split(', ')
        counter_score[a,b] = float(c)
def counter_distance(a, b):
    return counter_score.get((a,b), 0)
n_counter = [counter_distance(x[0], x[1]) for x in n_simlex]
v_counter = [counter_distance(x[0], x[1]) for x in v_simlex]
a_counter = [counter_distance(x[0], x[1]) for x in a_simlex]
print(spearmanr(n_scores, n_counter))
print(spearmanr(v_scores, v_counter))
print(spearmanr(a_scores, a_counter))