import os, pickle

from testing import get_freq_lookup_dicts, get_simlex, get_wordsim, evaluate, with_lookup
from __config__.filepath import AUX_DIR, OUT_DIR

prefix = 'multicore'
thresh = 5
name = 'example'

with open(os.path.join(AUX_DIR, '{}-{}-vocab.pkl'.format(prefix,thresh)), 'rb') as f:
    pred_name = pickle.load(f)
with open(os.path.join(AUX_DIR, '{}-{}-freq.pkl'.format(prefix,thresh)), 'rb') as f:
    pred_freq = pickle.load(f)

freq_lookup = get_freq_lookup_dicts(pred_name, pred_freq)

simlex = get_simlex()
wordsim = get_wordsim()

simlex_common = ([], [])
for pair, score in zip(*simlex[0]):
    if all(pred_freq[freq_lookup['n'].get(x,0)] > 1000 for x in pair):
        simlex_common[0].append(pair)
        simlex_common[1].append(score)

def test_all(setup):
    n_sim = with_lookup(setup.model.cosine_of_parameters, freq_lookup['n'])
    v_sim = with_lookup(setup.model.cosine_of_parameters, freq_lookup['v'])
    print('noun:', evaluate(n_sim, *simlex[0]))
    print('verb:', evaluate(v_sim, *simlex[1]))
    print('sim:', evaluate(n_sim, *wordsim[0]))
    print('rel:', evaluate(n_sim, *wordsim[1]))
    print('common:', evaluate(n_sim, *simlex_common))

with open(os.path.join(OUT_DIR, '{}-{}-{}.aux.pkl'.format(prefix,thresh,name)), 'rb') as f:
    aux = pickle.load(f)
    print(len(aux['completed_files']))
with open(os.path.join(OUT_DIR, '{}-{}-{}.pkl'.format(prefix,thresh,name)), 'rb') as f:
    new = pickle.load(f)

test_all(new)
