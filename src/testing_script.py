import os, pickle, gzip, numpy as np

from testing import get_test_all
from __config__.filepath import AUX_DIR, OUT_DIR
from utils import cosine

prefix = 'multicore'
thresh = 5
name = 'example'

test_all = get_test_all(prefix, thresh)

# Semfunc model

def test_all_semfunc(setup, *a, **kw):
    return test_all(setup.model.cosine_of_parameters, *a, **kw)

with open(os.path.join(OUT_DIR, '{}-{}-{}.aux.pkl'.format(prefix,thresh,name)), 'rb') as f:
    aux = pickle.load(f)
    print(len(aux['completed_files']))
with open(os.path.join(OUT_DIR, '{}-{}-{}.pkl'.format(prefix,thresh,name)), 'rb') as f:
    new = pickle.load(f)

test_all_semfunc(new, False)


# Simple vector models

def test_all_simplevec(vec, *a, **kw):
    def sim(a,b):
        return cosine(vec[a], vec[b])
    return test_all(sim, *a, **kw)

score_file = os.path.join(AUX_DIR, 'simplevec', 'scores.pkl')
try:
    with open(score_file, 'rb') as f:
        scores = pickle.load(f)
except FileNotFoundError:
    scores = {}

for filename in sorted(os.listdir(os.path.join(AUX_DIR, 'simplevec'))):
    parts = filename.split('-')
    if len(parts) != 6 or parts[0] != prefix or int(parts[1]) != thresh:
        continue
    settings = tuple(filename.split('.')[0].split('-'))
    if settings in scores:
        continue
    print(filename)
    with gzip.open(os.path.join(AUX_DIR, 'simplevec', filename), 'rb') as f:
        vec = pickle.load(f)
    scores[settings] = test_all_simplevec(vec)

with open(score_file, 'wb') as f:
    pickle.dump(scores, f)

av_scores = {}
for settings, results in scores.items():
    res_arr = np.array([cor for cor,sig in results])
    av_scores.setdefault(settings[:-1], []).append(res_arr)
for s, arrs in av_scores.items():
    av_scores[s] = np.array(arrs).mean(0)

def get_max(pos, constr=()):
    def key(s):
        if any(s[i] != v for i,v in constr):
            return 0
        elif isinstance(pos, (list, tuple)):
            return sum(av_scores[s][p] for p in pos)
        else:
            return av_scores[s][pos]
    best = max(av_scores, key=key)
    return best, av_scores[best][pos]

for i in [0,1,2,4]:
    print(get_max(i))
print(get_max([0,1,2,4]))

for i in [0,1,2,4]:
    print(get_max(i, [(0, '400')]))
print(get_max([0,1,2,4], [(0, '400')]))
