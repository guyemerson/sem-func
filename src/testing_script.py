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
    "Test semfunc model on all datasets"
    return test_all(setup.model.cosine_of_parameters, *a, **kw)

def load_and_test(prefix=prefix, thresh=thresh, name=name):
    "Load semfunc model and test it"
    with open(os.path.join(OUT_DIR, '{}-{}-{}.aux.pkl'.format(prefix,thresh,name)), 'rb') as f:
        aux = pickle.load(f)
        print(len(aux['completed_files']))
    with open(os.path.join(OUT_DIR, '{}-{}-{}.pkl'.format(prefix,thresh,name)), 'rb') as f:
        new = pickle.load(f)
    test_all_semfunc(new)
    return new

new = load_and_test()

# Simple vector models

def test_all_simplevec(vec, *a, **kw):
    "Test vector model on all datasets"
    def sim(a,b):
        "Cosine similarity of vectors"
        return cosine(vec[a], vec[b])
    return test_all(sim, *a, **kw)

def get_scores(subdir='simplevec'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, 'scores.pkl')
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_scores(scores, subdir='simplevec'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, 'scores.pkl')
    with open(score_file, 'wb') as f:
        pickle.dump(scores, f)

def add_scores(scores, subdir='simplevec', prefix=prefix, thresh=thresh, gz=True):
    "Add new scores"
    for filename in sorted(os.listdir(os.path.join(AUX_DIR, subdir))):
        # Ignore the summary file
        if filename is 'scores.pkl':
            continue
        # Process filename
        settings = tuple(filename.split('.')[0].split('-'))
        # Ignore files if we haven't loaded that version of the dataset
        if settings[0] != prefix or int(settings[1]) != thresh:
            continue
        # Ignore files we've already evaluated
        if settings in scores:
            continue
        # Load and test the file
        print(filename)
        if gz: open_fn = gzip.open
        else: open_fn = open
        with open_fn(os.path.join(AUX_DIR, subdir, filename), 'rb') as f:
            vec = pickle.load(f)
        scores[settings] = test_all_simplevec(vec)

def get_av_scores(scores, seed_index=5):
    "Average over random seeds"
    av_scores = {}
    # Group scores
    for settings, results in scores.items():
        # Get correlations (ignore significance values)
        cor_arr = np.array([cor for cor, _ in results])
        # Filter seed from settings
        nonseed_settings = settings[:seed_index] + settings[seed_index+1:]
        # Append correlations to list (initialising if necessary)
        av_scores.setdefault(nonseed_settings, []).append(cor_arr)
    # Average scores
    for s, arrs in av_scores.items():
        av_scores[s] = np.array(arrs).mean(0)
    return av_scores

def get_max(av_scores, pos, constr=()):
    """
    Get maximum score from all hyperparameter settings
    :param scores: dict mapping settings tuples to score tuples
    :param pos: position(s) out of tuple of scores to maximise (int or list/tuple of ints)
    :param constr: iterable of (index, value) pairs to constrain hyperparameters
    :return: best set of hyperparameters, and its scores
    """
    def key(s):
        "Give average score of considered positions, and only if constraints are satisfied"
        if any(s[i] != v for i,v in constr):
            return 0
        elif isinstance(pos, (list, tuple)):
            return sum(av_scores[s][p] for p in pos)
        else:
            return av_scores[s][pos]
    # Get the best hyperparameters
    best = max(av_scores, key=key)
    return best, av_scores[best][pos]

scores = get_scores()
add_scores(scores)
save_scores(scores)
av_scores = get_av_scores(scores)

for i in [0,1,2,4]:
    print(get_max(av_scores, i))
print(get_max(av_scores, [0,1,2,4]))

for i in [0,1,2,4]:
    print(get_max(av_scores, i, [(0, '400')]))
print(get_max(av_scores, [0,1,2,4], [(0, '400')]))


mf_scores = get_scores('meanfield')
add_scores(mf_scores, 'meanfield', gz=False)
save_scores(mf_scores, 'meanfield')
mf_av_scores = get_av_scores(mf_scores)

for i in [0,1,2,4]:
    print(get_max(mf_av_scores, i))
print(get_max(mf_av_scores, [0,1,2,4]))

