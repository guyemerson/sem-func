import os, pickle, gzip, numpy as np

from testing import get_test_all
from __config__.filepath import AUX_DIR, OUT_DIR
from utils import cosine
from simplevec_to_entity import get_semfuncs_from_vectors
from variational import marginal_approx

prefix = 'multicore'
thresh = 5
name = 'example'

test_all = get_test_all(prefix, thresh)

# Semfunc model

def test_all_semfunc(setup, **kw):
    "Test semfunc model on all datasets"
    return test_all(setup.model.cosine_of_parameters, **kw)

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

def test_all_simplevec(vec, **kw):
    "Test vector model on all datasets"
    def sim(a,b):
        "Cosine similarity of vectors"
        return cosine(vec[a], vec[b])
    return test_all(sim, **kw)

def test_all_implies(vec, sf, op=None, **kw):
    """
    Test semfunc model on all datasets
    :param vec: mean field posterior vectors
    :param sf: semantic functions
    :param op: how to combine the implications
    :param **kw: additional keyword arguments passed to test_all function
    :return: Spearman rank correlation results
    """
    if op is None:
        import operator
        op = operator.add
    def sim(a,b):
        "Combined truth of a=>b and b=>a"
        return op(sf[a](vec[b]), sf[b](vec[a]))
    return test_all(sim, **kw)

def get_scores(subdir='simplevec', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_scores(scores, subdir='simplevec', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    with open(score_file, 'wb') as f:
        pickle.dump(scores, f)

def float_with_point(s):
    "Convert a setting from a string to a float (between 0 and 10)"
    return float(s[0]+'.'+s[1:])

def add_scores(scores, subdir='simplevec', prefix=prefix, thresh=thresh, method='simple', constr=(), **kw):
    "Add new scores"
    for filename in sorted(os.listdir(os.path.join(AUX_DIR, subdir))):
        # Ignore summary files
        if filename.split('.')[-1] != 'gz':
            continue
        # Process filename
        settings = tuple(filename.split('.')[0].split('-'))
        # Ignore files if we haven't loaded that version of the dataset
        if settings[0] != prefix or int(settings[1]) != thresh:
            continue
        # Ignore files we've already evaluated
        if settings in scores:
            continue
        # Ignore files that don't match the given constraints:
        if any(settings[pos] not in [str(x).replace('.','') for x in val] for pos,val in constr):
            continue
        # Load the vectors
        print(filename)
        with gzip.open(os.path.join(AUX_DIR, subdir, filename), 'rb') as f:
            vec = pickle.load(f)
        # Test according to chosen method
        if method == 'simple':
            scores[settings] = test_all_simplevec(vec, **kw)
        elif method == 'implies':
            name = '-'.join(settings[:6])
            scale, C, target = settings[6:]
            scale = float_with_point(scale)
            C = int(C)
            target = float(target)
            vec = {i: marginal_approx(p, C) for i,p in vec.items()}
            sf = get_semfuncs_from_vectors(name, scale, C, target, as_dict=True)
            scores[settings] = test_all_implies(vec, sf, **kw)

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
        av_scores[s] = (len(arrs), np.array(arrs).mean(0))
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
add_scores(mf_scores, 'meanfield')
save_scores(mf_scores, 'meanfield')
mf_av_scores = get_av_scores(mf_scores)

for i in [0,1,2,4]:
    print(get_max(mf_av_scores, i))
print(get_max(mf_av_scores, [0,1,2,4]))


import operator
for op_name, op in [('add', operator.add),
                    ('mul', operator.mul),
                    ('min', min),
                    ('max', max)]:
    score_filename = 'scores_'+op_name
    scores_op = get_scores('meanfield', score_filename)
    add_scores(scores_op, 'meanfield', method='implies', op=op,
               constr=[(3, [0]),
                       (4, [0.6, 0.7, 0.75, 0.8]),
                       (6, [0.8, 1, 1.2]),
                       (7, [40, 80])])
    save_scores(scores_op, 'meanfield', score_filename)
    
    av_scores_op = get_av_scores(scores_op)
    for i in [0,1,2,4]:
        print(get_max(av_scores_op, i))
    print(get_max(av_scores_op, [0,1,2,4]))
