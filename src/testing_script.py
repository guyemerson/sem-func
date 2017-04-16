import os, pickle, gzip, numpy as np, operator
from itertools import product

from testing import get_test_all, get_test_preds, get_test_relpron_hypernym
from __config__.filepath import AUX_DIR, OUT_DIR
from utils import cosine
from simplevec_to_entity import get_verb_noun_freq
from variational import marginal_approx, get_semfunc

prefix = 'multicore'
thresh = 5
name = 'example'

test_all = get_test_all(prefix, thresh)
test_relpron_hypernym = get_test_relpron_hypernym(prefix, thresh)

preds, _ = get_test_preds(prefix, thresh)
pred_list = sorted(preds)
freq = get_verb_noun_freq(prefix, thresh, pred_list)

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

#new = load_and_test()

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
        op = operator.add
    def sim(a,b):
        "Combined truth of a=>b and b=>a"
        return op(sf[a](vec[b]), sf[b](vec[a]))
    return test_all(sim, **kw)

def test_hypernym(vec, sf, **kw):
    """
    Test semfunc model on finding hypernyms
    :param vec: mean field posterior vectors
    :param sf: semantic functions
    :param **kw: additional keyword arguments passed to test_relpron_hypernym function
    :return: mean average precision results
    """
    def implies(a,b):
        "Probability that the truth of a implies the truth of b"
        return sf[b](vec[a])
    return test_relpron_hypernym(implies, **kw)

def test_hypernym_reverse(vec, sf, **kw):
    """
    Test semfunc model on finding hypernyms
    :param vec: mean field posterior vectors
    :param sf: semantic functions
    :param **kw: additional keyword arguments passed to test_relpron_hypernym function
    :return: mean average precision results
    """
    def implies(a,b):
        "Probability that the truth of b implies the truth of a"
        return sf[a](vec[b])
    return test_relpron_hypernym(implies, **kw)

def implies_self(vec, sf):
    "Test each semantic function on its own meanfield vector"
    prob = [sf[p](vec[p]) for p in pred_list]
    av = sum(prob)/len(prob)
    print(av)
    return av

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

def add_simple_scores(scores, subdir='simplevec', length=7, pred_list=pred_list, constr=(), test_fn=test_all_implies, **test_fn_kwargs):
    "Add new scores"
    # Process each file
    for filename in sorted(os.listdir(os.path.join(AUX_DIR, subdir))):
        # Process filename
        name = filename.split('.')[0]
        settings = tuple(name.split('-'))
        # Ignore auxiliary files
        if len(settings) != length:
            continue
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
        with gzip.open(os.path.join(AUX_DIR, subdir, filename), 'rb') as f:
            vectors = pickle.load(f)
        # Test
        scores[settings] = test_fn(vectors, **test_fn_kwargs)

def add_semfunc_scores(scores, param_subdir='simplevec', meanfield_subdir='meanfield_freq', basic_length=7, full_length=11, C_index=8, pred_list=pred_list, constr=(), test_fn=test_all_implies, **test_fn_kwargs):
    "Add new scores"
    # Process each file
    prev_basic_name = None  # Avoid reloading vectors
    for filename in sorted(os.listdir(os.path.join(AUX_DIR, meanfield_subdir))):
        # Process filename
        name = filename.split('.')[0]
        settings = tuple(name.split('-'))
        # Ignore auxiliary files
        if len(settings) != full_length:
            continue
        # Ignore files if we haven't loaded that version of the dataset
        if settings[0] != prefix or int(settings[1]) != thresh:
            continue
        # Ignore files we've already evaluated
        if settings in scores:
            continue
        # Ignore files that don't match the given constraints:
        if any(settings[pos] not in [str(x).replace('.','') for x in val] for pos,val in constr):
            continue
        # Extract information from filename
        basic_settings = settings[:basic_length]
        basic_name = '-'.join(basic_settings)
        C = int(settings[C_index])
        # Load the parameter vectors
        print(filename)
        if basic_name != prev_basic_name:
            with gzip.open(os.path.join(AUX_DIR, param_subdir, basic_name+'.pkl.gz'), 'rb') as f:
                param_vec = pickle.load(f)
            prev_basic_name = basic_name
        # Load the biases
        with gzip.open(os.path.join(AUX_DIR, meanfield_subdir, name+'-bias.pkl.gz'), 'rb') as f:
            bias = pickle.load(f)
        # Construct the semantic functions
        semfuncs = {i:get_semfunc(param_vec[i], bias[i]) for i in bias}
        # Load the entity vectors
        with gzip.open(os.path.join(AUX_DIR, meanfield_subdir, filename), 'rb') as f:
            meanfield_vec = pickle.load(f)
        marginal_vec = {i: marginal_approx(p, C) for i,p in meanfield_vec.items()}
        # Test
        scores[settings] = test_fn(marginal_vec, semfuncs, **test_fn_kwargs)

def get_av_scores(scores, index=7):
    "Average over random seeds"
    av_scores = {}
    # Allow either a single index or a list of indices
    if isinstance(index, int):
        index = [index]
    else:
        index = sorted(index, reverse=True)
    # Group scores
    for settings, results in scores.items():
        # Get correlations (ignore significance values)
        if isinstance(results, (list, tuple)):
            cor_arr = np.array([cor for cor, _ in results])
        else:
            cor_arr = results
        # Filter seed from settings
        nonseed_settings = settings
        for i in index:
            nonseed_settings = nonseed_settings[:i] + nonseed_settings[i+1:]
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
        elif pos is None:
            return av_scores[s][1]
        elif isinstance(pos, (list, tuple)):
            return sum(av_scores[s][1][p] for p in pos)
        else:
            return av_scores[s][1][pos]
    # Get the best hyperparameters
    best = max(av_scores, key=key)
    if pos is None:
        return best, av_scores[best]
    else:
        b = av_scores[best]
        return best, (b[0], b[1][pos])

def harm(x,y):
    """Harmonic mean of x and y"""
    return x*y/(x+y)

def prop(x,y):
    """Combine P(a|b) and P(b|a) to P(a and b|a or b)"""
    return 1 / (1/x + 1/y - 1)

"""
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
""
self_prob = get_scores('meanfield_freq', 'self_prob')
add_semfunc_scores(self_prob, test_fn=implies_self)
save_scores(self_prob, 'meanfield_freq', 'self_prob')
av_self_prob = get_av_probs(self_prob)

""

for op_name, op in [('add', operator.add),
                    ('mul', operator.mul),
                    ('min', min),
                    ('max', max),
                    ('harm', harm),
                    ('prop', prop)]:
    score_filename = 'scores_'+op_name
    scores_op = get_scores('meanfield', score_filename)
    add_scores(scores_op, 'meanfield', method='implies', op=op,
               constr=[(3, [0]),
                       (4, [0.6, 0.7, 0.75, 0.8]),
                       (6, [0.8, 1, 1.2]),
                       (7, [40, 80])])
    save_scores(scores_op, 'meanfield', score_filename)
    
    av_scores = get_av_scores(scores_op)
    for i in [0,1,2,4]:
        print(get_max(av_scores, i))
    print(get_max(av_scores, [0,1,2,4]))
"""

#subdir = '/local/scratch/gete2/variational/meanfield_freq'
#semfunc_subdir = '/local/scratch/gete2/variational/simplevec'
#basic_length=6
#subdir = 'meanfield_freq_card'
#semfunc_subdir = 'simplevec_card'
#basic_length = 7

meanfield_subdir = 'meanfield'
basic_length = 8
kwargs = {
    'param_subdir': 'simplevec',
    'meanfield_subdir': meanfield_subdir,
    'basic_length': basic_length,
    'full_length': 13,
    'C_index': 9
}
"""
meanfield_subdir = '/local/scratch/gete2/variational/meanfield_freq'
basic_length = 6
kwargs = {
    'param_subdir': '/local/scratch/gete2/variational/simplevec',
    'meanfield_subdir': meanfield_subdir,
    'basic_length': basic_length,
    'full_length': 10,
    'C_index': 7
}
"""
#meanfield_subdir = 'meanfield_freq'
#basic_length = 7
#full_length = 11
"""
for op_name, op in [('mul', operator.mul),
                    ('add', operator.add),
                    ('min', min),
                    ('max', max),
                    ('harm', harm),
                    ('prop', prop)]:
    score_filename = 'scores_'+op_name
    scores = get_scores(meanfield_subdir, score_filename)
    #add_semfunc_scores(scores, op=op, **kwargs)
    #save_scores(scores, meanfield_subdir, score_filename)
    
    av_scores = get_av_scores(scores, kwargs['basic_length']-1)
    print(op_name, 'best:')
    for i in [0,1,2,4,5,6,7,8]:
        print(get_max(av_scores, i))
    print(get_max(av_scores, [0,2,4,5]))


"""
scores = get_scores(meanfield_subdir, 'scores_mul')
add_semfunc_scores(scores, op=operator.mul, **kwargs)
save_scores(scores, meanfield_subdir, 'scores_mul')
_scores = get_av_scores(scores, 0)
for x in ['8','32','64','91','97']:
    for i in [0,1,2,4,5]:
        print(get_max(_scores, i, [(6, x)]))
    print(get_max(_scores, [0,2,4,5], [(6, x)]))
    print(get_max(_scores, [0,2,4,5], [(6, x), (9, '0_01')]))
    print(get_max(_scores, [0,2,4,5], [(6, x), (11, '4')]))
    print(get_max(_scores, [0,2,4,5], [(6, x), (9, '0_01'), (11, '4')]))

"""
scores = get_scores(meanfield_subdir, 'scores_hyper')
add_semfunc_scores(scores, test_fn=test_hypernym, **kwargs)
save_scores(scores, meanfield_subdir, 'scores_hyper')
_scores = get_av_scores(scores, 0)
for x in ['8','32','64','91','97']:
    print(get_max(_scores, None, [(6, x)]))
"""
scores = get_scores(meanfield_subdir, 'scores_hyper_reverse')
add_semfunc_scores(scores, test_fn=test_hypernym_reverse, **kwargs)
save_scores(scores, meanfield_subdir, 'scores_hyper_reverse')
_scores = get_av_scores(scores, 0)
for x in ['8','32','64','91','97']:
    print(get_max(_scores, None, [(6, x)]))

self_prob = get_scores(meanfield_subdir, 'self_prob')
add_semfunc_scores(self_prob, test_fn=implies_self, **kwargs)
save_scores(self_prob, meanfield_subdir, 'self_prob')
av_self_prob = get_av_scores(self_prob)

"""
av_scores = get_av_scores(scores, basic_length-1)

av_av = get_av_scores(scores, [basic_length-1, kwargs['full_length']-1])

hyper_range = [None, None, None,
               [-1.5, -1, -0.6, -0.3, 0],
               [0.8, 1],
               [0, 0.1, 1, 3],
               None,
               [0.8, 1, 1.2],
               [30, 40, 50],
               [10., 1., 0.1, 0.01, 0.001, 0.0001],
               [0, 0.6, 0.8, 1],]
#               [1, 4, 16]]
for pos, x_range in enumerate(hyper_range):
    if x_range is None: continue
    print('\n', pos, '\n')
    for x in x_range:
        print(x)
        str_x = str(x).replace('-','~').replace('.','_')
        #for i in [0,1,2,4,5,6,7,8]: 
        #    print(get_max(av_av, i, [(pos, str_x)]))
        print(get_max(av_av, None, [(pos, str_x)]))

for pos in range(3,kwargs['full_length']):
    rest = list(range(kwargs['full_length']))
    rest.pop(pos)
    av_rest = get_av_scores(scores, rest)
    print(pos)
    for val, score in av_rest.items():
        print(val, score)
        

""

for k, C, Z in product(['0','~0_3'],['30','40','50'],['0_01','0_001','0_0001']):
    print(k, C, Z)
    try:
        print(av_av['multicore', '5', '400', k, '1', '0', '0', '1', C, Z, '1'])
    except KeyError:
        print('...')
"""

