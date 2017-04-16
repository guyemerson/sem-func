import os, pickle, gzip, numpy as np

from relpron_variational import load_scoring_fn, load_baseline_scoring_fn
from testing import get_test_relpron, get_test_relpron_ensemble, get_relpron_separated, load_freq_lookup_dicts
from __config__.filepath import AUX_DIR
from utils import cosine

test_fn = {False: get_test_relpron(),
           True: get_test_relpron(testset=True)}
test_ensemble = {False: get_test_relpron_ensemble(),
                 True: get_test_relpron_ensemble(testset=True)}

lookup = load_freq_lookup_dicts()
raw_items_dev, term_to_props_dev = get_relpron_separated()
raw_items_test, term_to_props_test = get_relpron_separated(True)
raw_items = {False: raw_items_dev, True: raw_items_test}
term_to_props = {False: term_to_props_dev, True: term_to_props_test}

terms = {testset:[lookup['n'][t] for t in t2p.keys()]
         for testset, t2p in term_to_props.items()}
convert_which = {'SBJ': 1, 'OBJ': 2}
items = {testset:[(convert_which[which], i) for i, (which, _) in enumerate(ri)]
         for testset, ri in raw_items.items()} 

def get_scores(subdir='meanfield_relpron', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def cache_scores(input_dir='meanfield_relpron', output_dir='meanfield_relpron_cache_dev', test=False):
    "Cache scores for each term and item"
    cached = set(os.listdir(os.path.join(AUX_DIR, output_dir)))
    new = set(os.listdir(os.path.join(AUX_DIR, input_dir)))
    for filename in new - cached:
        # Skip score files
        if filename[-3:] != '.gz':
            continue
        # Get scores
        name = filename.split('.')[0]
        print(name)
        score_fn = load_scoring_fn(name, meanfield_dir=input_dir)
        scores = {t:[score_fn(t,x) for x in items[test]] for t in terms[test]}
        # Save scores
        with gzip.open(os.path.join(AUX_DIR, output_dir, filename), 'wb') as f:
            pickle.dump(scores, f)

def convert_name(name):
    "Convert a hyphen-separated name to a tuple of values"
    string_parts = [p.replace('_','.').replace('~','-') for p in name.split('-')]
    parts = []
    for p in string_parts:
        if p.isalpha():
            parts.append(p)
        elif '.' in p:
            parts.append(float(p))
        else:
            parts.append(int(p))
    return tuple(parts)

def convert_settings(settings):
    "Convert a tuple of values to a hyphen-separated name"
    return '-'.join(str(x).replace('-','~').replace('.','_') for x in settings)

def add_scores(scores, subdir='meanfield_relpron', testset=False):
    "Add newly calculated scores"
    for filename in os.listdir(os.path.join(AUX_DIR, subdir)):
        # Skip score files
        if filename[-3:] != '.gz':
            continue
        # Skip files we've already processed
        string_name = filename.split('.')[0]
        tuple_name = convert_name(string_name)
        if tuple_name in scores:
            continue
        # Get score
        score_fn = load_scoring_fn(string_name, meanfield_dir=subdir)
        scores[tuple_name] = test_fn[testset](score_fn)
        print(tuple_name)

def add_scores_from_cache(scores, subdir='meanfield_relpron', cache_subdir='meanfield_relpron_cache_dev', testset=False):
    "Add newly calculated scores from cache"
    for filename in os.listdir(os.path.join(AUX_DIR, subdir)):
        # Skip score files
        if filename[-3:] != '.gz':
            continue
        # Skip files we've already processed
        tuple_name = convert_name(filename.split('.')[0])
        if tuple_name in scores:
            continue
        # Get score
        with gzip.open(os.path.join(AUX_DIR, cache_subdir, filename), 'rb') as f:
            cache = pickle.load(f)
        def score_fn(term, description):
            "Fetch a score from the cache"
            _, index = description
            return cache[term][index]
        new_score = test_fn[testset](score_fn)
        scores[tuple_name] = new_score 
        print(tuple_name)
        print(new_score)

def save_scores(scores, subdir='meanfield_relpron', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    with open(score_file, 'wb') as f:
        pickle.dump(scores, f)

def load_vector_model(filename='embeddings.500.npy', vocab_filename='vocab.txt', subdir='word2vec'):
    "Load a vector model"
    # Load vectors
    vec = np.load(os.path.join(AUX_DIR, subdir, filename))
    # Load vocab
    model = {}
    with open(os.path.join(AUX_DIR, subdir, vocab_filename)) as f:
        for i, line in enumerate(f):
            token, _ = line.split('\t')
            model[token] = vec[i]
    return model

def addition_score_fn(vec):
    "Define a score function using vector addition and cosine similarity"
    def score_fn(term, description, verb_weight=1, head_weight=1):
        "Score using vector addition and cosine similarity"
        which, triple = description
        if which == 'SBJ':
            weights = [verb_weight, head_weight, 1]
        elif which == 'OBJ':
            weights = [verb_weight, 1, head_weight]
        else:
            raise ValueError(which)
        combined = sum(vec[token] * wei for token, wei in zip(triple, weights))
        target = vec[term]
        return cosine(target, combined)
    return score_fn

if __name__ == "__main__":
    MEANFIELD_DIR = 'meanfield_relpron_test'
    CACHE_DIR = 'meanfield_relpron_cache_test'
    TEST = True
    
    cache_scores(MEANFIELD_DIR, CACHE_DIR, TEST)
    scores = get_scores(MEANFIELD_DIR)
    add_scores_from_cache(scores, MEANFIELD_DIR, CACHE_DIR, TEST)
    save_scores(scores, MEANFIELD_DIR)
    
    #score_fn = load_baseline_scoring_fn('multicore-5-400-0-1-0-0-32-1_0-40-0_01-1_0-4_0')
    #test_fn[TEST](score_fn)
    
    
    # Ensemble
    
    addition = addition_score_fn(load_vector_model())
    
    """
    sorted_settings = sorted(scores, key=lambda x:scores[x])
    top_settings = sorted_settings[-6:]
    for s in top_settings:
        print(s)
    top_semfunc = [load_scoring_fn(convert_settings(s), meanfield_dir=MEANFIELD_DIR) for s in top_settings]
    """
    
    top_semfunc = [load_scoring_fn('multicore-5-400-0-1-0-0-{}-1_0-30-0_01-1_0-4_0-0_5-0_2-0_8'.format(s), meanfield_dir=MEANFIELD_DIR)
                   for s in [8, 32, 64, 91, 97]]
    
    from itertools import product
    
    print(test_ensemble[TEST](top_semfunc, [addition], 1))
    
    
    for ratio, verb_wei, head_wei in product(np.arange(0.17, 0.211, 0.01),
                                             np.arange(0.7, 1.11, 0.1),
                                             np.arange(0.7, 1.11, 0.1)):
        print(ratio, verb_wei, head_wei)
        score = test_ensemble[TEST](top_semfunc, [addition], ratio, direct_score_kwargs={'verb_weight': verb_wei, 'head_weight': head_wei})
        print(score)
    
    """
    ensemble_scores = get_scores(MEANFIELD_DIR, 'scores_ensemble')
    
    prev_settings = None
    for settings, ratio, verb_wei, head_wei in product(sorted_settings[-1:],
                                                       np.arange(0.22, 0.251, 0.01),
                                                       np.arange(0.8, 1.01, 0.1),
                                                       np.arange(0.8, 1.01, 0.1)):
        full_settings = settings + (ratio, verb_wei, head_wei)
        if full_settings in ensemble_scores:
            continue
        print(settings)
        print(ratio, verb_wei, head_wei)
        if prev_settings != settings:
            semfunc = load_scoring_fn(convert_settings(settings), meanfield_dir=MEANFIELD_DIR)
        prev_settings = settings
        score = test_ensemble[TEST]([semfunc], [addition], ratio, direct_score_kwargs={'verb_weight': verb_wei, 'head_weight': head_wei})
        print(score)
        ensemble_scores[full_settings] = score
    
    save_scores(ensemble_scores, MEANFIELD_DIR, 'scores_ensemble')
    """
