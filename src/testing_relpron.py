import os, pickle, gzip, numpy as np

from relpron_variational import load_scoring_fn, load_baseline_scoring_fn
from testing import get_test_relpron, get_relpron_separated, load_freq_lookup_dicts
from __config__.filepath import AUX_DIR
from utils import cosine

test_fn = get_test_relpron()
raw_items, term_to_props = get_relpron_separated()
lookup = load_freq_lookup_dicts()

terms = [lookup['n'][t] for t in term_to_props.keys()]
convert_which = {'SBJ': 1, 'OBJ': 2}
items = [(convert_which[which], i) for i, (which, _) in enumerate(raw_items)]

def get_scores(subdir='meanfield_relpron', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def cache_scores(input_dir='meanfield_relpron', output_dir='meanfield_relpron_cache_dev'):
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
        score_fn = load_scoring_fn(name)
        scores = {t:[score_fn(t,x) for x in items] for t in terms}
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

def add_scores(scores, subdir='meanfield_relpron'):
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
        score_fn = load_scoring_fn(string_name)
        scores[tuple_name] = test_fn(score_fn)
        print(tuple_name)

def add_scores_from_cache(scores, subdir='meanfield_relpron', cache_subdir='meanfield_relpron_cache_dev'):
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
        scores[tuple_name] = test_fn(score_fn)
        print(tuple_name)

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

cache_scores()
scores = get_scores()
add_scores_from_cache(scores)
save_scores(scores)

#score_fn = load_baseline_scoring_fn('multicore-5-400-0-1-0-0-32-1_0-40-0_01-1_0-4_0')
#test_fn(score_fn)

from testing import get_test_relpron_ensemble

test_ensemble = get_test_relpron_ensemble()

addition = addition_score_fn(load_vector_model())
semfunc = load_scoring_fn('multicore-5-400-0-1-0-0-32-1_0-30-0_01-1_0-4_0-0_5-0_2-0_5')

for x in np.arange(0, 0.5, 0.01):
    print(x, test_ensemble(semfunc, addition, x, basic_score_kwargs={'verb_weight': 0.9, 'head_weight':0.8}))
    print(x, test_ensemble(semfunc, addition, x, basic_score_kwargs={'verb_weight': 0.6, 'head_weight':1.5}))
