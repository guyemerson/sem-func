import os, pickle

from relpron_variational import load_scoring_fn, load_baseline_scoring_fn
from testing import get_test_relpron
from __config__.filepath import AUX_DIR

test_fn = get_test_relpron()

def get_scores(subdir='meanfield_relpron', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def add_scores(scores, subdir='meanfield_relpron'):
    "Add newly calculated scores"
    for filename in os.listdir(os.path.join(AUX_DIR, subdir)):
        # Skip score files
        if filename[-3:] != '.gz':
            continue
        # Skip files we've already processed
        name = filename.split('.')[0]
        if name in scores:
            continue
        # Get score
        score_fn = load_scoring_fn(name)
        scores[name] = test_fn(score_fn)
        print(name)

def save_scores(scores, subdir='meanfield_relpron', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    with open(score_file, 'wb') as f:
        pickle.dump(scores, f)

scores = get_scores()
add_scores(scores)
save_scores(scores)

#score_fn = load_baseline_scoring_fn('multicore-5-400-0-1-0-0-32-1_0-40-0_01-1_0-4_0')
#test_fn(score_fn)