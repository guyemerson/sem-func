import numpy as np, pickle, os
from scipy.stats import spearmanr

from __config__.filepath import AUX_DIR

# Evaluation

def scores(sim, pairs):
    """
    Apply a similarity function to many pairs, replacing nan with 0
    :param sim: similarity function
    :param pairs: pairs of items
    :return: numpy array
    """
    return np.nan_to_num([sim(a,b) for a,b in pairs])

def evaluate(sim, pairs, gold):
    """
    Calculate the Spearman rank correlation on a dataset 
    :param sim: trained similarity function
    :param pairs: pairs of items
    :param gold: annotated similarity scores
    :return: Spearman rank correlation
    """
    return spearmanr(gold, scores(sim, pairs))

# Evaluation datasets

def get_simlex():
    """
    Get SimLex-999 data, split by part of speech
    :return: (n_pairs, n_scores), (v_pairs, v_scores), (a_pairs, a_scores)
    """
    # Read the file
    simlex = []
    with open('../data/SimLex-999/SimLex-999.txt') as f:
        f.readline()  # first line is headings
        for line in f:
            simlex.append(line.strip().split('\t'))
    # Split the data
    n_simlex = [x for x in simlex if x[2] == 'N']
    v_simlex = [x for x in simlex if x[2] == 'V']
    a_simlex = [x for x in simlex if x[2] == 'A']
    n_pairs = [(x[0], x[1]) for x in n_simlex]
    v_pairs = [(x[0], x[1]) for x in v_simlex]
    a_pairs = [(x[0], x[1]) for x in a_simlex]
    n_scores = np.array([float(x[3]) for x in n_simlex])
    v_scores = np.array([float(x[3]) for x in v_simlex])
    a_scores = np.array([float(x[3]) for x in a_simlex])
    
    return (n_pairs, n_scores), (v_pairs, v_scores), (a_pairs, a_scores)

def get_wordsim():
    """
    Get WordSim-353 data (separated into similarity and relatedness subsets)
    :return: (sim_pairs, sim_scores), (rel_pairs, rel_scores)
    """
    with open('/homes/gete2/git/sem-func/data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt') as f:
        ws_sim = [line.strip().split('\t') for line in f]
    with open('/homes/gete2/git/sem-func/data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt') as f:
        ws_rel = [line.strip().split('\t') for line in f]
    sim_pairs = [(x[0], x[1]) for x in ws_sim]
    rel_pairs = [(x[0], x[1]) for x in ws_rel]
    sim_scores = np.array([float(x[2]) for x in ws_sim])
    rel_scores = np.array([float(x[2]) for x in ws_rel])
    
    return (sim_pairs, sim_scores), (rel_pairs, rel_scores)

# Lookup

def get_pred_dict(pred_name):
    """
    Create a dict mapping predicate names to indices
    :param pred_name: list of pred names, in order
    :return: dict
    """
    return {p:i for i,p in enumerate(pred_name)}

def get_lookup_dicts(pred_name):
    """
    Create part-of-speech-specific dicts mapping lemmas to sets of indices
    :param pred_name: list of pred names, in order
    :return: {'v':verb_dict, 'n':noun_dict}
    """
    # Initialise lookup dicts
    v_lookup = {}
    n_lookup = {}
    lookup = {'v':v_lookup, 'n':n_lookup}
    # Sort preds by part of speech and lemma 
    for i, x in enumerate(pred_name):
        lemma, pos, _ = x[1:].rsplit('_', 2)
        # Add the index to the correct set, initialising one if necessary
        lookup[pos].setdefault(lemma, set()).add(i)
    return lookup

def get_freq_lookup_dicts(pred_name, pred_freq):
    """
    Create part-of-speech-specific dicts mapping lemmas to the most frequent index
    :param pred_name: list of pred names, in order
    :param pred_freq: frequency of preds
    :return: {'v':verb_dict, 'n':noun_dict}
    """
    # Get all indices
    lookup = get_lookup_dicts(pred_name)
    freq_lookup = {}
    for pos, x_lookup in lookup.items():
        # Get the most frequent indices
        key = lambda i:pred_freq[i]
        freq_lookup[pos] = {lemma:max(inds, key=key) for lemma, inds in x_lookup.items()}
    return freq_lookup

# Wrappers for similarity functions

def with_lookup(old_fn, lookup):
    """
    Wrap a similarity function, looking up a pred for each given lemma,
    returning 0 if the lemma was not found
    :param old_fn: function taking two indices and returning a similarity score
    :param lookup: mapping from lemmas to sets of indices
    :return: similarity function
    """
    def sim_fn(a, b):
        "Calculate the average similarity across all preds, weighted by frequency"
        try:
            return old_fn(lookup[a], lookup[b])
        except KeyError:
            # Unknown lemmas
            return 0
    return sim_fn

def with_weighted_lookup(old_fn, lookup, freq):
    """
    Wrap a similarity function, looking up all preds for the given lemmas,
    weightings the pairwise similarities by frequency,
    and returning 0 if the lemma was not found
    :param old_fn: function taking two indices and returning a similarity score
    :param lookup: mapping from lemmas to sets of indices
    :param freq: mapping from indices to frequencies
    :return: similarity function
    """
    def sim_fn(a, b):
        "Calculate the average similarity across all preds, weighted by frequency"
        scores = []
        freqs = []
        try:
            for p in lookup[a]:
                for q in lookup[b]:
                    # Get similarity and frequency
                    scores.append(old_fn(p, q))
                    freqs.append(freq[p]*freq[q])
            # Calculate weighted average
            totalscore = sum(scores[i] * freqs[i] for i in range(len(scores)))
            totalfreq = sum(freqs)
            return totalscore / totalfreq
        except KeyError:
            # Unknown lemmas
            return 0
    return sim_fn

def from_index(old_fn, pred_name):
    """
    Wrap a similarity function that takes predicate names as input (e.g. gensim Word2Vec.similarity)
    :param old_fn: function taking a pair of pred names and returning a similarity score
    :return: similarity function
    """
    def sim_fn(a, b):
        "Look up pred names, then apply the old function"
        return old_fn(pred_name[a], pred_name[b])
    return sim_fn

# Extra

def compare(old, new, pairs, gold=None, n=20, direction='both'):
    """
    Compare how similarity scores vary between two models, showing the biggest differences
    :param old: first model
    :param new: second model
    :param pairs: pairs of items
    :param gold: annotated scores
    :param n: how many pairs to return
    :param direction: 'pos' (new higher), 'neg' (new lower), default both
    """
    # Get scores and differences
    old_scores = scores(old, pairs)
    new_scores = scores(new, pairs)
    diffs = new_scores - old_scores
    # Choose which to keep
    if direction == 'pos':
        np.clip(diffs, 0, np.inf, out=diffs)
    elif direction == 'neg':
        np.clip(diffs, -np.inf, 0, out=diffs)
    # Get absolute difference
    diffs = np.abs(diffs)
    # Print the most different
    for i in diffs.argsort()[-n:]:
        print(*pairs[i])
        if gold: print('({})'.format(gold[i]))
        print(old_scores[i])
        print(new_scores[i])

# Evaluation with specific lookup

def get_test_all(prefix, thresh):
    """
    Get a testing function for a specific pred lookup
    :param prefix: name of dataset
    :param thresh: frequency threshold
    """
    # Load files
    with open(os.path.join(AUX_DIR, '{}-{}-vocab.pkl'.format(prefix,thresh)), 'rb') as f:
        pred_name = pickle.load(f)
    with open(os.path.join(AUX_DIR, '{}-{}-freq.pkl'.format(prefix,thresh)), 'rb') as f:
        pred_freq = pickle.load(f)
    # Get lookup dictionaries
    freq_lookup = get_freq_lookup_dicts(pred_name, pred_freq)
    # Get datasets
    simlex = get_simlex()
    wordsim = get_wordsim()
    # Get common pairs in SimLex
    simlex_common = ([], [])
    for pair, score in zip(*simlex[0]):
        if all(pred_freq[freq_lookup['n'].get(x,0)] > 1000 for x in pair):
            simlex_common[0].append(pair)
            simlex_common[1].append(score)
    # Define the testing function
    def test_all(sim, ret=True):
        """
        Test the similarity function on all datasets
        :param sim: function mapping pairs of predicate indices to similarity scores
        :param ret: set True to return the scores
        """
        n_sim = with_lookup(sim, freq_lookup['n'])
        v_sim = with_lookup(sim, freq_lookup['v'])
        n = evaluate(n_sim, *simlex[0])
        v = evaluate(v_sim, *simlex[1])
        s = evaluate(n_sim, *wordsim[0])
        r = evaluate(n_sim, *wordsim[1])
        c = evaluate(n_sim, *simlex_common)
        print('noun:', n)
        print('verb:', v)
        print('sim.:', s)
        print('rel.:', r)
        print('cmn.:', c)
        if ret:
            return n, v, s, r, c
    
    return test_all
