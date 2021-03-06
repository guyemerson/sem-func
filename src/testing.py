import numpy as np, pickle, os
from scipy.stats import spearmanr
from itertools import chain
from collections import defaultdict

from __config__.filepath import AUX_DIR

# Evaluation

def scores(sim, pairs, **kwargs):
    """
    Apply a similarity function to many pairs, replacing nan with 0
    :param sim: similarity function
    :param pairs: pairs of items
    :return: numpy array
    """
    return np.nan_to_num([sim(*x, **kwargs) for x in pairs])

def evaluate(sim, pairs, gold, **kwargs):
    """
    Calculate the Spearman rank correlation on a dataset 
    :param sim: trained similarity function
    :param pairs: pairs of items
    :param gold: annotated similarity scores
    :return: Spearman rank correlation
    """
    return spearmanr(gold, scores(sim, pairs, **kwargs))

def evaluate_relpron(score_fn, items, term_to_properties, verbose=False, **kwargs):
    """
    Calculate mean average precision (MAP) for finding properties of terms
    :param score_fn: function from (term, (which, (verb, agent, patient))) to score
    :param items: list of (which, (verb, agent, patient)) tuples
    :param term_to_properties: mapping from terms to indices of the items list
    """
    if verbose:
        kwargs['verbose'] = verbose
    av_precision = []
    for term in term_to_properties:
        pairs = ((term, item) for item in items)
        all_scores = scores(score_fn, pairs, **kwargs)
        ranking = list(reversed(all_scores.argsort()))
        positions = sorted([ranking.index(i) for i in term_to_properties[term]])
        precision = [(i+1)/(pos+1) for i, pos in enumerate(positions)]
        av_prec = sum(precision)/len(precision)
        if verbose:
            print('average precision for {}: {}'.format(term, av_prec))
        av_precision.append(av_prec)
    return sum(av_precision)/len(av_precision)

# TODO assume an oracle for hypernymy detection

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
    with open('../data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt') as f:
        ws_sim = [line.strip().split('\t') for line in f]
    with open('../data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt') as f:
        ws_rel = [line.strip().split('\t') for line in f]
    sim_pairs = [(x[0], x[1]) for x in ws_sim]
    rel_pairs = [(x[0], x[1]) for x in ws_rel]
    sim_scores = np.array([float(x[2]) for x in ws_sim])
    rel_scores = np.array([float(x[2]) for x in ws_rel])
    
    return (sim_pairs, sim_scores), (rel_pairs, rel_scores)

def get_men():
    """
    Get MEN data (noun pairs only)
    :return: (pairs, scores)
    """
    noun_pairs = []
    noun_scores = []
    with open('../data/MEN/MEN_dataset_lemma_form_full') as f:
        for line in f:
            # Only get pairs that are both nouns (2005 pairs)
            # This excludes: 29 pairs that are both verbs, 96 both adjectives, 870 mixed part-of-speech
            first, second, score = line.split()
            lemma1, pos1 = first.split('-')
            lemma2, pos2 = second.split('-')
            if pos1 == 'n' and pos2 == 'n':
                noun_pairs.append((lemma1, lemma2))
                noun_scores.append(float(score))
    noun_scores = np.array(noun_scores)
    return (noun_pairs, noun_scores)

def get_simverb(subset=None):
    """
    Get SimVerb-3500 data
    :return: (pairs, scores)
    """
    simverb = []
    if subset == 'dev':
        name = '500-dev'
    elif subset == 'test':
        name = '3000-test'
    else:
        name = '3500'
    with open('../data/SimVerb-3500/SimVerb-{}.txt'.format(name)) as f:
        f.readline()  # first line is headings
        for line in f:
            simverb.append(line.strip().split('\t'))
    all_pairs = [(x[0], x[1]) for x in simverb]
    all_scores = np.array([float(x[3]) for x in simverb])
    return (all_pairs, all_scores)

def get_relpron(testset=False):
    """
    Get RelPron data
    :param testset: use testset (default, use devset)
    :return: list of: (term, (SBJ-or-OBJ, (verb, subject, object)))
    """
    if testset:
        subset = 'testset'
    else:
        subset = 'devset'
    items = []
    with open('../data/relpron/'+subset, 'r') as f:
        for line in f:
            # e.g.
            # OBJ garrison_N: organization_N that army_N install_V
            which, noun, head, _, first, second = line.strip().split()
            if which == 'SBJ':
                subj = head
                verb = first
                obj = second
            else:
                subj = first
                verb = second
                obj = head
            # Strip "_N:", "_N", "_V"
            items.append((noun[:-3], (which, (verb[:-2], subj[:-2], obj[:-2]))))
    return items

def get_relpron_hyper(testset=False):
    """
    Get mapping from terms to hypernyms, from the RelPron data
    :param testset: use testset (default, use devset)
    :return: dict of {term: hypernym}
    """
    if testset:
        subset = 'testset'
    else:
        subset = 'devset'
    hyper = {}
    with open('../data/relpron/'+subset, 'r') as f:
        for line in f:
            # e.g.
            # OBJ garrison_N: organization_N that army_N install_V
            _, noun, head, _, _, _ = line.strip().split()
            hyper[noun[:-3]] = head[:-2]
    return hyper

def get_relpron_separated(testset=False):
    """
    Get RelPron data, separating properties from terms
    :param testset: use testset (default use devset)
    :return: list of (SBJ-or-OBJ, verb-subject-object), and {term : {property indices}}
    """
    items = get_relpron(testset)
    reduced_items = []
    term_to_properties = defaultdict(set)
    for i, (term, description) in enumerate(items):
        term_to_properties[term].add(i)
        reduced_items.append(description)
    return reduced_items, term_to_properties

def get_GS2011():
    """
    Get GS2011 data
    :return: (pairs-of-vso-triples, scores, high-bools)
    """
    pairs = []
    scores = []
    highs = []
    with open('../data/Oxford/GS2011data.txt', 'r') as f:
        f.readline()
        for line in f:
            _, v1, s, o, v2, sim, hilo = line.split()
            pairs.append(((v1,s,o), (v2,s,o)))
            scores.append(sim)
            highs.append(hilo == 'HIGH')
    return pairs, scores, highs

def get_GS2011_indexed():
    """
    Get GS2011 data, assigning an index to each unique triple
    :return: (vso_triples, (pairs-of-indices, scores))
    """
    triple_pairs, scores, _ = get_GS2011()
    triples = []
    triple_to_index = {}
    index_pairs = []
    for pair in triple_pairs:
        indices = []
        for trip in pair:
            if trip in triple_to_index:
                i = triple_to_index[trip]
            else:
                i = len(triples)
                triples.append(trip)
                triple_to_index[trip] = i
            indices.append(i)
        index_pairs.append(tuple(indices))
    return triples, (index_pairs, scores)

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

def load_freq_lookup_dicts(prefix='multicore', thresh=5):
    """
    Load part-of-speech-specific dicts mapping lemmas to the most frequent index
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :return: {'v':verb_dict, 'n':noun_dict}
    """
    # Load files
    with open(os.path.join(AUX_DIR, '{}-{}-vocab.pkl'.format(prefix,thresh)), 'rb') as f:
        pred_name = pickle.load(f)
    with open(os.path.join(AUX_DIR, '{}-{}-freq.pkl'.format(prefix,thresh)), 'rb') as f:
        pred_freq = pickle.load(f)
    # Get lookup dictionaries
    return get_freq_lookup_dicts(pred_name, pred_freq)

# Predicate indices from datasets

def get_simlex_wordsim_preds(prefix='multicore', thresh=5):
    """
    Get the set of pred indices from simlex and wordsim
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :return: {pred indices}, {out-of-vocabulary items}
    """
    flookup = load_freq_lookup_dicts(prefix, thresh)
    (n_pairs, _), (v_pairs, _), _ = get_simlex()
    (sim_pairs, _), (rel_pairs, _) = get_wordsim()
    
    preds = set()
    OOV = set()
    
    for pos, pairs in [('n', chain(n_pairs, sim_pairs, rel_pairs)), ('v', v_pairs)]:
        for p in pairs:
            for x in p:
                try:
                    preds.add(flookup[pos][x])
                except KeyError:
                    OOV.add(x)
    
    return preds, OOV

def get_relpron_preds(prefix='multicore', thresh=5, include_test=True, include_dev=True):
    """
    Get the set of pred indices from the relpron dataset
    :param include_test: include testset preds
    :param include_dev: include devset preds
    :return: {pred indices}, {out-of-vocabulary items}
    """
    flookup = load_freq_lookup_dicts(prefix, thresh)
    data = []
    if include_test:
        data.extend(get_relpron(True))
    if include_dev:
        data.extend(get_relpron(False))
    
    preds = set()
    OOV = set()
    
    for target, (_, (verb, agent, patient)) in data:
        for pos, x in [('v', verb),
                       ('n', target),
                       ('n', agent),
                       ('n', patient)]:
            try:
                preds.add(flookup[pos][x])
            except KeyError:
                OOV.add(x)
    
    return preds, OOV

def get_test_preds(prefix='multicore', thresh=5):
    """
    Get the set of pred indices from all test datasets
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :return: {pred indices}, {out-of-vocabulary items}
    """
    flookup = load_freq_lookup_dicts(prefix, thresh)
    (n_pairs, _), (v_pairs, _), _ = get_simlex()
    (sim_pairs, _), (rel_pairs, _) = get_wordsim()
    men_pairs, _ = get_men()
    verb_pairs, _ = get_simverb()
    
    preds = set()
    OOV = set()
    
    for pos, pairs in [('n', chain(n_pairs, sim_pairs, rel_pairs, men_pairs)),
                       ('v', chain(v_pairs, verb_pairs))]:
        for p in pairs:
            for x in p:
                try:
                    preds.add(flookup[pos][x])
                except KeyError:
                    OOV.add(x)
    
    relpron_preds, relpron_OOV = get_relpron_preds(prefix, thresh)
    preds.update(relpron_preds)
    OOV.update(relpron_OOV)
    
    return preds, OOV

# Wrappers for similarity functions

def with_lookup(old_fn, lookup):
    """
    Wrap a similarity function, looking up a pred for each given lemma,
    returning 0 if the lemma was not found
    :param old_fn: function taking two indices and returning a similarity score
    :param lookup: mapping from lemmas to sets of indices
    :return: similarity function
    """
    def sim_fn(a, b, *args, **kwargs):
        "Calculate similarity"
        try:
            return old_fn(lookup[a], lookup[b], *args, **kwargs)
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
    def sim_fn(a, b, *args, **kwargs):
        "Calculate the average similarity across all preds, weighted by frequency"
        scores = []
        freqs = []
        try:
            for p in lookup[a]:
                for q in lookup[b]:
                    # Get similarity and frequency
                    scores.append(old_fn(p, q, *args, **kwargs))
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
    def sim_fn(a, b, *args, **kwargs):
        "Look up pred names, then apply the old function"
        return old_fn(pred_name[a], pred_name[b], *args, **kwargs)
    return sim_fn

def with_lookup_for_relpron(old_fn, lookup, indices_only=True):
    """
    Wrap a similarity function, looking up a pred for each given lemma,
    returning 0 if the lemma was not found
    :param old_fn: function mapping (term, description) to scores
    :param lookup: mapping from lemmas to sets of indices, separately for nouns and verbs
    :return: similarity function
    """
    if indices_only:
        lookup_triple = lambda x:x
    else:
        lookup_triple = lambda verb, agent, patient : (lookup['v'][verb], lookup['n'][agent], lookup['n'][patient])
    
    def new_fn(term, description, verbose=False, **kwargs):
        "Calculate score"
        if verbose:
            print(term, description)
        which, triple = description
        try:
            transformed = (lookup['n'][term],
                           (which,
                            lookup_triple(triple)))
        except KeyError:
            # Unknown lemmas
            if verbose:
                print('OOV')
            return 0
        
        score = old_fn(*transformed, **kwargs)
        if verbose:
            print(score)
        return score
    
    return new_fn

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

def filter_common(freq, lookup, pairs, scores, threshold=1000):
    """
    Keep only those pairs whose frequencies are above the threshold
    :param freq: array of frequencies of predicates
    :param lookup: dict mapping strings to predicate indices
    :param pairs: list of pairs of strings
    :param scores: list of scores
    :param threshold: minimum frequency to keep a predicate
    """
    common = ([], [])
    for pair, score in zip(pairs, scores):
        try:
            if all(freq[lookup.get(x,'')] >= 1000 for x in pair):
                common[0].append(pair)
                common[1].append(score)
        except IndexError:  # a string could not be found (so we try freq['']) 
            pass
    return common

def get_test_simlex_wordsim(prefix='multicore', thresh=5):
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
    simlex_common = filter_common(pred_freq, freq_lookup['n'], *simlex[0])
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

def get_test_all(prefix='multicore', thresh=5):
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
    men = get_men()
    simverb = get_simverb()
    # Get common pairs in SimLex, MEN, and SimVerb
    simlex_common = filter_common(pred_freq, freq_lookup['n'], *simlex[0])
    men_common = filter_common(pred_freq, freq_lookup['n'], *men)
    simverb_common = filter_common(pred_freq, freq_lookup['v'], *simverb)
    # Define the testing function
    def test_all(sim, ret=True):
        """
        Test the similarity function on all datasets
        :param sim: function mapping pairs of predicate indices to similarity scores
        :param ret: set True to return the scores
        """
        n_sim = with_lookup(sim, freq_lookup['n'])
        v_sim = with_lookup(sim, freq_lookup['v'])
        sl_n = evaluate(n_sim, *simlex[0])
        sl_v = evaluate(v_sim, *simlex[1])
        ws_s = evaluate(n_sim, *wordsim[0])
        ws_r = evaluate(n_sim, *wordsim[1])
        mn   = evaluate(n_sim, *men)
        sv   = evaluate(v_sim, *simverb)
        sl_c = evaluate(n_sim, *simlex_common)
        mn_c = evaluate(n_sim, *men_common)
        sv_c = evaluate(v_sim, *simverb_common)
        print('sl noun:', sl_n)
        print('sl verb:', sl_v)
        print('ws sim.:', ws_s)
        print('ws rel.:', ws_r)
        print('men nn.:', mn)
        print('simverb:', sv)
        print('sl cmn.:', sl_c)
        print('mn cmn.:', mn_c)
        print('sv cmn.:', sv_c)
        if ret:
            return sl_n, sl_v, ws_s, ws_r, mn, sv, sl_c, mn_c, sv_c
    
    return test_all

def get_test_relpron(prefix='multicore', thresh=5, testset=False, indices_only=True):
    """
    Get a testing function for a specific pred lookup
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param testset: whether to use the testset (default devset)
    :param indices_only: only pass indices of triples, not the triples themselves (default True)
    """
    freq_lookup = load_freq_lookup_dicts(prefix, thresh)
    data = get_relpron_separated(testset)
    if indices_only:
        convert = {'SBJ': 1, 'OBJ': 2}
        reduced_items, term_to_properties = data
        indexed_items = [(convert[which], i) for i, (which, _) in enumerate(reduced_items)]
        data = (indexed_items, term_to_properties)
    def test(score_fn, **kwargs):
        """
        Test a scoring function on the relpron data
        :param score_fn: function from (which, term, (verb, agent, patient)) to score
        :return: mean average precision
        """
        wrapped_score_fn = with_lookup_for_relpron(score_fn, freq_lookup)
        mean_av_prec = evaluate_relpron(wrapped_score_fn, *data, **kwargs)
        #print(mean_av_prec)
        return mean_av_prec
    return test

def get_test_relpron_hypernym(prefix='multicore', thresh=5, testset=False):
    """
    Get a testing function for a specific pred lookup
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param testset: whether to use the testset (default devset)
    """
    freq_lookup = load_freq_lookup_dicts(prefix, thresh)['n']
    # Get data and convert to required form for mean-average-precision evaluation
    term_to_hypernym = get_relpron_hyper(testset)
    hypernyms = sorted(set(term_to_hypernym.values()))
    term_to_hyp_index = {t:[hypernyms.index(h)] for t, h in term_to_hypernym.items()}
    def test(score_fn):
        """
        Test a implications on the relpron hypernyms
        :param score_fn: function from (term, hypernym) to score
        :return: mean average precision
        """
        wrapped_score_fn = with_lookup(score_fn, freq_lookup)
        mean_av_prec = evaluate_relpron(wrapped_score_fn, hypernyms, term_to_hyp_index)
        #print(mean_av_prec)
        return mean_av_prec
    return test

def get_test_relpron_ensemble(prefix='multicore', thresh=5, testset=False):
    """
    Get a testing function for a specific pred lookup
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param testset: whether to use the testset (default devset)
    """
    freq_lookup = load_freq_lookup_dicts(prefix, thresh)
    data = get_relpron_separated(testset)
    convert_which = {'SBJ': 1, 'OBJ': 2}
    reduced_items, _ = data
    convert_description = {(which, triple):(convert_which[which], i) for i, (which, triple) in enumerate(reduced_items)}
    def test(cached_score_fns, direct_score_fns, weight=0.5, cached_score_kwargs={}, direct_score_kwargs={}, **kwargs):
        """
        Test a scoring function on the relpron data
        :param cached_score_fns: list of functions mapping (which, term_index, triple_index) to score
        :param direct_score_fns: list of functions mapping (which, term, (verb, agent, patient)) to score
        :param weight: how much to use the cached score functions rather than the direct score functions
        :param cached_score_kwargs: keyword arguments to pass to cached score functions
        :param direct_score_kwargs: keyword arguments to pass to direct score functions
        Additional keyword arguments are passed to evaluate_relpron
        :return: mean average precision
        """
        wrapped_score_fns = [with_lookup_for_relpron(fn, freq_lookup) for fn in cached_score_fns]
        def ensemble_score_fn(term, description):
            "Return a weighted geometric mean of two sets of score functions"
            cached_scores = [fn(term, convert_description[description], **cached_score_kwargs) for fn in wrapped_score_fns]
            direct_scores = [fn(term, description, **direct_score_kwargs) for fn in direct_score_fns]
            combined_score = weight * np.log(cached_scores).mean() \
                             + (1-weight) * np.log(direct_scores).mean()
            return combined_score
        mean_av_prec = evaluate_relpron(ensemble_score_fn, *data, **kwargs)
        #print(mean_av_prec)
        return mean_av_prec
    return test

def get_test_GS2011(prefix='multicore', thresh=5, indices_only=True, with_lookup=True):
    """
    Get a testing function for a specific pred lookup
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param indices_only: only pass indices of triples, not the triples themselves (default True)
    """
    if indices_only:
        _, data = get_GS2011_indexed()
    else:
        raw_pairs, scores = get_GS2011()[:2]
        if with_lookup:
            lup = load_freq_lookup_dicts(prefix, thresh)
            pairs = [[(lup['v'][v], lup['n'][s], lup['n'][o]) for v,s,o in pair] for pair in raw_pairs]
            data = (pairs, scores)
        else:
            data = (raw_pairs, scores)

    def test(score_fn, **kwargs):
        """
        Test a scoring function on the GS2011 data
        :param score_fn: function from pairs of triple indices to scores
        :return: Spearman rank correlation 
        """
        return evaluate(score_fn, *data)
    return test

# File name manipulation

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
