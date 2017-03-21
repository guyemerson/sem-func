import os, pickle, gzip, numpy as np

from variational import get_semfunc, mean_field
from testing import get_test_preds
from __config__.filepath import AUX_DIR
from utils import is_verb

def get_verbs(prefix='multicore', thresh=5):
    """
    Get boolean array, True if the pred is a verb
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :return: frequency array
    """
    with open(os.path.join(AUX_DIR, '{}-{}-vocab.pkl'.format(prefix, thresh)), 'rb') as f:
        vocab = pickle.load(f)
    return np.array([is_verb(x) for x in vocab])

def get_verb_noun_freq(prefix='multicore', thresh=5, pred_list=None):
    """
    Get frequency, normalised separately for nouns and verbs
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param pred_list: if given, only return frequency for these preds
    :return: frequency array
    """
    with open(os.path.join(AUX_DIR, '{}-{}-freq.pkl'.format(prefix, thresh)), 'rb') as f:
        freq = pickle.load(f)
    verbs = get_verbs(prefix, thresh)
    nouns = np.invert(verbs)
    freq = np.array(freq, dtype='float')  # to allow division
    freq[verbs] /= freq[verbs].sum()
    freq[nouns] /= freq[nouns].sum()
    if pred_list is False:
        return freq
    else:
        return freq[pred_list]

def get_semfuncs_from_vectors(name, bias_method, scale, C, target=None, Z=None, alpha=None, pred_list=None, vectors=None, freq=None, as_dict=False, include_bias=False, directory='simplevec'):
    """
    Get semantic functions from given weights
    :param name: name of parameter file
    :param bias_method: how to initialise biases ('target' or 'frequency')
    :param scale: factor to multiply simple vectors by
    :param C: total cardinality
    :param target: desired energy for an 'untypical' vector (only for bias method 'target')
    :param Z: weighted predicate truth of rest of vocabulary (only for bias method 'frequency')
    :param alpha: smoothing of frequency in generation (only for bias method 'frequency')
    :param pred_list: list of predicates to use
    :param as_dict: return as a dict (default as a list)
    :param include_bias: return all biases as an array
    :return: semantic functions (, bias)
    """
    prefix, thresh, dim, *_ = name.split('-')
    dim = int(dim)
    # Predicates to use
    if pred_list is None:
        preds, _ = get_test_preds(prefix, thresh)
        pred_list = sorted(preds)
    # Load vectors
    if vectors is None:
        with gzip.open(os.path.join(AUX_DIR, directory, name+'.pkl.gz'), 'rb') as f:
            all_vectors = pickle.load(f)
        if isinstance(all_vectors, np.ndarray):
            if pred_list is False:
                vectors = all_vectors
            else:
                vectors = all_vectors[pred_list]
                del all_vectors  # Conserve memory
        elif isinstance(all_vectors, dict):
            vectors = np.array([all_vectors[i] for i in pred_list])
        else:
            raise TypeError
    # Multiply by the scale factor
    vec = vectors * scale
    
    # Don't use negative weights in bias calculation
    vec_for_bias = vec.clip(0)
    
    # Define bias of predicates
    
    if bias_method == 'target':
        # Maximum activation of each predicate
        high = np.partition(vec_for_bias, -C, axis=1)[:,-C:].sum(axis=1)
        # Average activation if the top units are not used but still nouny/verby
        other = (vec_for_bias.sum(1) - high) / (dim - C) * C
        # Minimum pred bias makes an average predicate have the target energy
        bias = other + target
        # For preds with a bigger gap between max and other activation,
        # make the bias the average of the two
        gap = high - other
        mask = (gap > 2 * target)
        bias[mask] = other[mask] + gap[mask] / 2
    
    elif bias_method == 'frequency':
        # freq[i] ~ freq[i]^alpha / (Z + freq[i]^alpha) semfunc[i](ent) 
        if freq is None:
            freq = get_verb_noun_freq(prefix, thresh, pred_list)
        ent = np.ones(dim * 2) * C/dim
        bias = np.dot(vec_for_bias, ent) + np.log(1 / freq / (1 + Z * freq ** -alpha) - 1)
    
    else:
        raise ValueError('bias method not recognised')
    
    # Define semantic functions
    if as_dict:
        semfuncs = {i:get_semfunc(v,b) for i,v,b in zip(pred_list, vec, bias)}
    else:
        semfuncs = [get_semfunc(v,b) for v,b in zip(vec, bias)]
    
    if include_bias:
        return semfuncs, bias
    else:
        return semfuncs

def get_entities(bias_method, scale, C, target=None, Z=None, alpha=None, wrong_weight=0, name=None, basic_settings=None,
                 pred_list=None, mean_field_kwargs=None, output_dir='meanfield', input_dir='simplevec', skip_if_exists=True, bias_suffix='bias', verbose=False):
    """
    Get mean field entity vectors based on given parameter vectors
    Hyperparameters of binary-valued model:
    :param bias_method: how to initialise biases ('target' or 'frequency')
    :param scale: factor to multiply simple vectors by
    :param C: total cardinality
    :param target: desired energy for an 'untypical' vector (only for bias method 'target')
    :param Z: normalisation constant for predicate truth (only for bias method 'frequency')
    :param alpha: smoothing of frequency in generation (only for bias method 'frequency')
    :param wrong_weight: weight for decreasing the probability of the 'wrong' dimensions (for verbs or nouns)
    Simple vector model to load:
    :param name: name of model, as a string, or else use the following:
    :param basic_settings: alternatively, iterable of settings
    Other:
    :param pred_list: list of predicates to use
    :param mean_field_kwargs: kwargs when calculating meanfield vectors
    :param output_dir: directory to save meanfield vectors
    :param input_dir: directory for parameter vectors
    :param skip_if_exists: skip files that have already been calculated
    :param bias_suffix: name for files
    :param verbose: print messages
    :return: {pred: entity vector}
    """
    # File names
    if name is None:
        name = '-'.join(str(x).replace('.','_').replace('-','~')
                        for x in basic_settings)
    
    prefix, thresh, dim, *_ = name.split('-')
    dim = int(dim)
    
    if bias_method == 'target':
        fullname = name + '-' + '-'.join(str(x).replace('.','_').replace('-','~')
                                         for x in (scale, C, target, wrong_weight))
    elif bias_method == 'frequency':
        fullname = name + '-' + '-'.join(str(x).replace('.','_').replace('-','~')
                                         for x in (scale, C, Z, alpha, wrong_weight))
    
    # Skip if this setup has already been calculated
    if skip_if_exists and os.path.exists(os.path.join(AUX_DIR, output_dir, fullname+'.pkl.gz')):
        return
    
    # Predicates to use
    if pred_list is None:
        preds, _ = get_test_preds(prefix, thresh)
        pred_list = sorted(preds)
    
    # Load model
    if verbose: print("Loading model")
    semfuncs, biases = get_semfuncs_from_vectors(name, bias_method, scale, C, target, Z, alpha, pred_list, directory=input_dir, include_bias=True)
    # Save biases
    if pred_list:
        biases = {i:b for i, b in zip(pred_list, biases)}
    with gzip.open(os.path.join(AUX_DIR, output_dir, '{}-{}.pkl.gz'.format(fullname, bias_suffix)), 'wb') as f:
        pickle.dump(biases, f)
    
    # Calculate entity vectors
    
    noun_ratio = np.ones(2*dim)
    noun_ratio[dim:] = np.exp(-wrong_weight)
    verb_ratio = np.ones(2*dim)
    verb_ratio[:dim] = np.exp(-wrong_weight)
    ratio = {True: verb_ratio, False: noun_ratio}
    verbs = get_verbs(prefix, thresh)
    
    if verbose: print("Calculating entity vectors")
    if mean_field_kwargs is None:
        mean_field_kwargs = {}
    if pred_list is False:
        ent = np.zeros((len(semfuncs), dim*2))
        semfunc_iterator = enumerate(semfuncs)
    else:
        ent = {}
        semfunc_iterator = zip(pred_list, semfuncs)
    for i, sf in semfunc_iterator:
        if verbose: print(i)
        ent[i] = mean_field(sf, C, prob_ratio=ratio[verbs[i]], verbose=verbose, **mean_field_kwargs)
    
    # Save to disk
    
    with gzip.open(os.path.join(AUX_DIR, output_dir, fullname+'.pkl.gz'), 'wb') as f:
        pickle.dump(ent, f)
    
    print('done')


if __name__ == "__main__":
    from itertools import product
    from multiprocessing import Pool
    from random import shuffle
    
    # Grid search over hyperparameters
    
    bias_method = ['frequency']
    scales = [1]
    Cs = [30, 40, 50]
    targets = [None]
    Zs = [0.0001, 0.001, 0.01]
    alphas = [1]
    wrong_weights = [1, 4, 16]
    
    grid = product(bias_method, scales, Cs, targets, Zs, alphas, wrong_weights)
    
    # Vector models
    
    simplevec = os.listdir(os.path.join(AUX_DIR, 'simplevec'))
    simplevec_filtered = []
    for name in simplevec:
        parts = name.split('-')
        if len(parts) != 8:
            continue
        prefix, thresh, dim, k, a, right_smooth, all_smooth, seed = parts
        if all([k == '0',
                right_smooth == '0']):
            simplevec_filtered.append(name.split('.')[0])
    
    full_grid = list(product(grid, simplevec_filtered))
    shuffle(full_grid)
    
    def train(hyper, simplevec_name):
        print(hyper, simplevec_name)
        if hyper[5] == 0:
            hyper = list(hyper)
            hyper[4] *= 1000
        elif hyper[5] == 0.6:
            hyper = list(hyper)
            hyper[4] *= 10
        get_entities(*hyper, name=simplevec_name, mean_field_kwargs={"max_iter":500}, output_dir='meanfield')
    
    with Pool(16) as p:
        p.starmap(train, full_grid)
