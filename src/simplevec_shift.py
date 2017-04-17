import pickle, gzip, os, numpy as np
from multiprocessing import Pool

from utils import is_verb
from __config__.filepath import AUX_DIR

def pmi(obs_vec, D, smoothing_array, noun, verb, a=1, right_smooth=0, all_smooth=0):
    """
    Calculate pointwise mutual information
    (for verbs and nouns separately)
    :param obs_vec: observed numbers of contexts
    :param D: number of dimensions
    :param smoothing_array: boolean array of shape [predicates, dimensions] for relevant dimensions 
    :param noun: boolean array of shape [predicates], true if noun 
    :param verb: boolean array of shape [predicates], true if verb
    :param a: power to raise frequencies to, for smoothing
    :param right_smooth: value to add to contexts of the right type (noun/verb)
    :param all_smooth: value to add to all contexts
    """
    # Smooth frequencies
    vec = obs_vec + right_smooth * smoothing_array + all_smooth
    freq_pred = vec.sum(1) ** a
    freq_context = vec.sum(0) ** a
    
    # Normalise the smoothed marginal context probabilities (for noun and verb contexts separately)
    freq_context[:D] /= freq_context[:D].sum()
    freq_context[D:] /= freq_context[D:].sum()
    
    # Normalise the smoothed marginal pred probabilities (for noun and verb contexts separately)
    freq_verb = freq_pred * verb
    freq_noun = freq_pred * noun
    freq_pred = freq_noun / freq_noun.sum() + freq_verb / freq_verb.sum()
    
    # Calculate joint probabilities (for nouns and verbs separately)
    vec_verb = vec * verb.reshape((-1,1))
    vec_noun = vec * noun.reshape((-1,1))
    new = vec_noun / vec_noun.sum() + vec_verb / vec_verb.sum()
    
    # Take logs
    log_pred = np.log(freq_pred)
    log_context = np.log(freq_context)
    new = np.log(new)
    
    # Subtract logs
    new -= log_context
    new -= log_pred.reshape((-1,1))
    new += np.log(2)  # To put nouns and verbs in the same space
    
    return new
    

def expand_seed(D, seed, k_range=(0,), a_range=(1,), right_smooth_range=(0,), all_smooth_range=(0,), clip=True, prefix='multicore', thresh=5, subdir='simplevec_all'):
    """
    Convert raw counts to shifted PMI vectors
    :param D: number of dimensions
    :param seed: random seed
    :param k_range: PMI shifts
    :param a_range: powers to raise frequencies to
    :param right_smooth_range: Laplace smoothing values for contexts of the right type
    :param all_smooth_range: Laplace smoothing values for contexts of the wrong type
    :param clip: remove negative values
    :param prefix: dataset prefix
    :param thresh: threshold
    :param subdir: directory to save and load models
    """
    dataset = '{}-{}'.format(prefix,thresh)
    
    # Load files
    
    with open(os.path.join(AUX_DIR, '{}-vocab.pkl'.format(dataset)), 'rb') as f:
        pred_name = pickle.load(f)
    verb = np.array([is_verb(p) for p in pred_name], dtype='bool')
    noun = np.invert(verb)
    
    verb_ones = np.zeros(2*D)
    verb_ones[D:] = 1
    noun_ones = np.zeros(2*D)
    noun_ones[:D] = 1
    smoothing_array = np.outer(verb, verb_ones) + np.outer(noun, noun_ones)
    
    full_template = os.path.join(AUX_DIR, subdir, '{}-{}-{}-{}.pkl.gz')
    
    print('seed:', seed)
    
    with gzip.open(full_template.format(dataset, D, 'full', seed), 'rb') as f:
        count_vec = pickle.load(f)
    
    template = full_template.format(dataset, D, '{}-{}-{}-{}', seed)
    
    for a, right_smooth, all_smooth in product(a_range, right_smooth_range, all_smooth_range):
        print('a, right_smooth, all_smooth:', a, right_smooth, all_smooth)
        vec = pmi(count_vec, D, smoothing_array, noun, verb, a, right_smooth, all_smooth)
    
        # Shift and save
        
        for k in k_range:
            filename = template.format(str(k).replace('.','_').replace('-','~'),
                                       str(a).replace('.','_').replace('-','~'),
                                       str(right_smooth).replace('.','_').replace('-','~'),
                                       str(all_smooth).replace('.','_').replace('-','~'))
            if os.path.exists(filename): continue
            print('k:', k)
            new = (vec - k)
            if clip:
                new = new.clip(0, out=new)
            
            # Filter preds
            #filtered = {i: new[i] for i in pred_list}
            
            with gzip.open(filename, 'wb') as f:
                #pickle.dump(filtered, f)
                pickle.dump(new, f)


if __name__ == "__main__":
    # Command line options
    import argparse
    from itertools import product
    parser = argparse.ArgumentParser(description="Convert counts to PMI vectors")
    parser.add_argument('D', type=int)
    parser.add_argument('seed', nargs='+', type=int)
    parser.add_argument('--processes', type=int, default=5)
    args = parser.parse_args()
    with Pool(args.processes) as p:
        p.starmap(expand_seed, product([args.D], args.seed))
