import pickle, gzip, os, numpy as np
from itertools import product

from utils import is_verb
from __config__.filepath import AUX_DIR

# Hyperparameters

dataset = 'multicore-5'
D = 400  # Number of dimensions for nouns and verbs (separately) 
seed_range = [32, 8, 91, 64, 97]

smoothing_range = [0, 0.1, 1]  # Added to all counts
a_range = [0.75, 0.8, 0.9, 1]  # Power that frequencies are raised to under the null hypothesis
k_range = [-0.9, -0.69315, -0.5, 0]  # (-log(2), if we have half-half nouns and verbs in the same space)

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

full_template = os.path.join(AUX_DIR, 'simplevec', '{}-{}-{}-{}.pkl.gz')
    
# Calculate PPMI
def ppmi(obs_vec, a, smoothing, minimum=0):
    """
    Calculate positive pointwise mutual information
    (for verbs and nouns separately)
    :param vec: observed numbers of contexts
    :param a: power to raise frequencies to, for smoothing
    :param minimum: minimum ppmi score (default 0)
    """
    # Smooth frequencies
    vec = obs_vec + smoothing * smoothing_array
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
    
    # Keep positive
    new.clip(minimum, out=vec)
    return new

min_k = min(k_range)

for seed in seed_range:
    print('seed:', seed)
    
    with gzip.open(full_template.format(dataset, D, 'full', seed), 'rb') as f:
        count_vec = pickle.load(f)
    
    template = full_template.format(dataset, D, '{}-{}-{}', seed)
    
    for a, smoothing in product(a_range, smoothing_range):
        print('a, smoothing:', a, smoothing)
        vec = ppmi(count_vec, a, smoothing, min_k)
    
        # Shift and save
        
        for k in k_range:
            filename = template.format(str(k).replace('.','_').replace('-','~'),
                                       str(a).replace('.','_'),
                                       str(smoothing).replace('.','_'))
            if os.path.exists(filename): continue
            print('k:', k)
            new = (vec - k).clip(0)
            with gzip.open(filename, 'wb') as f:
                pickle.dump(new, f)
