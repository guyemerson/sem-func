import pickle, gzip, os, numpy as np
from itertools import product

from utils import is_verb
from __config__.filepath import AUX_DIR

# Hyperparameters

dataset = 'multicore-5'
D = 400  # Number of dimensions for nouns and verbs (separately) 
C_range = [30, 40]
seed_range = [32, 8, 91, 64, 97]

a_range = [0.75, 0.8, 0.9, 1]  # Power that frequencies are raised to under the null hypothesis
k_range = [0]

# Load files

with open(os.path.join(AUX_DIR, '{}-vocab.pkl'.format(dataset)), 'rb') as f:
    pred_name = pickle.load(f)
verb = np.array([is_verb(p) for p in pred_name], dtype='bool')

full_template = os.path.join(AUX_DIR, 'simplevec_card', '{}-{}-{}-{}-{}.pkl.gz')
    
# Calculate PPMI
def ppmi(vec, a, C):
    # Smooth frequencies
    freq_pred = vec.sum(1) ** a
    freq_context = vec.sum(0) ** a
    
    # Normalise the smoothed marginal context probabilities (for noun and verb contexts separately)
    freq_context[:D] /= freq_context[:D].sum()
    freq_context[D:] /= freq_context[D:].sum()
    
    # Calculate joint probabilities (for nouns and verbs separately)
    freq_noun = vec * verb.reshape((-1,1))
    freq_verb = vec * np.invert(verb).reshape((-1,1))
    new = freq_noun / freq_noun.sum() + freq_verb / freq_verb.sum()
    
    # Take logs
    log_pred = np.log(freq_pred / freq_pred.sum())
    log_context = np.log(freq_context / freq_context.sum() * C)
    new = np.log(new * C)
    # The joint probabilities sum to 1 over just nouns or verbs
    # The marginal probabilities sum to 1 over all preds
    # This will lead to boosted scores 
    
    # Subtract logs
    new -= log_context
    new -= log_pred.reshape((-1,1))
    
    # Keep positive
    new.clip(0, out=vec)
    return new

for C, seed in product(C_range, seed_range):
    print('C', C, 'seed', seed)
    
    with gzip.open(full_template.format(dataset, D, C, 'full', seed), 'rb') as f:
        count_vec = pickle.load(f)
    
    template = full_template.format(dataset, D, C, '{}-{}', seed)
    
    for a in a_range:
        print('a', a)
        vec = ppmi(count_vec, a, C)
    
        # Shift and save
        
        for k in k_range:
            filename = template.format(str(k).replace('.',''),
                                       str(a).replace('.',''))
            if os.path.exists(filename): continue
            print('k', k)
            new = (vec - k).clip(0)
            with gzip.open(filename, 'wb') as f:
                pickle.dump(new, f)
