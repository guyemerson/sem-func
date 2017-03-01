import os, pickle, gzip, numpy as np
from scipy.special import expit

from __config__.filepath import AUX_DIR

# Parameters

filename = 'multicore-5-400-0-08-91-1-40-0001-08'

hyp = filename.split('-')
D = int(hyp[2])*2
C = int(hyp[7])

# Load vectors 

with gzip.open(os.path.join(AUX_DIR, 'meanfield_freq_all', filename+'.pkl.gz'), 'rb') as f:
    ent = pickle.load(f)

### Link weights

# Sum over tuples

with open(os.path.join(AUX_DIR, 'multicore-5-count_tuple.pkl'), 'rb') as f:
    count = pickle.load(f)

print(len(count), "tuples in total")

link_total = np.zeros((2, D, D))

n_ag = 0
n_pat = 0

# Add connections from each tuple
for i, (triple, n) in enumerate(count.items()):
    print(i)
    if len(triple) == 2:  # Skip _be_v_id pairs
        continue
    v, ag, pat = triple
    if ag is not None:
        n_ag += n
        link_total[0] += n * np.outer(ent[v], ent[ag])
    if pat is not None:
        n_pat += n
        link_total[1] += n * np.outer(ent[v], ent[pat])

# Normalise to frequencies
link_total[0] /= n_ag
link_total[1] /= n_pat

# Take ppmi
link_wei = np.log(link_total) - 2 * np.log(C/D)
link_wei = link_wei.clip(0)

# Save all parameters to file

with open(os.path.join(AUX_DIR, 'simplevec_link', filename+'.pkl'), 'wb') as f:
    pickle.dump(link_wei, f)
