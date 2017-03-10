import os, pickle, gzip, numpy as np, argparse

from __config__.filepath import AUX_DIR

# Command line options

parser = argparse.ArgumentParser(description="Fit link weights")
parser.add_argument('filename')
args = parser.parse_args()

# Parameters

filename = args.filename

hyp = filename.split('-')
D = int(hyp[2])*2
C = int(hyp[7])

# Load vectors 

with gzip.open(os.path.join(AUX_DIR, 'meanfield_freq_all', filename+'.pkl.gz'), 'rb') as f:
    ent = pickle.load(f)

# Force cardinality
ent /= (ent.sum(1)/C).reshape((-1, 1))
np.clip(ent, 0, 1, ent)

### Link weights

# Sum over tuples

with open(os.path.join(AUX_DIR, 'multicore-5-count_pairs.pkl'), 'rb') as f:
    pairs = pickle.load(f)

print(len(pairs[0])+len(pairs[1]), "pairs in total")

link_total = np.zeros((2, D, D))  # Store sum of outer products
n_total = np.zeros(2)  # Store number of observations

# Add connections from each tuple
for label, label_pairs in enumerate(pairs):
    for (verb, noun), n in label_pairs.items():
        link_total[label] += n * np.outer(ent[verb], ent[noun])
        n_total[label] += n

# Normalise to frequencies
link_freq = link_total / n_total.reshape((-1, 1, 1))

# Save all parameters to file

with open(os.path.join(AUX_DIR, 'simplevec_link', filename+'-raw.pkl'), 'wb') as f:
    pickle.dump(link_freq, f)
