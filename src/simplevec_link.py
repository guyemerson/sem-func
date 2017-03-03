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

### Link weights

# Sum over tuples

with open(os.path.join(AUX_DIR, 'multicore-5-count_pairs.pkl'), 'rb') as f:
    pairs = pickle.load(f)

print(len(pairs[0])+len(pairs[1]), "pairs in total")

link_total = np.zeros((2, D, D))

n_total = [0,0]

# Add connections from each tuple
for label, label_pairs in enumerate(pairs):
    for i, ((verb, noun), n) in enumerate(label_pairs.items()):
        link_total[label] += n * np.outer(ent[verb], ent[noun])
        n_total[label] += n

# Normalise to frequencies
link_total[0] /= n_total[0]
link_total[1] /= n_total[1]

# Take ppmi
link_wei = np.log(link_total) - 2 * np.log(C/D)
link_wei = link_wei.clip(0)

# Save all parameters to file

with open(os.path.join(AUX_DIR, 'simplevec_link', filename+'.pkl'), 'wb') as f:
    pickle.dump(link_wei, f)
