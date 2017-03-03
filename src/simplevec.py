import pickle, gzip, os, numpy as np
from collections import defaultdict

from __config__.filepath import AUX_DIR

# Hyperparameters

D = 400  # Number of dimensions for nouns and verbs (separately)
seed = 32
np.random.seed(seed)

# Dataset

dataset = 'multicore-5'

# Assign each context to a random bin
# By using a defaultdict, the bin is chosen when needed, and then cached

def rand_bin():
    "Return a random dimension"
    return np.random.randint(D)

get_bin = defaultdict(rand_bin)

# Load data

print('loading')

with open(os.path.join(AUX_DIR, '{}-count_tuple.pkl'.format(dataset)), 'rb') as f:
    count = pickle.load(f)

with open(os.path.join(AUX_DIR, '{}-vocab.pkl'.format(dataset)), 'rb') as f:
    pred_name = pickle.load(f)
V = len(pred_name)

# Count contexts

print('counting contexts')

vec = np.zeros((V, 2*D))

# Contexts are pairs (pred_index, context_type)
# where context_type is one of: s(ubject), o(bject), b(e)
# Contexts for verbs are shifted to the second half of the dimensions
# Sort the graphs so that the order is stable (treating None as -1)
for graph, n in sorted(count.items(), key=lambda x:tuple(y if y is not None else -1 for y in x[0])):
    if len(graph) == 2:
        p, q = graph
        vec[p, get_bin[q,'b']] += n
        vec[q, get_bin[p,'b']] += n
    else:
        v, s, o = graph
        if s is not None:
            vec[v, get_bin[s,'s']+D] += n
            vec[s, get_bin[v,'s']] += n
        if o is not None:
            vec[v, get_bin[o,'o']+D] += n
            vec[o, get_bin[v,'o']] += n
# TODO include subject-object contexts?

# Save

print('saving')

template = os.path.join(AUX_DIR, 'simplevec', '{}-{}-full-{}.pkl.gz')

with gzip.open(template.format(dataset, D, seed), 'wb') as f:
    pickle.dump(vec, f)
