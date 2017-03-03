import pickle, gzip, os, numpy as np, argparse
from collections import defaultdict
from multiprocessing import Pool

from __config__.filepath import AUX_DIR

# Command line options

parser = argparse.ArgumentParser(description="Train a simple vector model")
parser.add_argument('C', type=int)
parser.add_argument('seed', type=int)
args = parser.parse_args()


# Hyperparameters

D = 400  # Number of dimensions for nouns and verbs (separately)
C = args.C  # Number of dimensions that are active
seed = args.seed
np.random.seed(seed)

# Dataset

dataset = 'multicore-5'

directory = os.path.join(AUX_DIR, 'simplevec_card')
if not os.path.exists(directory):
    os.makedirs(directory)
output_file = os.path.join(directory, '{}-{}-{}-full-{}.pkl.gz').format(dataset, D, C, seed)
if os.path.exists(output_file):
    raise FileExistsError

# Assign each context to a random bin
# By using a defaultdict, the bin is chosen when needed, and then cached

r = np.arange(D)
def rand_vec():
    "Return a random binary vector of fixed cardinality"
    v = np.zeros(D)
    i = np.random.choice(r, C, replace=False)
    v[i] = 1
    return v

get_vec = defaultdict(rand_vec)

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
        vec[p, :D] += n * get_vec[q,'b']
        vec[q, :D] += n * get_vec[p,'b']
    else:
        v, s, o = graph
        if s is not None:
            vec[v, D:] += n * get_vec[s,'s']
            vec[s, :D] += n * get_vec[v,'s']
        if o is not None:
            vec[v, D:] += n * get_vec[o,'o']
            vec[o, :D] += n * get_vec[v,'o']
# TODO include subject-object contexts?

# Save

print('saving')

with gzip.open(output_file, 'wb') as f:
    pickle.dump(vec, f)
