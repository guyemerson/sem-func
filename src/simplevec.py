import pickle, gzip, numpy as np
from collections import defaultdict

from utils import is_verb

# Hyperparameters

D = 400  # Number of dimensions for nouns and verbs (separately)
k = 0  # Negative offset for PPMI (equivalent to log of number of word2vec neg samples) 
a = 0.75  # Power that frequencies are raised to under the null hypothesis
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

with open('/anfs/bigdisc/gete2/wikiwoods/{}-count_tuple.pkl'.format(dataset), 'rb') as f:
    count = pickle.load(f)

with open('/anfs/bigdisc/gete2/wikiwoods/{}-vocab.pkl'.format(dataset), 'rb') as f:
    pred_name = pickle.load(f)
V = len(pred_name)
verb = np.array([is_verb(p) for p in pred_name], dtype='bool')

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

# Calculate PPMI

print('calculating ppmi')

# Smooth frequencies
freq_pred = vec.sum(1) ** a
freq_context = vec.sum(0) ** a

# Calculate marginal context probabilities (for noun and verb contexts separately)
freq_context[:D] /= freq_context[:D].sum()
freq_context[D:] /= freq_context[D:].sum()

# Calculate marginal predicate probabilities (for nouns and verbs separately)
freq_noun = vec * verb.reshape((-1,1))
freq_verb = vec * np.invert(verb).reshape((-1,1))
vec = freq_noun / freq_noun.sum() + freq_verb / freq_verb.sum()

# Take logs
log_pred = np.log(freq_pred / freq_pred.sum())
log_context = np.log(freq_context / freq_context.sum())
vec = np.log(vec)

# Subtract logs
vec -= k
vec -= log_context
vec -= log_pred.reshape((-1,1))

# Keep positive
vec.clip(0, out=vec)

# Save

print('saving')

template = '/anfs/bigdisc/gete2/wikiwoods/simplevec/{}-{}-{}-{}-{}.pkl.gz'
k_str = str(k).replace('.','')
a_str = str(a).replace('.','')

with gzip.open(template.format(dataset, D, k_str, a_str, seed), 'wb') as f:
    pickle.dump(vec, f)