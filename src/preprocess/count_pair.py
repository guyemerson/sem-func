import os, pickle
from collections import defaultdict

from __config__.filepath import AUX_DIR

prefix = 'multicore'
thresh = 5

with open(os.path.join(AUX_DIR, '{}-{}-count_tuple.pkl'.format(prefix,thresh)), 'rb') as f:
    full_count = pickle.load(f)

pair_count = [defaultdict(int), defaultdict(int)]

for graph, n in full_count.items():
    if len(graph) == 2:
        continue
    else:
        v, s, o = graph
        if s is not None:
            pair_count[0][v,s] += n
        if o is not None:
            pair_count[1][v,o] += n

with open(os.path.join(AUX_DIR, '{}-{}-count_pair.pkl'.format(prefix,thresh)), 'wb') as f:
    pickle.dump(pair_count, f)