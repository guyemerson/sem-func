import pickle, numpy as np

# Collapse triples with _be_v_id to a single node and two preds
# (discarding if _be_v_id has only one argument)
# and randomise the order of graphs

# Load data

print('loading')

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-count_tuple.pkl', 'rb') as f:
    count = pickle.load(f)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    pred_name = pickle.load(f)
ind = {p:i for i,p in enumerate(pred_name)}
V = len(pred_name)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl', 'rb') as f:
    pred_freq = pickle.load(f)

# Filter the triples

print('filtering')

multi_count = {}

be = ind['_be_v_id']

for triple, n in count.items():
    v,s,o = triple
    if v != be:
        multi_count[triple] = n
    elif s is None:
        pred_freq[o] -= n
    elif o is None:
        pred_freq[s] -= n
    else:
        multi_count[(s,o)] = n

del count
pred_freq[be] = 0

# Ensure minimum threshold

print('thresholding')

thresh = 5

# Whether preds should be kept
keep = np.ones(V, dtype='bool')
keep[be] = 0

min_count = 0

while min_count < thresh:
    # Find preds below threshold
    below_arr = (pred_freq < thresh) * keep
    below_set = set(below_arr.nonzero()[0])
    keep[below_arr] = 0
    print('below threshold:')
    for p in below_set:
        print('\t', pred_name[p], pred_freq[p])
    # Remove graphs with these preds, and update freq
    remove = []
    for graph, n in multi_count.items():
        if any(p in below_set for p in graph):
            remove.append(graph)
            for p in graph:
                if p is not None:
                    pred_freq[p] -= n
    for graph in remove:
        multi_count.pop(graph)
    # Find new smallest count
    min_count = pred_freq[keep.nonzero()].min()

# Condense data

print('re-assigning indices')

old_inds = keep.nonzero()[0]
pred_name = [pred_name[p] for p in old_inds]
pred_freq = pred_freq[old_inds]

convert = {old:new for new, old in enumerate(old_inds)}
convert[None] = None
multi_count = {tuple(convert[p] for p in graph):n for graph, n in multi_count.items()}

# Save aggregate data

print('saving')

with open('/anfs/bigdisc/gete2/wikiwoods/multicore-5-count_tuple.pkl', 'wb') as f:
    pickle.dump(multi_count, f)

with open('/anfs/bigdisc/gete2/wikiwoods/multicore-5-vocab.pkl', 'wb') as f:
    pickle.dump(pred_name, f)

with open('/anfs/bigdisc/gete2/wikiwoods/multicore-5-freq.pkl', 'wb') as f:
    pickle.dump(pred_freq, f)
