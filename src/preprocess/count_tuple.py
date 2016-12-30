import os, pickle
from multiprocessing import Pool
from collections import Counter

DIR = '/anfs/bigdisc/gete2/wikiwoods/core-5'

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl', 'rb') as f:
    pred_name = pickle.load(f)
ind = {p:i for i,p in enumerate(pred_name)}

filenames = os.listdir(DIR)

def count_file(fname):
    count = Counter()
    with open(os.path.join(DIR, fname), 'rb') as f:
        data = pickle.load(f)
    for triple in data:
        t = tuple([ind[str(x)] if x else None for x in triple])
        count[t] += 1
    return count

total_count = Counter()

with Pool(40) as p:
    for c in p.imap_unordered(count_file, filenames):
        total_count.update(c)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-count_tuple.pkl', 'wb') as f:
    pickle.dump(total_count, f)
