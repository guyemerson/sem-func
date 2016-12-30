import os, pickle
from multiprocessing import Pool
from collections import Counter

DIR = '/anfs/bigdisc/gete2/wikiwoods/core-5-nodes'

filenames = os.listdir(DIR)

def count_file(fname):
    counts = (Counter(), Counter())
    with open(os.path.join(DIR, fname), 'rb') as f:
        data = pickle.load(f)
    for _, pred, out_labs, out_ids, _, _ in data:
        for i, lab in enumerate(out_labs):
            counts[lab][pred, data[out_ids[i]][1]] += 1
    return counts

total_counts = (Counter(), Counter())

with Pool(80) as p:
    for counts in p.imap_unordered(count_file, filenames):
        for i, c in enumerate(counts):
            total_counts[i].update(c)

with open('/anfs/bigdisc/gete2/wikiwoods/core-5-count.pkl', 'wb') as f:
    pickle.dump(total_counts, f)