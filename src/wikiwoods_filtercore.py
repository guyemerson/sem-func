import sys, os, pickle
from collections import Counter

MIN = 5

DATA = '/anfs/bigdisc/gete2/wikiwoods/core'
OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/core-' + str(MIN)

def get_preds(triples, skip):
    "Yield the preds in a triple, skipping certain ones"
    for t in triples:
        for p in t:
            if p not in skip:
                yield p

def count_preds(directory, skip):
    "Count how many times each pred appears"
    count = Counter()
    # Iterate through files in the data directory
    for filename in sorted(os.listdir(directory)):
        with open(os.path.join(directory, filename),'rb') as f:
            print('count', filename)
            data = pickle.load(f)
            count.update(get_preds(data, skip))
            print(len(count))
    return count

def filter_preds(triples, skip):
    "Yield triples, skipping certain ones (but ignoring None)"
    for t in triples:
        verb, agent, patient = t
        if verb in skip: continue
        if agent and agent in skip: continue
        if patient and patient in skip: continue
        yield t

def filter_file(in_file, out_file, skip):
    print('filter', filename)
    with open(in_file, 'rb') as f:
        triples = pickle.load(f)
    filtered = list(filter_preds(triples, skip))
    with open(out_file, 'wb') as f:
        pickle.dump(filtered, f)

# Count the preds, filter, and repeat
datadir = DATA
skip = {None}

while True:
    # Count and fild preds below the cutoff
    count = count_preds(datadir, skip)
    newskip = {p for p,n in count.items() if n < MIN}
    # If nothing's below the cutoff, we're done
    if not newskip:
        break
    # Otherwise, filter the files
    skip.update(newskip)
    for filename in sorted(os.listdir(datadir)):
        in_file = os.path.join(datadir, filename)
        out_file = os.path.join(OUTPUT, filename)
        filter_file(in_file, out_file, skip)
    # Use the OUTPUT directory for future iterations
    datadir = OUTPUT