import os, pickle
from collections import Counter
from multiprocessing import Pool  # @UnresolvedImport

MIN = 10000

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-100'
OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/core-' + str(MIN)

PROC = 80

all_names = sorted(os.listdir(DATA))

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

global_directory_holder = [DATA]
global_skip = {None}    

def get_preds(triples):
    "Yield the preds in a triple, skipping None"
    for t in triples:
        for p in t:
            if p not in global_skip:
                yield p

def count_file(fname):
    "Count preds in a file, with mutable references"
    with open(os.path.join(global_directory_holder[0], fname),'rb') as f:
        print('count', fname)
        data = pickle.load(f)
    return Counter(get_preds(data))

def count_preds():
    "Count how many times each pred appears"
    count = Counter()
    # Iterate through files in the data directory
    with Pool(PROC) as p:
        for sub_count in p.imap_unordered(count_file, all_names):
            # imap_unordered ensures that we process results as they are ready
            # Combine results:
            count.update(sub_count)
            print(len(count))
    return count

def filter_preds(triples):
    "Yield triples, skipping certain ones (but ignoring None)"
    for t in triples:
        verb, agent, patient = t
        if verb in global_skip: continue
        if agent and agent in global_skip: continue
        if patient and patient in global_skip: continue
        yield t

def filter_file(fname):
    "Filter a file and write to another"
    print('filter', fname)
    in_file = os.path.join(global_directory_holder[0], fname)
    out_file = os.path.join(OUTPUT, fname)
    with open(in_file, 'rb') as f:
        triples = pickle.load(f)
    filtered = list(filter_preds(triples))
    with open(out_file, 'wb') as f:
        pickle.dump(filtered, f)

# Count the preds, filter, and repeat
while True:
    # Count and filter preds below the cutoff
    count = count_preds()
    newskip = {p for p,n in count.items() if n < MIN}
    # If nothing's below the cutoff, we're done
    if not newskip:
        print("Done!")
        break
    # Otherwise, filter the files
    print("Filtering...")
    global_skip.update(newskip)
    # Run in multiple processes
    with Pool(PROC) as p:
        p.map(filter_file, all_names)
        #!#p.map(process_file, sorted(os.listdir(global_directory_holder[0])))
    # Use the OUTPUT directory for future iterations
    global_directory_holder[0] = OUTPUT
