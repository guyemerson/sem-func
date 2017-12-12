import os, pickle, gzip, numpy as np
from itertools import islice
from multiprocessing import Pool

from __config__.filepath import AUX_DIR

# The following two functions are used by observe_links
def get_work_fn(D, ent):
    "Define a work function in each worker process"
    global work
    def work(data):
        "Observe a set of linked pairs"
        link_subtotal = np.zeros((D,D))
        n_subtotal = 0
        for (verb, noun), n in data:
            link_subtotal += n * np.outer(ent[verb], ent[noun]).clip(0)
            n_subtotal += n
        return link_subtotal, n_subtotal

def do_work(data):
    "Use the work function that was initialised"
    return work(data)

def observe_links(filename, input_dir='meanfield_all', output_dir='meanfield_link', C_index=9, chunk_size=100, processes=32):
    """
    Observe connections between meanfield entities, and save observed frequencies to file
    :param filename: filename without file extension
    :param input_dir: directory of meanfield vectors
    :param output_dir: directory to save observed link frequencies
    :param C_index: index of cardinality in filename parameters
    """
    # Skip if it's already been processed
    output_name = os.path.join(AUX_DIR, output_dir, filename+'-raw.pkl.gz')
    if os.path.exists(output_name):
        return
    
    # Parameters
    
    hyp = filename.split('-')
    D = int(hyp[2])*2
    C = int(hyp[C_index])
    
    if hyp[-1] == 'bias':
        return
    if D != 2000:
        return
    
    # Load vectors 
    
    with gzip.open(os.path.join(AUX_DIR, input_dir, filename+'.pkl.gz'), 'rb') as f:
        ent = pickle.load(f)
    
    # Force cardinality
    ent /= (ent.sum(1)/C).reshape((-1, 1))
    np.clip(ent, 0, 1, ent)
    
    ### Link weights
    
    # Sum over tuples
    
    with open(os.path.join(AUX_DIR, '{}-{}-count_pair.pkl'.format(*hyp[:2])), 'rb') as f:
        pairs = pickle.load(f)
    
    print(len(pairs[0])+len(pairs[1]), "pairs in total")
    
    link_total = np.zeros((2, D, D))  # Store sum of outer products
    n_total = np.zeros(2)  # Store number of observations
    
    # Multiprocessing to process tuples
    # Chunk pairs
    def chunk(data):
        "Chunk a dictionary into lists of items"
        iterator = iter(data.items())
        for _ in range(0, len(data), chunk_size):
            yield list(islice(iterator, chunk_size))
    # Spawn workers
    for label, label_pairs in enumerate(pairs):
        with Pool(processes, get_work_fn, (D, ent)) as p:
            for link_subtotal, n_subtotal in p.imap_unordered(do_work, chunk(label_pairs)):
                link_total[label] += link_subtotal
                n_total[label] += n_subtotal
    
    # Normalise to frequencies
    link_freq = link_total / n_total.reshape((-1, 1, 1))
    
    # Save all parameters to file
    
    with gzip.open(output_name, 'wb') as f:
        pickle.dump(link_freq, f)

if __name__ == "__main__":
    # Command line options
    import argparse
    parser = argparse.ArgumentParser(description="Fit link weights")
    parser.add_argument('filename', nargs='?', default=None)
    args = parser.parse_args()
    
    if args.filename:
        observe_links(args.filename)
    
    else:
        for filename_ext in os.listdir(os.path.join(AUX_DIR, 'meanfield_all')):
            observe_links(filename_ext.split('.')[0])
