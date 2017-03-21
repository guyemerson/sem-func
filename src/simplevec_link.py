import os, pickle, gzip, numpy as np

from __config__.filepath import AUX_DIR

def observe_links(filename, input_dir='meanfield_all', output_dir='meanfield_link', C_index=9):
    """
    Observe connections between meanfield entities, and save observed frequencies to file
    :param filename: filename without file extension
    :param input_dir: directory of meanfield vectors
    :param output_dir: directory to save observed link frequencies
    :param C_index: index of cardinality in filename parameters
    """
    # Parameters
    
    hyp = filename.split('-')
    D = int(hyp[2])*2
    C = int(hyp[C_index])
    
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
    
    # Add connections from each tuple
    for label, label_pairs in enumerate(pairs):
        for (verb, noun), n in label_pairs.items():
            link_total[label] += n * np.outer(ent[verb], ent[noun])
            n_total[label] += n
    
    # Normalise to frequencies
    link_freq = link_total / n_total.reshape((-1, 1, 1))
    
    # Save all parameters to file
    
    with gzip.open(os.path.join(AUX_DIR, output_dir, filename+'-raw.pkl.gz'), 'wb') as f:
        pickle.dump(link_freq, f)

if __name__ == "__main__":
    # Command line options
    import argparse
    parser = argparse.ArgumentParser(description="Fit link weights")
    parser.add_argument('filename')
    args = parser.parse_args()
    
    observe_links(args.filename)