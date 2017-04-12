import os, gzip, pickle, numpy as np

from variational import mean_field_vso, marginal_approx, get_semfunc
from __config__.filepath import AUX_DIR

def get_scoring_fn(pred_wei, pred_bias, C, meanfield_vecs):
    """
    Get a scoring function for the relpron dataset
    :param pred_wei: weights of semantic functions
    :param pred_bias: biases of semantic functions
    :param C: total cardinality
    :param meanfield_vecs: mean-field vectors for relpron triples
    :return: scoring function
    """
    # Set up semantic functions
    semfuncs = [get_semfunc(pred_wei[i], pred_bias[i]) for i in range(len(pred_wei))]
    # Get marginal distributions
    marg = [[marginal_approx(prob, C) for prob in triple] for triple in meanfield_vecs]
    
    def scoring_fn(term, description, **kwargs):
        """
        Calculate how much the triple implies the term
        :param term: noun index
        :param description: (index-of-SBJ-or-OBJ, index-of-triple)
        :return: probability
        """
        which, index = description
        return semfuncs[term](marg[index][which])
    
    return scoring_fn

def get_meanfield_fn(pred_wei, pred_bias, link_wei, ent_bias, C, init_vecs):
    """
    Get a function mapping vso triples to meanfield vectors
    :param pred_wei: weights of semantic functions
    :param pred_bias: biases of semantic functions
    :param link_wei: link weight matrix
    :param ent_bias: entity bias
    :param C: total cardinality
    :param init_vecs: zero-context mean-field vectors, by pred index
    :return: scoring function
    """
    # Set up semantic functions
    semfuncs = [get_semfunc(pred_wei[i], pred_bias[i]) for i in range(len(pred_wei))]
    # Set up constant function
    D = pred_wei.shape[1]
    constant = get_semfunc(np.zeros(D), 0)
    av_ent = np.ones(D) * (C/D)
    
    def meanfield_fn(triple, **kwargs):
        """
        Calculate meanfield vectors for the triple.
        For OOV items, the semfunc is a constant function.
        :param triple: (verb, agent, patient)
        :return: probability
        """
        sf = []
        vecs = []
        for i in triple:
            if i is None:
                sf.append(constant)
                vecs.append(av_ent)
            else:
                sf.append(semfuncs[i])
                vecs.append(init_vecs[i]) 
        meanfield_vecs = mean_field_vso(sf, link_wei, ent_bias, C=C, vecs=vecs, **kwargs)
        return meanfield_vecs
    
    return meanfield_fn

# TODO (above two functions): allow decreasing the bias for hypernyms
# TODO (top function only): combine with normal vectors for relatedness (could also rank separately and combine ranks)

def get_baseline_scoring_fn(pred_wei, pred_bias, C, ent_vecs):
    """
    Get a scoring function for the relpron dataset
    :param pred_wei: weights of semantic functions
    :param pred_bias: biases of semantic functions
    :param ent_vecs: zero-context mean-field vectors, by pred index
    :return: scoring function
    """
    # Set up semantic functions
    semfuncs = [get_semfunc(pred_wei[i], pred_bias[i]) for i in range(len(pred_wei))]
    
    def score(term, description, **kwargs):
        """
        Calculate how much the triple implies the target
        :param term: noun index
        :param description: (SBJ-or-OBJ, (verb, agent, patient))
        :return: probability
        """
        which, triple = description
        if which == 'SBJ':
            i = 1
        elif which == 'OBJ':
            i = 2
        else:
            raise ValueError(which)
        marg = marginal_approx(ent_vecs[triple[i]], C)
        return semfuncs[term](marg)
    
    return score

def load_model(name, pred_wei_dir='simplevec_all', link_wei_dir='meanfield_link', meanfield_dir='meanfield_all', basic_length=8, meanfield_length=13, C_index=9):
    """
    Load a model from file
    :param name: filename of full model (without file extension)
    :param pred_wei_dir: directory for pred weights
    :param link_wei_dir: directory for link weights
    :param meanfield_dir: directory for meanfield vectors
    :param basic_length: number of settings for predicate weights
    :param meanfield_length: number of settings for meanfield vectors and biases
    :param C_index: index of setting for cardinality
    :return: pred_wei, pred_bias, link_wei, ent_bias, C, init_vecs
    """
    parts = name.split('-')
    basic_name = '-'.join(parts[:basic_length])
    meanfield_name = '-'.join(parts[:meanfield_length])
    
    C = int(parts[C_index])
    
    with gzip.open(os.path.join(AUX_DIR, pred_wei_dir, basic_name+'.pkl.gz'), 'rb') as f:
        pred_wei = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, meanfield_name+'.pkl.gz'), 'rb') as f:
        init_vecs = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, meanfield_name+'-bias.pkl.gz'), 'rb') as f:
        pred_bias = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, link_wei_dir, name+'.pkl.gz'), 'rb') as f:
        link_wei = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, link_wei_dir, name+'-bias.pkl.gz'), 'rb') as f:
        ent_bias = pickle.load(f)
    
    return pred_wei, pred_bias, link_wei, ent_bias, C, init_vecs

def load_baseline_model(name, pred_wei_dir='simplevec_all', meanfield_dir='meanfield_all', basic_length=8, C_index=9):
    """
    Load a model from file
    :param name: filename of full model (without file extension)
    :param pred_wei_dir: directory for pred weights
    :param link_wei_dir: directory for link weights
    :param meanfield_dir: directory for meanfield vectors
    :param basic_length: number of settings for predicate weights
    :param meanfield_length: number of settings for meanfield vectors and biases
    :param C_index: index of setting for cardinality
    :return: pred_wei, pred_bias, link_wei, ent_bias, C, init_vecs
    """
    parts = name.split('-')
    basic_name = '-'.join(parts[:basic_length])
    
    C = int(parts[C_index])
    
    with gzip.open(os.path.join(AUX_DIR, pred_wei_dir, basic_name+'.pkl.gz'), 'rb') as f:
        pred_wei = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, name+'.pkl.gz'), 'rb') as f:
        init_vecs = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, name+'-bias.pkl.gz'), 'rb') as f:
        pred_bias = pickle.load(f)
    
    return pred_wei, pred_bias, C, init_vecs

def load_scoring_fn(name, pred_wei_dir='simplevec_all', bias_dir='meanfield_all', meanfield_dir='meanfield_relpron', basic_length=8, bias_length=13, C_index=9):
    """
    Load a scoring function from file
    :param name: filename of full model (without file extension)
    :param pred_wei_dir: directory for pred weights
    :param bias_dir: direcotory for biases
    :param meanfield_dir: directory for meanfield vectors
    :param basic_length: number of settings for predicate weights
    :param bias_length: number of settings for biases
    :param C_index: index of setting for cardinality
    :return: scoring function
    """
    parts = name.split('-')
    basic_name = '-'.join(parts[:basic_length])
    bias_name = '-'.join(parts[:bias_length])
    
    C = int(parts[C_index])
    
    with gzip.open(os.path.join(AUX_DIR, pred_wei_dir, basic_name+'.pkl.gz'), 'rb') as f:
        pred_wei = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, bias_dir, bias_name+'-bias.pkl.gz'), 'rb') as f:
        pred_bias = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, name+'.pkl.gz'), 'rb') as f:
        vecs = pickle.load(f)
    
    return get_scoring_fn(pred_wei, pred_bias, C, vecs)


def load_baseline_scoring_fn(*args, **kwargs):
    """
    Load a baseline scoring function from file.
    All arguments are passed to load_baseline_model
    :return: scoring function
    """
    return get_baseline_scoring_fn(*load_baseline_model(*args, **kwargs))


if __name__ == "__main__":
    from testing import get_relpron_separated, get_GS2011_indexed, load_freq_lookup_dicts
    
    LINK_DIR = 'meanfield_link'
    OUTPUT_DIR = 'meanfield_relpron_test'
    
    # Choose dataset
    #raw_triples, _ = get_relpron_separated()
    raw_triples, _ = get_relpron_separated(True)
    #raw_triples, _ = get_GS2011_indexed()    
    
    # Convert to indices
    lookup = load_freq_lookup_dicts()
    props = [(which, (lookup['v'].get(verb),
                      lookup['n'].get(agent),
                      lookup['n'].get(patient)))
             for which, (verb, agent, patient) in raw_triples]
    
    def apply_model(filename, bias_shift):
        "Calculate meanfield vectors for a given model"
        if os.path.exists(os.path.join(AUX_DIR, OUTPUT_DIR, filename+'.pkl.gz')):
            return
        # Load model
        fullname = filename + '-' + str(bias_shift).replace('.','_').replace('-','~')
        print('loading', filename, bias_shift)
        params = list(load_model(filename, link_wei_dir=LINK_DIR))
        params[3] -= bias_shift
        meanfield_fn = get_meanfield_fn(*params)
        # Get meanfield vectors
        print('calculating', fullname)
        vecs = [meanfield_fn(triple, max_iter=500) for _, triple in props]
        # Save vectors
        print('saving', fullname)
        with gzip.open(os.path.join(AUX_DIR, OUTPUT_DIR, fullname+'.pkl.gz'), 'wb') as f:
            pickle.dump(vecs, f)
    
    apply_model('multicore-5-400-0-1-0-0-32-1_0-30-0_01-1_0-4_0-0_5-0_2', 0.5)
    
    """
    # Process files
    from multiprocessing import Pool
    from random import shuffle
    from itertools import product
    
    files = []
    for fullname in os.listdir(os.path.join(AUX_DIR, LINK_DIR)):
        name = fullname.split('.')[0]
        if name.split('-')[-1] not in ['raw', 'bias']:
            files.append(name)
    files_and_shifts = list(product(files, [0.5, 0.8]))
    shuffle(files_and_shifts)
    
    with Pool(16) as p:
        p.starmap(apply_model, files_and_shifts)
    """
