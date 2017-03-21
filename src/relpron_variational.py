import os, gzip, pickle, numpy as np

from variational import mean_field_vso, marginal_approx, get_semfunc
from __config__.filepath import AUX_DIR

def get_scoring_fn(pred_wei, pred_bias, link_wei, ent_bias, C, init_vecs):
    """
    Get a scoring function for the relpron dataset
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
    
    def score(term, description, **kwargs):
        """
        Calculate how much the triple implies the target
        :param term: noun index
        :param description: (SBJ-or-OBJ, (verb, agent, patient))
        :return: probability
        """
        which, triple = description
        sf = [semfuncs[i] for i in triple]
        vecs = [init_vecs[i] for i in triple]
        meanfield = mean_field_vso(sf, link_wei, ent_bias, C=C, vecs=vecs, **kwargs)
        marg = [marginal_approx(prob, C) for prob in meanfield]
        if which == 'SBJ':
            return semfuncs[term](marg[1])
        elif which == 'OBJ':
            return semfuncs[term](marg[2])
        else:
            raise ValueError(which)
    
    return score

# TODO allow decreasing the bias for hypernyms
# TODO combine with normal vectors for relatedness (could also rank separately and combine ranks)

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

def load_scoring_fn(*args, **kwargs):
    """
    Load a scoring function from file.
    All arguments are passed to load_model
    :return: scoring function
    """
    return get_scoring_fn(*load_model(*args, **kwargs))


def load_baseline_scoring_fn(*args, **kwargs):
    """
    Load a baseline scoring function from file.
    All arguments are passed to load_baseline_model
    :return: scoring function
    """
    return get_baseline_scoring_fn(*load_baseline_model(*args, **kwargs))