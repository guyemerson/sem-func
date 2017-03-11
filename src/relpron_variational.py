import os, gzip, pickle, numpy as np

from variational import mean_field_vso, marginal_approx
from simplevec_to_entity import get_semfuncs_from_vectors, get_verb_noun_freq
from testing import get_relpron_preds
from __config__.filepath import AUX_DIR

def get_scoring_fn(semfuncs, link_wei, ent_bias, C, init_vecs):
    """
    Get a scoring function for the relpron dataset
    :param semfuncs: semantic functions by pred index
    :param link_wei: link weight matrix
    :param ent_bias: entity bias
    :param C: total cardinality
    :param init_vecs: zero-context mean-field vectors, by pred index
    :return: scoring function
    """
    def score(which, target, triple, **kwargs):
        """
        Calculate how much the triple implies the target
        :param which: SBJ or OBJ
        :param target: noun
        :param triple: verb, agent, patient
        :return: probability
        """
        sf = [semfuncs[i] for i in triple]
        vecs = [init_vecs[i] for i in triple]
        meanfield = mean_field_vso(sf, link_wei, ent_bias, C=C, vecs=vecs, **kwargs)
        marg = [marginal_approx(prob, C) for prob in meanfield]
        if which == 'SBJ':
            return semfuncs[target](marg[1])
        else:
            return semfuncs[target](marg[2])
    
    return score
    
def load_scoring_fn(fname, pred_wei_dir='simplevec', link_wei_dir='simplevec_link', meanfield_dir='meanfield_freq'):
    """
    Load a scoring function from file
    :param fname: filename of full model (without file extension)
    :param pred_wei_dir: directory for pred weights
    :param link_wei_dir: directory for link weights
    :param meanfield_dir: directory for meanfield vectors
    :return: scoring function
    """
    parts = fname.split('-')
    simple_name = '-'.join(parts[:6])
    
    prefix, thresh, dims, *_ = parts[:6]
    thresh = int(thresh)
    dims = int(dims)
    preds, _ = get_relpron_preds(prefix, thresh, include_test=False)
    pred_list = sorted(preds)
    freq = get_verb_noun_freq(prefix, thresh, pred_list)
    
    scale, C, Z, alpha = parts[6:]
    C = int(C)
    scale, Z, alpha = [float(x[0]+'.'+x[1:]) for x in [scale, Z, alpha]]
    
    semfuncs = get_semfuncs_from_vectors(simple_name, 'frequency', scale, C, Z=Z, alpha=alpha, pred_list=pred_list, freq=freq, as_dict=True, directory=pred_wei_dir)
    
    with open(os.path.join(AUX_DIR, link_wei_dir, fname+'.pkl'), 'rb') as f:
        link_wei = pickle.load(f)
    with gzip.open(os.path.join(AUX_DIR, meanfield_dir, fname+'.pkl.gz'), 'rb') as f:
        init_vecs = pickle.load(f)
    
    ent_bias = link_wei.mean() * 4 * C + np.log(dims/C - 1)
    
    return get_scoring_fn(semfuncs, link_wei, ent_bias, C, init_vecs)
