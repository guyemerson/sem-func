import os, pickle, gzip, numpy as np

from variational import get_semfunc, mean_field
from testing import get_simlex_wordsim_preds
from __config__.filepath import AUX_DIR

def get_entities(scale, C, target, name=None, prefix='multicore', thresh=5, dim=400, k=0, a=0.75, seed=32, pred_list=None, mean_field_kwargs=None, skip_if_exists=True, verbose=False):
    """
    Get mean field entity vectors based on given parameter vectors
    Hyperparameters of binary-valued model:
    :param scale: factor to multiply simple vectors by
    :param C: total cardinality
    :param target: desired energy for an 'untypical' vector
    Simple vector model to load:
    :param name: name of model, as a string, or else use the following:
    :param prefix: name of dataset
    :param thresh: frequency threshold
    :param dim: number of dimensions (D/2)
    :param k: negative shift
    :param a: power that frequencies are raised to
    :param seed: random seed
    Other:
    :param pred_list: list of predicates to use
    :param verbose: print messages
    :return: {pred: entity vector}
    """
    # File names
    if name is None:
        name = '-'.join(str(x).replace('.','')
                        for x in (prefix, thresh, dim, k, a, seed))
    else:
        prefix, thresh, dim, k, a, seed = name.split('-')
        dim = int(dim)
    fullname = name + '-' + '-'.join(str(x).replace('.','')
                                 for x in (scale, C, target))
    
    # Skip if this setup has already been calculated
    if skip_if_exists and os.path.exists(os.path.join(AUX_DIR, 'meanfield', fullname+'.pkl')):
        return
    
    # Predicates to use
    if pred_list is None:
        preds, _ = get_simlex_wordsim_preds(prefix, thresh)
        pred_list = sorted(preds)
    
    # Load model
    if verbose: print("Loading model")
    with gzip.open(os.path.join(AUX_DIR, 'simplevec', name+'.pkl.gz'), 'rb') as f:
        all_vectors = pickle.load(f)    
    # Multiply by the scale factor
    vec = all_vectors[pred_list] * scale
    del all_vectors  # Conserve memory
    
    # Define bias of predicates
    
    # Number of dimensions in full space is double noun or verb space
    D = dim * 2
    # Maximum activation of each predicate
    high = np.partition(vec, -C, axis=1)[:,-C:].sum(axis=1)
    # Average activation if the top units are not used but still nouny/verby
    other = (vec.sum(1) - high) / (D/2 - C) * C
    # Minimum pred bias makes an average predicate have the target energy
    bias = other + target
    # For preds with a bigger gap between max and other activation,
    # make the bias the average of the two
    gap = high - other
    mask = (gap > 2 * target)
    bias[mask] = other[mask] + gap[mask] / 2
    
    # Define semantic functions
    
    semfuncs = [get_semfunc(v,b) for v,b in zip(vec, bias)]
    
    # Calculate entity vectors
    
    if verbose: print("Calculating entity vectors")
    if mean_field_kwargs is None:
        mean_field_kwargs = {}
    ent = {}
    for i, sf in zip(pred_list, semfuncs):
        if verbose: print(i)
        ent[i] = mean_field(sf, C, **mean_field_kwargs)
    
    # Save to disk
    
    with open(os.path.join(AUX_DIR, 'meanfield', fullname+'.pkl'), 'wb') as f:
        pickle.dump(ent, f)
    
    return ent


if __name__ == "__main__":
    from itertools import product
    from multiprocessing import Pool
    from random import shuffle
    
    # Grid search over hyperparameters
    
    scales = [0.5, 0.8, 1, 1.2]
    Cs = [20, 40, 80]
    targets = [10, 20, 40]
    
    grid = product(scales, Cs, targets)
    
    # Vector models
    
    simplevec = os.listdir(os.path.join(AUX_DIR, 'simplevec'))
    simplevec_filtered = []
    for name in simplevec:
        parts = name.split('-')
        if len(parts) != 6:
            continue
        prefix, thresh, dim, *_ = parts
        if prefix == 'multicore' and thresh == '5' and dim == '400':
            simplevec_filtered.append(name.split('.')[0])
    
    full_grid = list(product(grid, simplevec_filtered))
    shuffle(full_grid)
    
    def train(hyper, simple):
        print(hyper, simple)
        get_entities(*hyper, name=simple, mean_field_kwargs={"max_iter":500})
    
    with Pool(4) as p:
        p.starmap(train, full_grid)
