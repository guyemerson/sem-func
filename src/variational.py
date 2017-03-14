import numpy as np
from scipy.special import expit
from math import ceil

# Given a predicate's semantic function, calculate a distribution over entity vectors,
# approximating the posterior distribution over entities, given that the predicate is true

### Initialisation

def get_semfunc(pred_wei, pred_bias):
    """
    Define a semantic function, based on a set of weights, and a bias
    :param pred_wei: weight for each entity dimension
    :param pred_bias: bias weight
    :return: semantic function
    """
    def prob(ent):
        """
        Calculate the probability of a predicate being true of an entity
        :param ent: an entity vector
        :return: a probability
        """
        return expit(np.dot(ent, pred_wei) - pred_bias)
    # Make weights accessible
    prob.wei = pred_wei
    prob.bias = pred_bias
    # Record zero and nonzero indices, for convenience
    prob.nonzero = pred_wei.nonzero()[0]
    prob.zero = (pred_wei == 0).nonzero()[0]
    return prob

def init_vec(pred_wei, C=None, max_value=0.5):
    """
    Return a reasonably sparse entity vector for a parameter vector 
    :param pred_wei: a predicate parameter vector
    :param C: total cardinality
    :param max_value: maximum value for each dimension
    :return: an entity vector
    """
    D = pred_wei.size
    if C is None:
        C = ceil(D/20)
    
    vec = np.zeros(D)
    nonzero = pred_wei.nonzero()[0]  # Indices on nonzero elements
    if nonzero.size * max_value > C:
        # Just get the largest weights, if there are too many
        max_indices = ceil(C/max_value)
        indices = np.argpartition(pred_wei, -max_indices)
        vec[indices[-max_indices+1:]] = max_value
        vec[indices[-max_indices]] = C - (max_indices-1)*max_value
        vec[np.argpartition(pred_wei, -C-1)[-C-1:]] = max_value
    else:
        # Get all nonzero weights, if there are too few
        vec += (C - nonzero.size*max_value) / (D - nonzero.size) 
        vec[nonzero] = max_value
    return vec

### Message passing for a cardinality potential

def pass_messages_up(prob, C):
    """
    Pass messages for belief propagation from individual components to the cardinality potential
    :param prob: non-sparse probabilities of each component
    :param C: total cardinality
    :return: cardinality probabilities for successive subsets of components
    """
    D = prob.size
    intermed = np.zeros((D-1, C+1))
    # intermed[i,j] is the probability, ignoring the cardinality potential,
    # that the units from 0 to i have total cardinality j 
    intermed[0,0] = 1-prob[0]
    intermed[0,1] = prob[0]
    # TODO This loop is a bottleneck, because it's done in Python rather than NumPy or C.
    # (We call this function often enough that it matters)
    # It's not easy to vectorise in NumPy, because intermed[i] depends on intermed[i-1]
    for i in range(1, D-1):
        intermed[i] = intermed[i-1] * (1 - prob[i])  # unit i is off
        intermed[i,1:] += intermed[i-1,:-1] * prob[i]  # unit i is on
    return intermed

def pass_messages_down(prob, intermed, C):
    """
    Calculate marginal probabilities, using the cardinality probabilities
    :param prob: non-sparse probabilities of each component
    :param intermed: cardinality probabilities for successive subsets of components
    :param C: total cardinality
    :return: marginal probabilities
    """
    D = prob.size
    vec = np.empty(D)  # Output vector
    # Distribution over number of components still to come (between 0 and C, inclusive)
    aux = np.zeros(C+1)
    aux[C] = 1
    # Iteratively calculate marginal probability
    for i in range(D-1, 0, -1):  # [D-1, D-2, ..., 1]
        p = prob[i]
        # Unnormalised probabilities of being on or off, for each possible number of units to come:
        aus = (1-p) * intermed[i-1]
        ein = np.empty(C+1)
        ein[1:] = p * intermed[i-1][:-1]
        ein[0] = 0
        # Probability of being on, for each possible number of units to come
        on = np.nan_to_num(ein / (ein+aus))  # TODO deal with nan?
        # Marginalise out number of possible units
        vec[i] = np.dot(on, aux)
        # Update distribution over number of units to come
        new_aux = (1-on) * aux
        new_aux[:-1] += on[1:] * aux[1:]
        aux = new_aux
    # For the final unit, we don't need to marginalise over other units
    vec[0] = aux[1]
    return vec

def marginal(prob, C):
    """
    Calculate marginal probabilities, for a fixed cardinality
    :param prob: probabilities of each unit, without cardinality potential
    :param C: total cardinality
    :return: marginal probabilities
    """
    intermed = pass_messages_up(prob, C)
    return pass_messages_down(prob, intermed, C)

### Approximate marginals for cardinality potential

def marginal_approx(prob, C):
    """
    Calculate approximate marginal probabilities, for a fixed cardinality
    :param prob: probabilities of each unit, without cardinality potential
    :param C: total cardinality
    :return: marginal probabilities
    """
    scaled = prob / prob.sum() * C
    np.clip(scaled, 0, 1, scaled)
    # (This may leave the total below C)
    return scaled

### Optimisation

def new_value(vec, semfunc, i, C, prob_ratio=None):
    """
    Optimise the value of one component of the posterior, holding the rest fixed
    :param vec: current approximation of the posterior
    :param semfunc: semantic function
    :param i: index of component to optimise
    :param C: total cardinality
    :return: new value for vec[i]
    """
    D = vec.size
    # Vector with the unit switched on
    on = np.copy(vec)
    on[i] = 1
    # Vector with the unit switched off
    off = np.copy(vec)
    off[i] = 0
    # Marginal distributions when restricting the cardinality
    on_marg = marginal(on, C)
    off_marg = marginal(off, C)
    # Truth of the semantic function when applied to the mean entity vector
    on_truth = semfunc(on_marg)
    off_truth = semfunc(off_marg)
    factor = on_truth/off_truth
    # If potentials are given, modify the ratio
    if prob_ratio is not None:
        factor *= prob_ratio[i]
    # Optimal update (under our assumptions)
    return 1 / (1 + (D-C)/(C * factor))

def new_value_approx(vec, semfunc, i, C, prob_ratio=None):
    """
    Optimise the value of one component of the posterior, holding the rest fixed,
    and using a simple approximation for the marginals
    :param vec: current approximation of the posterior
    :param semfunc: semantic function
    :param i: index of component to optimise
    :param C: total cardinality
    :param prob_ratio: ratio of probabilities of each each dimension being on or off
    :return: new value for vec[i]
    """
    D = vec.size
    # Vector with the unit switched off
    off = np.copy(vec)
    off[i] = 0
    # Approximate marginal distributions when restricting the cardinality
    off_marg = marginal_approx(off, C)
    on_marg = marginal_approx(off, C-1)
    on_marg[i] = 1
    # Truth of the semantic function when applied to the mean entity vector
    on_truth = semfunc(on_marg)
    off_truth = semfunc(off_marg)
    factor = on_truth/off_truth
    # If potentials are given, modify the ratio
    if prob_ratio is not None:
        factor *= prob_ratio[i]
    # Optimal update (under our assumptions)
    return 1 / (1 + (D-C)/(C * factor))

def update(vec, semfunc, C, new_value_fn, prob_ratio=None):
    """
    Calculate probability of each unit, with the current mean field approximation,
    then calculate marginals with fixed cardinality.
    Repeat this separately for all units.
    :param vec: entity vector
    :param semfunc: semantic function
    :param C: total cardinality
    :param new_value_fn: function giving optimal updates for individual components
    :param prob_ratio: ratio of probabilities of each each dimension being on or off
    :return: updated entity vector
    """
    new = np.copy(vec)
    # Update each component of the posterior
    # Updates are identical for all components with zero weight
    # If prob_ratio is given, we also need to make sure these components have the same value
    if prob_ratio is None:
        zero = semfunc.zero
        nonzero = semfunc.nonzero
    else:
        zero_bool = (semfunc.wei == 0) * (prob_ratio == prob_ratio.min())
        zero = zero_bool.nonzero()[0]
        nonzero = np.invert(zero_bool).nonzero()[0]
    
    if zero.size:
        new[zero] = new_value_fn(new, semfunc, zero[0], C, prob_ratio=prob_ratio)
    # Update non-zero entries
    for i in nonzero:
        new[i] = new_value_fn(new, semfunc, i, C, prob_ratio=prob_ratio)
    return new

def mean_field(semfunc, C, prob_ratio=None, max_iter=50, delta=10**-4, init_max_value=0.5, new_value_fn=new_value_approx, verbose=False):
    """
    Calculate the posterior distribution over entities, under a mean field approximation
    :param semfunc: semantic function
    :param C: total cardinality
    :param prob_ratio: ratio of probabilities of each each dimension being on or off
    :param max_iter: stop after this number of iterations
    :param delta: stop when the max difference between iterations is less than this
    :param init_max_value: max value when initialising the vector
    :param new_value_fn: function giving optimal updates for individual components
    :param verbose: print summary information during descent
    :return: mean field entity vector
    """
    # Initialise the entity vector
    vec = init_vec(semfunc.wei, C, max_value=init_max_value)
    if verbose:
        print('initial prob', semfunc(vec))
        print('nonzero entries', len(semfunc.nonzero))
    # Optimise the entity vector
    for _ in range(max_iter):
        new = update(vec, semfunc, C, new_value_fn=new_value_fn, prob_ratio=prob_ratio)
        diff = np.abs(new - vec).max()
        
        if verbose:
            print('diff inc', (new - vec).max())
            print('diff dec', (new - vec).min())
            print('diff 000', new.min() - vec.min())
            print('min', new.min())
            print('max', new.max())
            print('sum', new.sum())
            print('prob', semfunc(new))
        
        vec = new
        if diff < delta:
            break
    else:
        print('max iter reached with delta {}'.format(diff))
         
    return vec

def mean_field_vso(semfuncs, link_wei, ent_bias, C, max_iter=50, delta=10**-4, init_max_value=0.5, vecs=None, new_value_fn=new_value_approx, verbose=False):
    """
    Calculate the posterior distribution over entities, under a mean field approximation,
    for a VSO triple
    :param semfuncs: semantic functions (verb, agent, patient)
    :param link_wei: link weights
    :param ent_bias: bias for invididual dimensions
    :param C: total cardinality
    :param max_iter: stop after this number of iterations
    :param delta: stop when the max difference between iterations is less than this
    :param init_max_value: max value when initialising the vector
    :param vecs: vectors to initialise approximation with
    :param new_value_fn: function giving optimal updates for individual components
    :param verbose: print summary information during descent
    :return: mean field entity vector
    """
    names = ['verb','agent','patient']
    # Initialise the entity vector
    if vecs is None:
        vecs = [init_vec(sf.wei, C, max_value=init_max_value) for sf in semfuncs]
    if verbose:
        for nm, sf, v in zip(names, semfuncs, vecs):
            print(nm)
            print('\tinitial prob', sf(v))
            print('\tnonzero entries', len(sf.nonzero))
    # Optimise the entity vector
    for _ in range(max_iter):
        max_diff = 0
        for i in range(3):
            if verbose:
                print(names[i])
            v = vecs[i]
            sf = semfuncs[i]
            
            marg = [marginal_approx(p, C) for p in vecs]
            # Get probability ratio from link weights
            if i == 0:
                prob_ratio = np.exp(np.dot(link_wei[0], marg[1]) + np.dot(link_wei[1], marg[2]) - 2 * ent_bias)
            elif i == 1:
                prob_ratio = np.exp(np.dot(marg[0], link_wei[0]) - ent_bias)
            elif i == 2:
                prob_ratio = np.exp(np.dot(marg[0], link_wei[1]) - ent_bias)
            #print(i, prob_ratio)
            # Update the vector
            new = update(v, sf, C, new_value_fn=new_value_fn, prob_ratio=prob_ratio)
            diff = np.abs(new - v).max()
            
            if verbose:
                print('diff inc', (new - v).max())
                print('diff dec', (new - v).min())
                print('diff 000', new.min() - v.min())
                print('min', new.min())
                print('max', new.max())
                print('sum', new.sum())
                print('prob', sf(new))
            
            vecs[i] = new
            max_diff = max(diff, max_diff)
        if max_diff < delta:
            break
    else:
        print('max iter reached with delta {}'.format(diff))
         
    return vecs
