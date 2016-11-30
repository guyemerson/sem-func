import os, pickle, numpy as np

from collections import Counter
from math import log
from warnings import warn

from __config__.filepath import AUX_DIR, FREQ_FILE, VOCAB_FILE 

def generate_random_data(n_trans, n_subj, n_obj):
    """
    Generate a random set of tuples
    :param n_trans: number of transitive tuples
    :param n_subj: number of subject-verb tuples
    :param n_obj: number of verb-object tuples
    :return: list of (verb, subj, obj) tuples, with None for missing arguments 
    """
    
    # Load vocabulary
    
    with open(os.path.join(AUX_DIR, VOCAB_FILE), 'rb') as f:
        pred_name = pickle.load(f)
    with open(os.path.join(AUX_DIR, FREQ_FILE), 'rb') as f:
        pred_freq = pickle.load(f)
    
    verbs = [i for i,x in enumerate(pred_name) if x.rsplit('_', 2)[1] == 'v']
    nouns = [i for i,x in enumerate(pred_name) if x.rsplit('_', 2)[1] == 'n']
    
    # Get noun and verb tokens to sample from 
    
    verb_tokens = np.zeros(pred_freq[verbs].sum(), dtype='int64')
    i = 0
    for p in verbs:
        f = pred_freq[p]
        verb_tokens[i : i+f] = p
        i += f
    
    noun_tokens = np.zeros(pred_freq[nouns].sum(), dtype='int64')
    i = 0
    for p in nouns:
        f = pred_freq[p]
        noun_tokens[i : i+f] = p
        i += f
    
    # Sample the tuples
    
    n_total = n_trans + n_subj + n_obj
    subj = np.random.choice(noun_tokens, n_total)
    verb = np.random.choice(verb_tokens, n_total)
    obj  = np.random.choice(noun_tokens, n_total)
    
    data =  [(int(verb[i]), int(subj[i]), int(obj[i])) for i in range(n_trans)]
    data += [(int(verb[i]), int(subj[i]), None)        for i in range(n_trans, n_trans+n_subj)]
    data += [(int(verb[i]), None,         int(obj[i])) for i in range(n_trans+n_subj, n_trans+n_subj+n_obj)]
    
    return data

def count_separate(data):
    """
    Convert a list of SVO triples with missing arguments,
    to three counts of tuples
    :param data: list of triples
    :return: {SVO}, {SV}, {VO}
    """
    svo, sv, vo = Counter(), Counter(), Counter()
    for verb, subj, obj in data:
        if subj is None:
            vo[verb, obj] += 1
        elif obj is None:
            sv[subj, verb] += 1
        else:
            svo[subj, verb, obj] += 1
    return svo, sv, vo

def compare_KL(model, real_data, fake_data, samples=None, **kwargs):
    """
    Approximately calculate the Kullback-Leibler divergence from the model to two sets of data
    :param model: the sem-func model
    :param real_data: first set of tuples
    :param fake_data: second set of tuples
    :param samples: number of samples to draw, for: SVO, SV, VO graphs
    :return: (real KLs, fake KLs), each for (SVO, SV, VO) subsets
    """
    real_count = count_separate(real_data)
    fake_count = count_separate(fake_data)
    
    n_real = tuple(sum(c.values()) for c in real_count)
    n_fake = tuple(sum(c.values()) for c in fake_count)
    if n_real != n_fake:
        warn('real_data and fake_data should contain the same number of each configuration')
    
    if samples is None:
        samples = n_real
    
    # Initialise counts for generated samples
    real_match = [{tup: 0 for tup in c} for c in real_count]
    fake_match = [{tup: 0 for tup in c} for c in fake_count]
    
    # Sample from the model
    sampler = [model.sample_background_svo, model.sample_background_sv, model.sample_background_vo]
    
    for i in range(3):
        # Sample entities for each graph configuration
        for ents in sampler[i](samples=samples[i], **kwargs):
            # For the sampled entities, find the distribution over predicates
            pred_dist = [model.pred_dist(e) for e in ents]
            # Add the probability that this sample would generate the observed predicates
            for preds in real_match[i]:
                real_match[i][preds] += np.prod([pred_dist[i][p] for i,p in enumerate(preds)])
            for preds in fake_match[i]:
                fake_match[i][preds] += np.prod([pred_dist[i][p] for i,p in enumerate(preds)])
        # Average the probabilities
        for preds in real_match[i]:
            real_match[i][preds] /= samples[i]
        for preds in fake_match[i]:
            fake_match[i][preds] /= samples[i]
    
    # Calculate Kullback-Leibler divergence
    # x are generated tuples, P is real/fake, Q is generated 
    # sum_x P(x) ( logP(x) - logQ(x) )
    
    real_KL = [0,0,0]
    fake_KL = [0,0,0]
    
    for i in range(3):
        for preds, count in real_count[i].items():
            prob = count / n_real[i]
            real_KL[i] += prob * (log(prob) - log(real_match[i][preds]))
        for preds, count in fake_count[i].items():
            prob = count / n_fake[i]
            fake_KL[i] += prob * (log(prob) - log(fake_match[i][preds]))
        
    return real_KL, fake_KL
