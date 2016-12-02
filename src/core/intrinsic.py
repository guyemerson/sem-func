import os, pickle, numpy as np

from collections import Counter
from math import log
from warnings import warn

from __config__.filepath import AUX_DIR, FREQ_FILE, VOCAB_FILE
from utils import product


# Load vocabulary

with open(os.path.join(AUX_DIR, VOCAB_FILE), 'rb') as f:
    pred_name = pickle.load(f)
with open(os.path.join(AUX_DIR, FREQ_FILE), 'rb') as f:
    pred_freq = pickle.load(f)

verbs = [i for i,x in enumerate(pred_name) if x.rsplit('_', 2)[1] == 'v']
nouns = [i for i,x in enumerate(pred_name) if x.rsplit('_', 2)[1] == 'n']

noun_mask = np.array([x.rsplit('_', 2)[1] == 'n' for x in pred_name])
verb_mask = np.array([x.rsplit('_', 2)[1] == 'v' for x in pred_name])


def generate_random_data(n_trans, n_subj, n_obj):
    """
    Generate a random set of tuples
    :param n_trans: number of transitive tuples
    :param n_subj: number of subject-verb tuples
    :param n_obj: number of verb-object tuples
    :return: list of (verb, subj, obj) tuples, with None for missing arguments 
    """
    
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


def separate_prob(data):
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
    totals = [sum(c.values()) for c in (svo, sv, vo)]
    probs = [{tup: num/totals[i] for tup, num in c.items()} for i,c in enumerate((svo, sv, vo))]
    return probs


def KL(P, Q):
    """
    Calculate Kullback-Leibler divergence from Q to P
    Both P and Q should be dicts from elements to probabilities
    :param P: true distribution
    :param Q: approximating distribution
    :return: divergence
    """
    # sum_x P(x) ( logP(x) - logQ(x) )
    res = 0
    for item, prob in P.items():
        res += prob * (log(prob) - log(Q[item]))
    return res


def compare_KL(model, real_data, fake_data, samples=(100,100,100), **kwargs):
    """
    Approximately calculate the Kullback-Leibler divergence from the model to two sets of data
    :param model: the sem-func model
    :param real_data: first set of tuples
    :param fake_data: second set of tuples
    :param samples: number of samples to draw, for: SVO, SV, VO graphs
    :return: (real KLs, fake KLs), each for (SVO, SV, VO) subsets
    """
    # Get sample probabilities from the data
    real_prob = separate_prob(real_data)
    fake_prob = separate_prob(fake_data)
    
    # Initialise counts for generated samples
    real_match = [{tup: 0 for tup in c} for c in real_prob]
    fake_match = [{tup: 0 for tup in c} for c in fake_prob]
    
    # Sample from the model
    sampler = [model.sample_background_svo, model.sample_background_sv, model.sample_background_vo]
    
    for i in range(3):
        # Sample entities for each graph configuration
        for ents in sampler[i](samples=samples[i], **kwargs):
            # For the sampled entities, find the distribution over predicates
            pred_dist = [model.pred_dist(e) for e in ents]
            # Add the probability that this sample would generate the observed predicates
            for preds in real_match[i]:
                real_match[i][preds] += product(pred_dist[j][p] for j,p in enumerate(preds))
            for preds in fake_match[i]:
                fake_match[i][preds] += product(pred_dist[j][p] for j,p in enumerate(preds))
        # Average the probabilities
        for preds in real_match[i]:
            real_match[i][preds] /= samples[i]
        for preds in fake_match[i]:
            fake_match[i][preds] /= samples[i]
    
    real_KL = [KL(real_prob[i], real_match[i]) for i in range(3)]
    fake_KL = [KL(fake_prob[i], fake_match[i]) for i in range(3)]

    return real_KL, fake_KL


def baseline_KL(real_data, fake_data):
    """
    Calculate the Kullback-Leibler divergence from the null hypothesis (sample nouns and verbs according to frequency) to two sets of data
    :param real_data: first set of tuples
    :param fake_data: second set of tuples
    :return: (real KLs, fake KLs), each for (SVO, SV, VO) subsets
    """
    real_prob = separate_prob(real_data)
    fake_prob = separate_prob(fake_data)
    
    noun_prob = pred_freq * noun_mask / pred_freq[nouns].sum()
    verb_prob = pred_freq * verb_mask / pred_freq[verbs].sum()
    both_prob = noun_prob + verb_prob
    
    real_match = [{tup: product(both_prob[p] for p in tup)
                   for tup in c}
                  for c in real_prob]
    fake_match = [{tup: product(both_prob[p] for p in tup)
                   for tup in c}
                  for c in fake_prob]
    
    real_KL = [KL(real_prob[i], real_match[i]) for i in range(3)]
    fake_KL = [KL(fake_prob[i], fake_match[i]) for i in range(3)]

    return real_KL, fake_KL
    