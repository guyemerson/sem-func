import os, pickle, numpy as np

from testing import get_test_GS2011, get_GS2011_indexed, convert_name, load_freq_lookup_dicts
from relpron_variational import load_weights_and_vectors
from variational import marginal_approx, get_semfunc
from utils import cosine, product
from __config__.filepath import AUX_DIR

# Get test function and data
test_fn = get_test_GS2011()
raw_triples, _ = get_GS2011_indexed()
lookup = load_freq_lookup_dicts()
verbs = [lookup['v'][verb] for verb, _, _ in raw_triples]

def get_scores(subdir='meanfield_gs2011', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    try:
        with open(score_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_scores(scores, subdir='meanfield_gs2011', filename='scores'):
    "Get previously calculated scores"
    score_file = os.path.join(AUX_DIR, subdir, filename) + '.pkl'
    with open(score_file, 'wb') as f:
        pickle.dump(scores, f)

def load_scoring_fn(filename, option='implies', **kwargs):
    "Load a scoring function"
    # Load files
    pred_wei, pred_bias, C, meanfield_vecs = load_weights_and_vectors(filename, meanfield_dir='meanfield_gs2011', **kwargs)
    # Set up semantic functions
    semfuncs = [get_semfunc(pred_wei[i], pred_bias[i]) for i in range(len(pred_wei))]
    # Get marginal distributions
    marg = [marginal_approx(prob, C) for prob, _, _ in meanfield_vecs]
    
    if option == 'implies':
        def scoring_fn(i, j):
            """
            Calculate how much each verbal predicate is true of the other verb's entity
            :param i: triple index
            :param j: triple index
            :return: product of probabilities
            """
            # Get verb semfuncs and verb entities
            funcs = (semfuncs[verbs[i]], semfuncs[verbs[j]])
            ents = (marg[i], marg[j])
            # Multiply probs
            return funcs[0](ents[1]) * funcs[1](ents[0])
    
    elif option == 'similarity':
        def scoring_fn(i, j):
            """
            Calculate how similar the entities are
            :param i: triple index
            :param j: triple index
            :return: cosine similarity
            """
            # Get verb entities
            ents = (marg[i], marg[j])
            # Get cosine similarity
            return cosine(*ents)
    
    else:
        raise ValueError('option not recognised')
    
    return scoring_fn

def load_baseline_scoring_fn(filename, subdir='word2vec', with_lookup=True):
    "Load a vector addition scoring model"
    from gensim.models import Word2Vec
    model = Word2Vec.load(os.path.join(AUX_DIR, subdir, filename))
    with open(os.path.join(AUX_DIR, 'multicore-5-vocab.pkl'), 'rb') as f:
        pred_name = pickle.load(f)
    vec = model.syn0
    def scoring_fn(i, j):
        "Add triples and take cosine"
        # Get tokens
        v1 = raw_triples[i][0]
        v2, s, o = raw_triples[j]
        if with_lookup:
            v1 = pred_name[lookup['v'][v1]]
            v2 = pred_name[lookup['v'][v2]]
            s = pred_name[lookup['n'][s]]
            o = pred_name[lookup['n'][o]]
        # Lookup vectors
        v1, v2, s, o = [vec[model.vocab[x].index] for x in [v1,v2,s,o]]
        comp1 = v1 + s + o
        comp2 = v2 + s + o
        return cosine(comp1, comp2)
    return scoring_fn

if __name__ == "__main__":
    vso_model = load_baseline_scoring_fn('model', with_lookup=True)
    print(test_fn(vso_model))
    
    plain_model = load_baseline_scoring_fn('model-plain', with_lookup=False)
    print(test_fn(plain_model))

    for option in ['implies', 'similarity']:
        print(option)
        score_name = 'scores_'+option
        scores = get_scores(filename=score_name)
        
        all_score_fns = []
        all_settings = []
        
        for filename in os.listdir(os.path.join(AUX_DIR, 'meanfield_gs2011')):
            # Skip score files
            if filename[-3:] != '.gz':
                continue
            name = filename.split('.')[0]
            tuple_name = convert_name(name)
            # Load score functions for ensemble
            # Skip models we don't want
            if tuple_name[2] != 400:
                continue
            score_fn = load_scoring_fn(name, option)
            all_score_fns.append(score_fn)
            all_settings.append(tuple_name)
            # Skip files we've already processed
            if tuple_name in scores:
                continue
            # Get rank correlation
            """
            result = test_fn(score_fn)
            scores[tuple_name] = result
            print(tuple_name)
            print(result)
            """
        
        def ensemble(i,j):
            "Ensemble score fn"
            return product(fn(i,j) for fn in all_score_fns)
        
        ensemble_score = test_fn(ensemble)
        scores[frozenset(all_settings)] = ensemble_score
        print('ensemble')
        print(ensemble_score)
        
        for x in np.arange(0.001, 0.0401, 0.001):
            def combined(i,j):
                return x * np.log(ensemble(i,j)) + (1-x) * np.log(vso_model(i,j))
            score = test_fn(combined)
            print(x, score)
            scores[frozenset(all_settings), 'vso_word2vec', x] = score
        
        save_scores(scores, filename=score_name)
