import os, pickle
from gensim.models import Word2Vec

from testing import get_test_all
from __config__.filepath import AUX_DIR

test_fn = get_test_all()

with open(os.path.join(AUX_DIR, 'multicore-5-vocab.pkl'), 'rb') as f:
    pred_name = pickle.load(f)

model2 = Word2Vec.load(os.path.join(AUX_DIR, 'word2vec', 'model-plain'))

def sim2(a, b):
    x = pred_name[a].split('_')[1]
    y = pred_name[b].split('_')[1]
    return model2.similarity(x,y)

test_fn(sim2)


model = Word2Vec.load(os.path.join(AUX_DIR, 'word2vec', 'model'))

def sim(a, b):
    x = pred_name[a]
    y = pred_name[b]
    return model.similarity(x,y)

test_fn(sim)


