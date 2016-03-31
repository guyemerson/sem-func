import os, pickle
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool  # @UnresolvedImport

from pydmrs.components import RealPred

DATA = '/anfs/bigdisc/gete2/wikiwoods/core'
PROC = 3

lemmatizer = WordNetLemmatizer()

###
# For Python <3.3:
from contextlib import contextmanager
@contextmanager
def terminating(thing):
    try:
        yield thing
    finally:
        thing.terminate()
_Pool = Pool
def Pool(*args, **kwargs):
    return terminating(_Pool(*args, **kwargs))
###

def lemmatize_pred(pred, pos):
    old = pred.lemma.rsplit('/', 1)[0]
    new = lemmatizer.lemmatize(old, pos)
    return RealPred(new, pos, 'unknown')
    
def lemmatize_file(fname):
    "Lemmatize the unknown words in a file"
    print(fname)
    with open(os.path.join(DATA, fname), 'rb') as f:
        trips = pickle.load(f)
    for t in trips:
        v, a, p = t
        if v.pos == 'u':
            t[0] = lemmatize_pred(v, 'v')
        if a and a.pos == 'u':
            t[1] = lemmatize_pred(a, 'n')
        if p and p.pos == 'u':
            t[2] = lemmatize_pred(p, 'n')
    with open(os.path.join(DATA, fname), 'wb') as f:
        pickle.dump(trips, f)

all_names = sorted(os.listdir(DATA))

with Pool(PROC) as p:
    p.map(lemmatize_file, all_names)