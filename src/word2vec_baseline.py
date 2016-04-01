import os, pickle, gensim
from collections import Counter

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-5'
OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/word2vec/model'

class WikiWoodsIterator():
    "Iterate through all triples in a directory"
    def __init__(self, directory):
        self.dir = directory
    
    def __iter__(self):
        for fname in sorted(os.listdir(self.dir)):
            with open(os.path.join(self.dir, fname), 'rb') as f:
                trips = pickle.load(f)
            for t in trips:
                yield [str(x) for x in t if x is not None]

wikiter = WikiWoodsIterator(DATA)

model = gensim.models.Word2Vec(wikiter, workers=3)

model.save(OUTPUT)

count = Counter({pred:vec.count for pred,vec in model.vocab.items()})
preds = [x for x,n in count.most_common(20)]
pairs = [(x,y) for x in preds for y in preds if y != x]
for x,y in pairs:
    print(x,y, model.similarity(x,y))
print()
for x in preds:
    print(x, model.most_similar(positive=[x]))