import os, pickle, gensim, gzip, re, string

DATA = '/usr/groups/corpora/wikiwoods-1212/uio/wikiwoods/1212/gml/'
OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/word2vec/model-plain'

regex = [re.compile(r'⌊\S+¦'),
         re.compile(r'¦\S+⌋'),
         re.compile(r'⌊\S'),
         re.compile(r'\S⌋')]
nopunc = str.maketrans('', '', string.punctuation)

class WikiWoodsIterator():
    "Iterate through all files in a directory"
    def __init__(self, directory):
        self.dir = directory
    
    def __iter__(self):
        for fname in sorted(os.listdir(self.dir)):
            with gzip.open(os.path.join(self.dir, fname), 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    text = line.decode().lower().split('|')[1]
                    if not text:
                        continue
                    if text[0] == '¦':
                        text = text[1:]
                    text.replace('⌊⌋⌋', ' ⌋')
                    text.replace('⌊⌊⌋', '⌊ ')
                    text.replace('⌊¦⌋', ' ¦ ')
                    for r in regex:
                        text = r.sub('', text)
                    text = text.translate(nopunc)
                    tokens = text.split()
                    if not tokens:
                        continue 
                    yield tokens

wikiter = WikiWoodsIterator(DATA)

model = gensim.models.Word2Vec(wikiter, workers=32)

model.save(OUTPUT)
