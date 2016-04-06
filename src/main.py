import sys, os, pickle, numpy
from multiprocessing import Pool

from model import DirectTrainingSetup, DirectTrainer, \
    SemFuncModel_IndependentPreds, SemFuncModel_FactorisedPreds

numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)

THRESH = 10000

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-{}-nodes'.format(THRESH)
VOCAB = '/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl'
FREQ = '/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl'

OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/sem-func/core-{}-4.pkl'.format(THRESH)

if os.path.exists(OUTPUT):
    raise Exception('File already exists')

# Load vocab for model
with open(VOCAB, 'rb') as f:
    preds = pickle.load(f)
with open(FREQ, 'rb') as f:
    pred_freq = pickle.load(f)
links = ['ARG1', 'ARG2']

# Ignore rare preds (if using core-100)
for i in range(len(pred_freq)):
    if pred_freq[i] < THRESH:
        pred_freq[i] = 0

# Set up model
model = SemFuncModel_FactorisedPreds(preds, links, pred_freq,
                                      dims = 50,
                                      card = 5,
                                      embed_dims = 20, 
                                      init_bias = 5,
                                      init_card = 3,
                                      init_range = 1)

# Set up training hyperparameters
setup = DirectTrainingSetup(model,
                            rate = 0.001,
                            rate_ratio = 1,
                            l2 = 1,
                            l2_ratio = 1,
                            l1 = 1,
                            l1_ratio = 1,
                            ent_steps = 3,
                            pred_steps = 2)


def create_particle(p_full, p_agent, p_patient):
    """
    Create a fantasy particle with a given number of
    transitive and intransitive situations
    """
    particle = []
    nid = 0
    for _ in range(p_full):
        particle.extend([(nid, [0,1], [nid+1,nid+2], (), ()),
                         (nid+1, (), (), [0], [nid]),
                         (nid+2, (), (), [1], [nid])])
        nid += 3
    for _ in range(p_agent):
        particle.extend([(nid, [0], [nid+1], (), ()),
                         (nid+1, (), (), [0], [nid])])
        nid += 2
    for _ in range(p_agent):
        particle.extend([(nid, [1], [nid+1], (), ()),
                         (nid+1, (), (), [1], [nid])])
        nid += 2
    return particle

# Set up training (without data)
trainer = DirectTrainer(setup, (),
                        create_particle(3,2,5),
                        neg_samples = 5)

print("Set up complete, beginning training...")
sys.stdout.flush()

def train_on_file(fname):
    print(fname)
    with open(os.path.join(DATA, fname), 'rb') as f:
        trainer.load_file(f)
    trainer.train(epochs = 3,
                  minibatch = 20,
                  print_every = 1)

#from time import time

# Train on each file
with Pool(20) as p:
    #t0 = time()
    p.map(train_on_file, sorted(os.listdir(DATA)))
#print(time()-t0)

"""
# Train on each file
t0 = time()
for filename in sorted(os.listdir(DATA))[:5]:
    print('\nLoading ', filename)
    with open(os.path.join(DATA, filename), 'rb') as f:
        trainer.load_file(f)
    print('Training')
    # Burn-in?
    trainer.train(epochs = 1,
                  minibatch = 20,
                  print_every = 2)#,dump_file = OUTPUT)
print(time()-t0)
"""

"""
import cProfile
cProfile.runctx('...', globals(), locals())
"""