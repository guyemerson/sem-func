import sys, os, pickle, numpy
from multiprocessing import Pool, Manager
from time import sleep
from copy import copy

from model import DirectTrainingSetup, DirectTrainer, \
    SemFuncModel_IndependentPreds, SemFuncModel_FactorisedPreds

numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)

THRESH = 10000

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-{}-nodes'.format(THRESH)
VOCAB = '/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl'
FREQ = '/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl'

OUTPUT = '/anfs/bigdisc/gete2/wikiwoods/sem-func/core-{}-4'.format(THRESH)
# Save under OUTPUT.pkl and OUTPUT.aux.pkl

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

manager = Manager()

aux_info = {'particle': create_particle(3,2,5),
            'neg_samples': 5,
            'epochs': 3,
            'minibatch': 20,
            'processes': 50,
            'completed_files': manager.list()}
globals().update(aux_info)

# Set up training (without data)
trainer = DirectTrainer(setup, (),
                        particle,  # @UndefinedVariable
                        neg_samples = neg_samples)  # @UndefinedVariable

print("Set up complete, beginning training...")
sys.stdout.flush()

def train_on_file(fname):
    """
    Train on a single file
    (without saving to disk)
    """
    print(fname)
    with open(os.path.join(DATA, fname), 'rb') as f:
        trainer.load_file(f)
    trainer.train(epochs = epochs,  # @UndefinedVariable
                  minibatch = minibatch)  # @UndefinedVariable
    completed_files.append(fname)  # @UndefinedVariable

def save():
    """
    Save trained model to disk
    """
    with open(OUTPUT+'.pkl', 'wb') as f:
        crucial = copy(setup)
        crucial.pred_tokens = FREQ
        crucial.freq = FREQ
        crucial.pred_name = VOCAB
        pickle.dump(crucial, f)
    with open(OUTPUT+'.aux.pkl', 'wb') as f:
        actual_info = copy(aux_info)
        actual_info['completed_files'] = aux_info['completed_files']._getvalue()
        pickle.dump(actual_info, f)

# Give different files to different processes
with Pool(processes) as p:  # @UndefinedVariable
    res = p.map_async(train_on_file, sorted(os.listdir(DATA)))
    # While waiting for training to finish, save regularly
    while not res.ready():
        save()
        sleep(60)
save()

"""
import cProfile
cProfile.runctx('...', globals(), locals())
"""