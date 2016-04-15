import sys, os, pickle, numpy, argparse
from multiprocessing import Pool, Manager, Process, Queue
from time import sleep
from copy import copy

from model import SemFuncModel_IndependentPreds, SemFuncModel_FactorisedPreds
from trainingsetup import DirectTrainingSetup, AdaGradTrainingSetup
from trainer import DirectTrainer, create_particle
from utils import sub_namespace

numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)

parser = argparse.ArgumentParser(description="Train a sem-func model")
# Output and input
parser.add_argument('suffix', nargs='?', default=None)
parser.add_argument('-thresh', type=int, default=10000)
# Model hyperparameters
parser.add_argument('-model', type=str, default='independent')
parser.add_argument('-dims', type=int, default=40)
parser.add_argument('-embed_dims', type=int, default=20)
parser.add_argument('-card', type=int, default=5)
parser.add_argument('-init_bias', type=float, default=5)
parser.add_argument('-init_card', type=float, default=8)
parser.add_argument('-init_range', type=float, default=1)
# Training setup parameters
parser.add_argument('-setup', type=str, default='adagrad')
parser.add_argument('-rate', type=float, default=0.01)
parser.add_argument('-rate_ratio', type=float, default=1)
parser.add_argument('-l2', type=float, default=0.001)
parser.add_argument('-l2_ratio', type=float, default=1)
parser.add_argument('-l1', type=float, default=0.001)
parser.add_argument('-l1_ratio', type=float, default=1)
parser.add_argument('-ent_steps', type=int, default=10)
parser.add_argument('-pred_steps', type=int, default=1)
parser.add_argument('-ada_decay', type=float, default=1-10**-8)
# Negative sample parameters
parser.add_argument('-neg_samples', type=int, default=5)
parser.add_argument('-particle', type=int, nargs=3, default=(3,2,5))
# Training parameters
parser.add_argument('-epochs', type=int, default=3)
parser.add_argument('-minibatch', type=int, default=20)
parser.add_argument('-processes', type=int, default=50)
parser.add_argument('-ent_burnin', type=int, default=10)
parser.add_argument('-pred_burnin', type=int, default=2)

args = parser.parse_args()

# Set input and output
DATA = '/anfs/bigdisc/gete2/wikiwoods/core-{}-nodes'.format(args.thresh)

output_template = '/anfs/bigdisc/gete2/wikiwoods/sem-func/core-{}-{}'

if args.suffix is None:
    args.suffix = 1
    while os.path.exists(output_template.format(args.thresh, args.suffix)+'.pkl'):
        args.suffix += 1

OUTPUT = output_template.format(args.thresh, args.suffix)
# Save under OUTPUT.pkl and OUTPUT.aux.pkl

VOCAB = '/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl'
FREQ = '/anfs/bigdisc/gete2/wikiwoods/core-5-freq.pkl'

# Check the output path is clear
if os.path.exists(OUTPUT+'.pkl'):
    raise Exception('File already exists')

# Load vocab for model
with open(VOCAB, 'rb') as f:
    preds = pickle.load(f)
with open(FREQ, 'rb') as f:
    pred_freq = pickle.load(f)
links = ['ARG1', 'ARG2']

# Ignore rare preds (e.g. if using core-100)
for i in range(len(pred_freq)):
    if pred_freq[i] < args.thresh:
        pred_freq[i] = 0

# Set up model
model_kwargs = sub_namespace(args, ["dims",
                                    "card",
                                    "init_bias",
                                    "init_card",
                                    "init_range"])
if args.model == 'independent':
    model_class = SemFuncModel_IndependentPreds
elif args.model == 'factorised':
    model_class = SemFuncModel_FactorisedPreds
    model_kwargs.update(sub_namespace(args, ["embed_dims"]))
else:
    raise Exception('model class not recognised')
model = model_class(preds, links, pred_freq, **model_kwargs)

# Set up training hyperparameters
setup_kwargs = sub_namespace(args, ["rate",
                                    "rate_ratio",
                                    "l2",
                                    "l2_ratio",
                                    "l1",
                                    "l1_ratio",
                                    "ent_steps",
                                    "pred_steps"])
if args.setup == 'direct':
    setup_class = DirectTrainingSetup
elif args.setup == 'adagrad':
    setup_class = AdaGradTrainingSetup
    setup_kwargs.update(sub_namespace(args, ["ada_decay"]))
else:
    raise Exception('setup class not recognised')
setup = setup_class(model, **setup_kwargs)

# Set up training (without data)
particle = create_particle(3,2,5)
trainer = DirectTrainer(setup, (),
                        particle,
                        neg_samples = args.neg_samples)

# Aux info (outside of setup)
manager = Manager()
aux_info = sub_namespace(args, ["neg_samples",
                                "epochs",
                                "minibatch",
                                "processes",
                                "ent_burnin",
                                "pred_burnin"])
aux_info['particle'] = particle
completed_files = manager.list()

print("Set up complete, beginning training...")
sys.stdout.flush()

# Functions for multiprocessing
def train_on_file(fname):
    """
    Train on a single file
    (without saving to disk)
    """
    global trainer
    print(fname)
    with open(os.path.join(DATA, fname), 'rb') as f:
        trainer.load_file(f,
                          ent_burnin = args.ent_burnin,
                          pred_burnin = args.pred_burnin)
    trainer.train(epochs = args.epochs,
                  minibatch = args.minibatch)
    completed_files.append(fname)

def save():
    """
    Save trained model to disk
    """
    with open(OUTPUT+'.pkl', 'wb') as f:
        # Remove things that weren't trained
        crucial = copy(setup)
        crucial.model = copy(crucial.model)
        crucial.model.pred_tokens = FREQ
        crucial.model.freq = FREQ
        crucial.model.pred_name = VOCAB
        # Remove queues (which can't be pickled)
        crucial.link_update_queues = None
        crucial.pred_update_queues = None
        if args.setup == 'adagrad':
            crucial.link_sqsum_queues = None
            crucial.pred_sqsum_queues = None
        # Save the file!
        pickle.dump(crucial, f)
    with open(OUTPUT+'.aux.pkl', 'wb') as f:
        actual_info = copy(aux_info)
        actual_info['completed_files'] = completed_files._getvalue()
        pickle.dump(actual_info, f)

# Workers to update the shared weights from queued gradients

assert args.setup == 'adagrad'

for i, q in enumerate(setup.link_update_queues):
    weight = setup.link_weights[i]
    sqsum = setup.link_sqsum[i]
    def update():
        global weight, sqsum
        while True:
            sqgrad, step = q.get()
            sqsum += sqgrad
            weight += step.clip(-weight)
    worker = Process(target=update)
    worker.start()

for i, q in enumerate(setup.pred_update_queues):
    weight = setup.pred_weights[i]
    sqsum = setup.pred_sqsum[i]
    if i < len(setup.pred_local_weights):
        def update():
            global weight, sqsum
            while True:
                sqgrad, step = q.get()
                assert step.next == step.indices.shape[0]
                sqsum[step.indices] += sqgrad
                weight[step.indices] += step.array.clip(-weight[step.indices])
    else:
        raise NotImplementedError
    worker = Process(target=update)
    worker.start()

# Workers to process training data

file_queue = Queue()
for fname in sorted(os.listdir(DATA)):
    file_queue.put(fname)
for _ in range(args.processes):
    file_queue.put(None)

def train():
    while True:
        fname = file_queue.get()
        if fname is not None:
            train_on_file(fname)
        else:
            break
    #print('worker done')

training_workers = []
for _ in range(args.processes):
    worker = Process(target=train)
    worker.start()
    training_workers.append(worker)

while (not file_queue.empty()) or any(w.is_alive() for w in training_workers):
    save()
    sleep(60)
while not (all(q.empty() for q in setup.link_update_queues) and \
           all(q.empty() for q in setup.pred_update_queues)):
    print('Waiting for updates to finish...')
    save()
    sleep(60)
print('Training complete')
save()

"""
import cProfile
cProfile.runctx('train_on_file(os.listdir(DATA)[0])', globals(), locals())
"""