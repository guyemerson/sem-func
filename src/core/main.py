import sys, os, pickle, numpy, argparse
import numpy as np

from model import SemFuncModel_IndependentPreds, SemFuncModel_FactorisedPreds
from trainingsetup import AdaGradTrainingSetup, AdamTrainingSetup
from trainer import DataInterface, create_particle, Trainer
from utils import sub_dict
from __config__.filepath import DATA_DIR, AUX_DIR, OUT_DIR, VOCAB_FILE, FREQ_FILE

def setup_trainer(**kw):
    """
    Setup a semantic function model, ready for training
    """
    # Set input and output
    DATA = os.path.join(DATA_DIR, 'core-{}-nodes'.format(kw['thresh']))
    
    output_template = os.path.join(OUT_DIR, 'core-{}-{}')
    
    if kw['suffix'] is None:
        kw['suffix'] = 1
        while os.path.exists(output_template.format(kw['thresh'], kw['suffix'])+'.pkl'):
            kw['suffix'] += 1
    
    OUTPUT = output_template.format(kw['thresh'], kw['suffix'])
    # Save under OUTPUT.pkl and OUTPUT.aux.pkl
    
    # Check the output path is clear
    if os.path.exists(OUTPUT+'.pkl'):
        raise Exception('File already exists')
    
    # Load vocab for model
    with open(os.path.join(AUX_DIR, VOCAB_FILE), 'rb') as f:
        preds = pickle.load(f)
    with open(os.path.join(AUX_DIR, FREQ_FILE), 'rb') as f:
        pred_freq = pickle.load(f)
    links = ['ARG1', 'ARG2']
    
    # Ignore rare preds (e.g. if using core-100)
    for i in range(len(pred_freq)):
        if pred_freq[i] < kw['thresh']:
            pred_freq[i] = 0
    
    # Set random seed, if specified
    if kw['seed']:
        np.random.seed(kw['seed'])
    
    # Set up model
    model_kwargs = sub_dict(kw, ["dims",
                                 "card",
                                 "init_bias",
                                 "init_card",
                                 "init_range",
                                 "init_ent_bias",
                                 "init_link_str",
                                 "init_verb_prop",
                                 "init_pat_prop",
                                 "init_ag_prop"])
    if kw['model'] == 'independent':
        model_class = SemFuncModel_IndependentPreds
    elif kw['model'] == 'factorised':
        model_class = SemFuncModel_FactorisedPreds
        model_kwargs.update(sub_dict(kw, ["embed_dims"]))
    else:
        raise Exception('model class not recognised')
    model = model_class(preds, links, pred_freq, verbose=False, **model_kwargs)
    
    # Set up training hyperparameters
    setup_kwargs = sub_dict(kw, ["rate",
                                 "rate_ratio",
                                 "l2",
                                 "l2_ratio",
                                 "l2_ent",
                                 "l1",
                                 "l1_ratio",
                                 "l1_ent",
                                 "ent_steps",
                                 "pred_steps"])
    if kw['setup'] == 'adagrad':
        setup_class = AdaGradTrainingSetup
        setup_kwargs.update(sub_dict(kw, ["ada_decay"]))
    elif kw['setup'] == 'adam':
        setup_class = AdamTrainingSetup
        setup_kwargs.update(sub_dict(kw, ["mean_decay",
                                          "var_decay"]))
    else:
        raise Exception('setup class not recognised')
    setup = setup_class(model, **setup_kwargs)
    
    # Set up training (without data)
    particle = create_particle(3,2,5)
    interface = DataInterface(setup, (),
                              particle,
                              neg_samples = kw['neg_samples'])
    
    trainer_kwargs = sub_dict(kw, ["processes",
                                   "epochs",
                                   "minibatch",
                                   "ent_burnin",
                                   "pred_burnin"])
    
    trainer = Trainer(interface,
                      data_dir = DATA,
                      output_name = OUTPUT,
                      **trainer_kwargs)
    
    return trainer

if __name__ == "__main__":
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
    parser.add_argument('-init_ent_bias', type=float, default=5)
    parser.add_argument('-init_link_str', type=float, default=3)
    parser.add_argument('-init_verb_prop', type=float, default=0.5)
    parser.add_argument('-init_pat_prop', type=float, default=0.6)
    parser.add_argument('-init_ag_prop', type=float, default=0.6)
    # Training setup parameters
    parser.add_argument('-setup', type=str, default='adagrad')
    parser.add_argument('-rate', type=float, default=0.01)
    parser.add_argument('-rate_ratio', type=float, default=1)
    parser.add_argument('-l2', type=float, default=0.001)
    parser.add_argument('-l2_ratio', type=float, default=1)
    parser.add_argument('-l2_ent', type=float, default=0.001)
    parser.add_argument('-l1', type=float, default=0.001)
    parser.add_argument('-l1_ratio', type=float, default=1)
    parser.add_argument('-l1_ent', type=float, default=0.001)
    parser.add_argument('-ent_steps', type=int, default=10)
    parser.add_argument('-pred_steps', type=int, default=1)
    parser.add_argument('-ada_decay', type=float, default=1-10**-8)
    parser.add_argument('-mean_decay', type=float, default=0.9)
    parser.add_argument('-var_decay', type=float, default=0.999)
    # Negative sample parameters
    parser.add_argument('-neg_samples', type=int, default=5)
    parser.add_argument('-particle', type=int, nargs=3, default=(3,2,5))
    # Training parameters
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-minibatch', type=int, default=20)
    parser.add_argument('-processes', type=int, default=50)
    parser.add_argument('-ent_burnin', type=int, default=10)
    parser.add_argument('-pred_burnin', type=int, default=2)
    parser.add_argument('-seed', type=int, default=0)
    
    args = parser.parse_args()
    arg_dict = dict(args._get_kwargs())
    
    trainer = setup_trainer(**arg_dict)
    
    print("Set up complete, beginning training...")
    sys.stdout.flush()
    
    trainer.start()
    