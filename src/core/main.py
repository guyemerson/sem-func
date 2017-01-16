import sys, os, pickle, numpy, argparse, logging
import numpy as np

from model import SemFuncModel_IndependentPreds, SemFuncModel_MultiIndependentPreds
from trainingsetup import AdaGradTrainingSetup, AdamTrainingSetup
from trainer import DataInterface, create_particle, Trainer
from utils import sub_dict
from __config__.filepath import DATA_DIR, AUX_DIR, OUT_DIR

def setup_trainer(**kw):
    """
    Setup a semantic function model, ready for training
    """
    # Set input and output filepaths
    # Naming convention is <dataset>-<threshold>-<name>
    
    if kw['multipred']:
        prefix = 'multicore'
    else:
        prefix = 'core'
    thresh = kw['thresh']
    suffix = kw['suffix']
    
    # Directory for the data
    DATA = os.path.join(DATA_DIR, '{}-{}-nodes'.format(prefix, thresh))
    
    output_template = os.path.join(OUT_DIR, '{}-{}-{}')
    
    # If no suffix is given, use the smallest integer for which no file exists
    if suffix is None:
        suffix = 1
        while os.path.exists(output_template.format(prefix, thresh, suffix)+'.pkl'):
            suffix += 1
    
    # Save under OUTPUT.pkl and OUTPUT.aux.pkl
    OUTPUT = output_template.format(prefix, thresh, suffix)
    
    # Check the output path is clear, unless overwriting is allowed
    if not kw['overwrite'] and os.path.exists(OUTPUT+'.pkl'):
        raise Exception("File already exists - did you mean to use '-overwrite'?")
    
    # Load vocab for model
    with open(os.path.join(AUX_DIR, '{}-{}-vocab.pkl'.format(prefix, thresh)), 'rb') as f:
        preds = pickle.load(f)
    with open(os.path.join(AUX_DIR, '{}-{}-freq.pkl'.format(prefix, thresh)), 'rb') as f:
        pred_freq = pickle.load(f)
    links = ['ARG1', 'ARG2']
    
    # Set random seed, if specified
    if kw['seed']:
        np.random.seed(kw['seed'])
    
    # Set up model
    
    # Get hyperparameters
    model_kwargs = sub_dict(kw, ["dims",
                                 "card",
                                 "init_bias",
                                 "init_card",
                                 "init_range",
                                 "init_ent_bias",
                                 "init_link_str",
                                 "init_verb_prop",
                                 "init_pat_prop",
                                 "init_ag_prop",
                                 "freq_alpha"])
    # Choose model class 
    if kw['model'] == 'independent':
        if kw['multipred']:
            model_class = SemFuncModel_MultiIndependentPreds
        else:
            model_class = SemFuncModel_IndependentPreds
    elif kw['model'] == 'factorised':
        raise ValueError('factorised pred model is deprecated')
        #model_class = SemFuncModel_FactorisedPreds
        #model_kwargs.update(sub_dict(kw, ["embed_dims"]))
    else:
        raise ValueError('model class not recognised')
    # Initialise model
    model = model_class(preds, links, pred_freq, verbose=False, **model_kwargs)
    
    # Set up gradient descent algorithm
    
    # Get hyperparameters
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
    # Choose training setup class
    if kw['setup'] == 'adagrad':
        setup_class = AdaGradTrainingSetup
        setup_kwargs.update(sub_dict(kw, ["ada_decay"]))
    elif kw['setup'] == 'adam':
        setup_class = AdamTrainingSetup
        setup_kwargs.update(sub_dict(kw, ["mean_decay",
                                          "var_decay"]))
    else:
        raise ValueError('setup class not recognised')
    # Initialise training setup
    setup = setup_class(model, **setup_kwargs)
    
    # Set up trainer (without data)
    
    # Initialise particle
    particle = create_particle(*kw['particle'])
    # Initialise data interface
    interface = DataInterface(setup,
                              particle,
                              neg_samples = kw['neg_samples'])
    # Get hyperparameters
    trainer_kwargs = sub_dict(kw, ["processes",
                                   "epochs",
                                   "minibatch",
                                   "ent_burnin",
                                   "pred_burnin"])
    # Initialise trainer
    trainer = Trainer(interface,
                      data_dir = DATA,
                      output_name = OUTPUT,
                      **trainer_kwargs)
    
    return trainer

if __name__ == "__main__":
    numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Train a sem-func model")
    # Output and input
    parser.add_argument('suffix', nargs='?', default=None)
    parser.add_argument('-thresh', type=int, default=5)
    parser.add_argument('-overwrite', action='store_true')
    # Data format
    parser.add_argument('-multipred', action='store_true')
    # Model hyperparameters
    parser.add_argument('-model', type=str, default='independent')
    parser.add_argument('-dims', type=int, default=40)
    parser.add_argument('-embed_dims', type=int, default=20)
    parser.add_argument('-card', type=int, default=5)
    parser.add_argument('-freq_alpha', type=float, default=0.75)
    # Model initialisation
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
    parser.add_argument('-processes', type=int, default=12)
    parser.add_argument('-ent_burnin', type=int, default=10)
    parser.add_argument('-pred_burnin', type=int, default=2)
    # Practical stuff
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-timeout', type=int, default=0)
    parser.add_argument('-validation', nargs='+', default=[])
    parser.add_argument('-maxtasksperchild', type=int, default=None)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-logging', default='WARNING')
    
    # Get command line arguments
    args = parser.parse_args()
    arg_dict = dict(args._get_kwargs())
    
    # Allow remote debugging
    if args.debug:  
        from pdb_clone import pdbhandler
        pdbhandler.register()
        # To debug, run: `pdb-attach --kill --pid PID`
        # child pid can be found with: `ps x --forest`
    
    # Initialise trainer object
    trainer = setup_trainer(**arg_dict)
    
    print("Set up complete, beginning training...")
    sys.stdout.flush()
    
    # Start training
    trainer.start(timeout=args.timeout,
                  validation=[x+'.pkl' for x in args.validation],
                  debug=args.debug,
                  logging_level=logging.getLevelName(args.logging),
                  maxtasksperchild=args.maxtasksperchild)
