import argparse, numpy as np, logging

from trainer import Trainer

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, threshold=np.nan)
    
    parser = argparse.ArgumentParser(description="Train a sem-func model")
    # File name
    parser.add_argument('fullname', nargs='?', default=None)
    parser.add_argument('-p', '--prefix', default='multicore')
    parser.add_argument('-t', '--threshold', type=int, default=5)
    parser.add_argument('-s', '--suffix', default=None)
    parser.add_argument('-o', '--output', default=None)
    # Practical stuff
    parser.add_argument('-c', '--clear', action='store_true')
    parser.add_argument('-T', '--timeout', type=int, default=0)
    parser.add_argument('-v', '--validation', nargs='+', default=[])
    parser.add_argument('-m', '--maxtasksperchild', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-l', '--logging', default='WARNING')
    
    args = parser.parse_args()
    
    # Allow remote debugging
    if args.debug:
        from pdb_clone import pdbhandler
        pdbhandler.register()
        # To debug, run: `pdb-attach --kill --pid PID`
        # child pid can be found with: `ps x --forest`
    
    # Set input and output names
    if args.fullname:
        name = args.fullname
    else:
        name = '{}-{}-{}'.format(args.prefix, args.threshold, args.suffix)
    
    if args.output:
        output_name = '{}-{}-{}'.format(args.prefix, args.threshold, args.output)
    else:
        output_name = None
    
    # Load the trainer
    trainer = Trainer.load(name, output_name=output_name)
    
    # Clear list of completed files, if desired
    if args.clear:
        while trainer.completed_files:
            trainer.completed_files.pop()
    
    # Start training again
    trainer.start(timeout=args.timeout,
                  validation=[x+'.pkl' for x in args.validation],
                  debug=args.debug,
                  logging_level=logging.getLevelName(args.logging),
                  maxtasksperchild=args.maxtasksperchild)
