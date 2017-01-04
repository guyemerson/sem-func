import argparse, numpy as np

from trainer import Trainer

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, threshold=np.nan)
    
    parser = argparse.ArgumentParser(description="Train a sem-func model")
    # File name
    parser.add_argument('fullname', nargs='?', default=None)
    parser.add_argument('-p', '-prefix', default='multicore')
    parser.add_argument('-t', '-threshold', type=int, default=5)
    parser.add_argument('-s', '-suffix', default=None)
    parser.add_argument('-o', '-output', default=None)
    # Practicalities
    parser.add_argument('-clear', action='store_true')
    parser.add_argument('-timeout', type=int, default=0)
    parser.add_argument('-validation', nargs='+', default=[])
    parser.add_argument('-maxtasksperchild', type=int, default=None)
    
    args = parser.parse_args()
    
    # Set input and output names
    if args.name:
        name = args.name
    else:
        name = '{}-{}-{}'.format(args.prefix, args.threshold, args.suffix)
    
    if args.output:
        output_name = '{}-{}-{}'.format(args.prefix, args.threshold, args.output)
    else:
        output_name = None
    
    # Load the file
    trainer = Trainer.load(name, output_name=output_name)
    
    # Clear list of completed files, if desired
    if args.clear:
        while trainer.completed_files:
            trainer.completed_files.pop()
    
    # Start training again
    trainer.start(timeout=args.timeout,
                  validation=args.validation,
                  maxtasksperchild=args.maxtasksperchild)
