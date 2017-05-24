from __future__ import print_function, division
import platform
import os
from os.path import join
from copy import deepcopy
import argparse
from datasets.utils import pascal_classes

# --- settings common to train and eval ---
defaults = argparse.Namespace()
defaults.OUTPUT_DIR = './weights'

# --- train settings ---

defaults_train = deepcopy(defaults)
defaults_train.BATCH_SIZE = 1
defaults_train.GRAD_UPDATE_EVERY = 10  # gradient accumulation
defaults_train.INPUT_SIZE = '321,321'
defaults_train.LEARNING_RATE = 5e-4
defaults_train.MOMENTUM = 0.9
defaults_train.NUM_STEPS = 1000
defaults_train.RANDOM_SEED = 1234
defaults_train.SAVE_NUM_IMAGES = 1
defaults_train.SAVE_PRED_EVERY = 2000
defaults_train.DO_VAL_EVERY = 300

#  --- eval settings ---
defaults_eval = deepcopy(defaults)

def get_arguments(argv, mode='eval'):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    classes = pascal_classes(with_void=False)
    inv_classes = pascal_classes(with_void=False, reverse=True)
    def pascal_cls(s):
        n_classes = len(classes)
        if s in classes:
            return classes[s]
        elif 0 <= int(s) < n_classes:
            return int(s)
        raise argparse.ArgumentTypeError('Pascal classes: {}'.format(classes))

    parser = argparse.ArgumentParser(description="Deeplab-resnet-multiscale")
    if mode == 'eval':
        defaults = defaults_eval
    elif mode == 'train':
        defaults = defaults_train
    parser.add_argument("--expname", type=str, required=True,
                        help="Name of the experiment.")
    parser.add_argument("--nickname", type=str, required=True,
                        help="Name given to the run (useful for output paths and logging).")
    parser.add_argument("--restore-from", type=str, required=True,
                        help="Where restore model parameters from.")
    parser.add_argument("--binary", type=pascal_cls, metavar="[0-20]", default=-1,
                        help="Binary classifier with specified class. (class name or id)")
    parser.add_argument("--sampling", type=str, choices=['sequential', 'shuffle', 'balanced', 'exclusive'],
                            default='shuffle', help="Trainset sampling (balanced applies to binary)")
    if mode == 'eval':
        parser.add_argument("--num-steps", type=int, default=defaults.NUM_STEPS,
                        help="Number of images in the validation set.")
    if mode == 'train':
        parser.add_argument("--threads", type=int, default=4,
                        help="Number of data fetcher threads")
        parser.add_argument("--epochs", action="store_true",
                            help="Count steps in epochs (1 step is 1 epoch)")
        parser.add_argument("--proximal", action="store_true",
                            help="Use proximal variant")
        parser.add_argument("--proxreg", type=float, default=0.5,
                            help="proximal parameter")
        parser.add_argument("--maxproxsteps", type=int, default=30,
                            help="maximal prox. computation steps")
        parser.add_argument("--no-startval", action="store_true",
                            help="Don't start with a validation run")
        parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--grad-update-every", type=int, default=defaults.GRAD_UPDATE_EVERY,
                            help="Number of steps after which gradient update is applied.")
        parser.add_argument("--input-size", type=str, default=defaults.INPUT_SIZE,
                            help="Comma-separated string with height and width of images.")
        parser.add_argument("-lr", "--learning-rate", type=float, default=defaults.LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=defaults.MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--no-random-mirror", action="store_false",
                            help="No random mirror of the inputs during the training.")
        parser.add_argument("--no-random-scale", action="store_false",
                            help="No random scale of the inputs during the training.")
        parser.add_argument("--save-pred-every", type=int, default=defaults.SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--do-val-every", type=int, default=defaults.DO_VAL_EVERY,
                            help="Do validation every...")
        parser.add_argument("--jaccard", action="store_true",
                            help="Use lovasz hinge in the binary case.")
        parser.add_argument("--hinge", action="store_true",
                            help="Use hinge loss.")
        parser.add_argument("--num-steps", type=float, default=defaults.NUM_STEPS,
                        help="Number of training steps.")
        parser.add_argument("--start-step", type=int, default=0,
                        help="Start from this training set.")
        parser.add_argument("--train-last", type=int, default=-1,
                        help="Train last .. layers (default -1 -> all).")
        parser.add_argument("--schedule", action="store_true",
                        help="Use learning rate schedule.")
        parser.add_argument("--delete-previous", action="store_true",
                        help="Delete previous logdir if exists.")
    args = parser.parse_args(argv)
    args.snapshot_dir = join(args.output_dir, args.expname)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if args.binary != -1:
        args.binary_str = inv_classes[args.binary]
        print('binary selected, class ' + args.binary_str)
    if mode == 'train':
        args.random_mirror = not args.no_random_mirror
        args.random_scale = not args.no_random_scale
    if args.sampling == 'exclusive':
        if args.binary == -1:
            parser.error('The --exclusive flag requires --binary set.')
    if args.sampling == 'balanced':
        if args.binary == -1:
            parser.error('The --balanced flag requires --binary set.')
    if args.jaccard:
        if args.binary == -1:
            parser.error('Jaccard loss requires --binary set.')
    if args.hinge:
        if args.binary == -1:
            parser.error('Hinge loss requires --binary set.')
        
    return args