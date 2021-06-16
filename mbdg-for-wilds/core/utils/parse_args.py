import argparse
import json
import os

ALGORITHMS = ['ERM', 'MBDG-Reg', 'MBDG']
OPTIMIZERS = ['Adam', 'SGD', 'AdaDelta']
DISTANCE_METRICS = ['KL']

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch training')
    
    # training
    parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training.')
    parser.add_argument('--n-epochs', type=int, default=20,
                            help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-2,
                            help='Learning rate for training.')
    parser.add_argument('--half-prec', action='store_true',
                            help='Performs mixed-precision training.')
    parser.add_argument('--log-interval', type=int, default=100,
                            help='How often to print during training.')
    parser.add_argument('--architecture', type=str, default='resnet18',
                            help='Architecture for classifier.')
    parser.add_argument('--pretrained', action='store_true', 
                            help='Use pretrained weights from ImageNet-trained classifier.')
    parser.add_argument('--trial-index', type=int, default=0,
                            help='Trial index for trianing.')
    parser.add_argument('--short-epoch', action='store_true',
                            help='Short epoch with 1000 training/val/test images for debugging.')
    parser.add_argument('--optim', type=str, default='SGD', choices=OPTIMIZERS,
                            help='Optimizer to use for training.')
    parser.add_argument('--momentum', type=float, default=0.9,
                            help='Use momentum for optimization (if available).')

    # algorithms
    parser.add_argument('--train-alg', type=str, choices=ALGORITHMS, default='ERM',
                            help='Training algorithm.')
    parser.add_argument('--dual-var-init-val', type=float, default=1.0,
                            help='Default initialization for dual variables.')
    parser.add_argument('--dual-lr', type=float, default=1e-2,
                            help='Learning rate for dual ascent.')
    parser.add_argument('--mbdg-static-lam-dist', type=float, default=0.5,
                            help='Static trade-off parameter for distance constraint in MBDG.')
    parser.add_argument('--mbdg-static-lam-grad', type=float, default=0.0,
                            help='Static trade-off parameter for gradient constraint in MBDG.')
    parser.add_argument('--mbdg-num-steps', type=int, default=1,
                            help='Number of steps to take in inner loop of MBDG.')
    parser.add_argument('--mbdg-dist-metric', type=str, choices=DISTANCE_METRICS, default='KL',
                            help='Distance metric for MBDG.')
    parser.add_argument('--mbdg-gamma', type=float, default=0.1, 
                            help='Constraint margin for MBDG.')
    parser.add_argument('--mbdg-dual-step-size', type=float, default=0.01,
                            help='Dual ascent step size for MBDG.')

    # model of natural variation
    parser.add_argument('--model-path', type=str,
                            help='Path to saved model of natural variation.')
    parser.add_argument('--munit-config-path', type=str,
                            help='Path to MUNIT config .yaml file.')
    parser.add_argument('--delta-dim', type=int, default=8,
                            help='Dimension of nuisnace space.')

    # paths
    parser.add_argument('--results-path', type=str, 
                            help='Path to save outputs, images, plots, etc.')
    parser.add_argument('--data-root', type=str,
                            help='Path to datasets.')
    parser.add_argument('--logging-root', type=str,
                            help='Path to logging directory.')

    # distributed utils
    parser.add_argument('--distributed', action='store_true', 
                            help='Run distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                            help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                            help='Used for multi-process training. Can either be manually set ' +
                            'or automatically set by using \'python -m multiproc\'.')

    # datasets
    parser.add_argument('--dataset', type=str, required=True,
                            help='Dataset to use for training/testing')

    # camelyon17 arguments
    parser.add_argument('--camelyon17-split-scheme', type=str, choices=['official', 'in-dist'], default='official',
                            help='Split scheme for camelyon17 dataset.')

    # fmow arguments
    parser.add_argument('--fmow-split-scheme', type=str, default='official',
                            help='Split scheme for camelyon17 dataset.')

    args = parser.parse_args()
    os.makedirs(args.results_path, exist_ok=True)
    _save_args(args)

    return args

def _save_args(args):
    """Save command line arguments to JSON file."""

    fname = os.path.join(args.results_path, 'args.json')
    with open(fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)