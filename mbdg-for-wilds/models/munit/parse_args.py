import argparse
import json
import os

import core.loaders.retrieve_loaders as loaders

DATASETS = loaders.SUPPROTED_DATASETS

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch training')
    
    parser.add_argument('--config', type=str, default='models/munit/munit.yaml', 
                            help='Path to the MUNIT config file.')
    parser.add_argument('--output_path', type=str, default='./models/results', 
                            help="Path where images/checkpoints will be saved")
    parser.add_argument('--data-root', type=str,
                            help='Path to datasets.')
    parser.add_argument('--dataset', type=str, default='camelyon17', choices=DATASETS, 
                            help='Dataset to use for training MUNIT')
    parser.add_argument("--resume", action="store_true",
                            help='Resumes training from last avaiable checkpoint')

    parser.add_argument('--camelyon17-split-scheme', type=str, choices=['official', 'in-dist'], default='official',
                            help='Split scheme for camelyon17 dataset.')
    parser.add_argument('--fmow-split-scheme', type=str, default='official',
                            help='Split scheme for camelyon17 dataset.')
    parser.add_argument('--pacs-train-domains', nargs='+', 
                            help='Training domains for PACS.')
    parser.add_argument('--pacs-test-domain', type=str,
                            help='Test domains for PACS.')

    parser.add_argument('--terra-incog-train-domains', nargs='+', 
                            help='Training domains for Terra Incognita.')
    parser.add_argument('--terra-incog-test-domain', type=str,
                            help='Test domains for Terra Incognita.')

    parser.add_argument('--vlcs-train-domains', nargs='+', 
                            help='Training domains for VLCS.')
    parser.add_argument('--vlcs-test-domain', type=str,
                            help='Test domains for VLCS.')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    _save_args(args)

    return args

def _save_args(args):
    """Save command line arguments to JSON file."""

    fname = os.path.join(args.output_path, 'args.json')
    with open(fname, 'w') as f:
        json.dump(args.__dict__, f, indent=2)