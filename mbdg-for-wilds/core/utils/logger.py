import logging
import os

def get_logger(fname, args):

    log_root = os.path.join(args.logging_root, args.train_alg)
    os.makedirs(log_root, exist_ok=True)
    fname = os.path.join(log_root, fname + '.log')

    with open(fname, 'w'):
        pass

    handlers = [logging.FileHandler(fname), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='  %(message)s', handlers=handlers)

    return logging.getLogger(f'Alg:{args.train_alg.upper()}')