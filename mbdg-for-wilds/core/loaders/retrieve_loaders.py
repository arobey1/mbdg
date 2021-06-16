from core.loaders.camelyon17 import get_camelyon17_loaders, get_camelyon17_munit_loaders
from core.loaders.fmow import get_fmow_loaders, get_fmow_munit_loaders


SUPPROTED_DATASETS = [
    'camelyon17', 'fmow'
]

def retrieve_training_loaders(args):

    if args.dataset.lower() == 'camelyon17':
        return get_camelyon17_loaders(args)

    elif args.dataset.lower() == 'fmow':
        return get_fmow_loaders(args)

    else:
        raise_data_not_supported_error(args.dataset)

def retrieve_munit_loaders(args):
    if args.dataset.lower() == 'camelyon17':
        return get_camelyon17_munit_loaders(args)

    elif args.dataset.lower() == 'fmow':
        return get_fmow_munit_loaders(args)

    else:
        raise_data_not_supported_error(args.dataset)

def get_num_classes(dataset):
    if dataset.lower() == 'camelyon17':
        return 2
    elif dataset.lower() == 'fmow':
        return 62
    else:
        raise_data_not_supported_error(dataset)


def raise_data_not_supported_error(dataset):
    raise NotImplementedError(
        f'Dataset {dataset} is not implemented.\n'  \
        f'Supported datasets are {" ".join(SUPPROTED_DATASETS)}.'
    )
