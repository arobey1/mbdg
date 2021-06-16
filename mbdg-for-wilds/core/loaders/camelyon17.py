from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

from core.loaders.utils import to_sampler, to_loader

def get_camelyon17_loaders(args):
    kwargs = {'root': args.data_root, 'split_scheme': args.camelyon17_split_scheme,
                'return_labels': True, 'return_meta_data': False, 'short_epoch': args.short_epoch}

    train_data = CamelyonDataset(split='train', **kwargs)
    val_data = CamelyonDataset(split='val', **kwargs)
    test_data = CamelyonDataset(split='test', **kwargs)

    train_smp, val_smp, test_smp = to_sampler(args.distributed, train_data, val_data, test_data)
    
    train_ldr = to_loader(train_data, args.batch_size, train_smp, train=True)
    val_ldr = to_loader(val_data, args.batch_size, val_smp, train=True)
    test_ldr = to_loader(test_data, args.batch_size, test_smp, train=False)

    return train_ldr, train_smp, val_ldr, val_smp, test_ldr, test_smp

def get_camelyon17_munit_loaders(args):
    kwargs = {'root': args.data_root, 'split_scheme': args.camelyon17_split_scheme,
                'return_labels': False, 'return_meta_data': False}

    train_data = CamelyonDataset(split='train', **kwargs)
    test_data = CamelyonDataset(split='test', **kwargs)

    train_ldr_A = train_ldr_B = to_loader(train_data, bs=1, samp=None, train=True)
    test_ldr_A = test_ldr_B = to_loader(test_data, bs=1, samp=None, train=False)

    return train_ldr_A, train_ldr_B, test_ldr_A, test_ldr_B

class CamelyonDataset:
    def __init__(self, split, root, split_scheme, return_labels=True, 
                    return_meta_data=False, short_epoch=False):

        xforms = transforms.ToTensor()
        dataset = Camelyon17Dataset(root_dir=root, split_scheme=split_scheme)
        self._data = dataset.get_subset(split, transform=xforms)

        if short_epoch is True:
            self._data = Subset(self._data, range(1000))

        self._return_labels = return_labels
        self._return_meta_data = return_meta_data

    def __getitem__(self, index):
        imgs, labels, meta_data = self._data[index]

        if self._return_labels is True and self._return_meta_data is True:
            return imgs, labels, meta_data
        elif self._return_labels is True:
            return imgs, labels
        elif self._return_meta_data is True:
            return imgs, meta_data
        else:
            return imgs

    def __len__(self):
        return len(self._data)
