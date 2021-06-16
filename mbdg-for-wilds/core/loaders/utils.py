from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def to_sampler(distributed, *datasets):
    if distributed is True:
        return [DistributedSampler(d) if d is not None else None for d in datasets]
    return [None for d in datasets]

def to_loader(dataset, bs, samp, train):

    if dataset is not None:
        shuffle = False if train is False or samp is not None else True
        return DataLoader(dataset, batch_size=bs, num_workers=4, pin_memory=True,
            shuffle=shuffle, sampler=samp)
    return None