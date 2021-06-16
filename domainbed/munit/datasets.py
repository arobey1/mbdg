import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class ColoredMNIST(Dataset):
    def __init__(self, env, env_idx, n_envs):
        super().__init__()
        
        transform = Compose([Resize((32, 32)), ToTensor()])

        root = '../data/MNIST'
        original_dataset_tr = MNIST(root, train=True, download=False, transform=transform)
        original_dataset_te = MNIST(root, train=False, download=False, transform=transform)
        full_dataset = ConcatDataset([original_dataset_tr, original_dataset_te])

        orig_imgs = torch.cat([img for img, _ in full_dataset])
        orig_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in full_dataset])

        # original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))
        # original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        images = orig_imgs[env_idx::n_envs]
        labels = orig_labels[env_idx::n_envs]
        self.dataset = self.color_dataset(images, labels, env)

    def color_dataset(self, images, labels, environment):

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        return images.float()

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

    def __getitem__(self, index):
        x = self.dataset[index]
        x = torch.cat([x, torch.zeros(1, 32, 32)], dim=0)
        return x

    def __len__(self):
        return  len(self.dataset)


def get_mnist_loaders():

    # env0_dataset = ColoredMNIST(env=0.1, env_idx=0, n_envs=2)
    env1_dataset = ColoredMNIST(env=0.2, env_idx=1, n_envs=2)
    env3_dataset = ColoredMNIST(env=0.9, env_idx=2, n_envs=2)
    datasets = ConcatDataset([env1_dataset, env3_dataset])
    loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True)

    return loader, loader, loader, loader

