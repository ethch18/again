import os
import torch
from torchvision import datasets, transforms


class DataLoader():
    def __init__(self):
        return

    def dataset_name(self):
        raise NotImplementedError

    def dataset_dim(self):
        raise NotImplementedError

    def num_classes(self):
        raise NotImplementedError

    def get_dataset(self, path):
        raise NotImplementedError


class MNISTDataLoader(DataLoader):
    def __init__(self):
        super(MNISTDataLoader, self).__init__()

    def dataset_name(self):
        return "MNIST"

    def dataset_dim(self):
        return 28

    def num_classes(self):
        return 10

    def get_dataset(self, path, normalize=False, resize=None):
        os.makedirs(path, exist_ok=True)
        xform = []
        if resize is not None:
            xform.append(transforms.Resize(resize))
        xform.append(transforms.ToTensor())
        if normalize:
            xform.append(transforms.Normalize((0.5,), (0.5,)))

        return datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(xform),
        )


class CIFAR10DataLoader(DataLoader):
    def __init__(self):
        super(CIFAR10DataLoader, self).__init__()

    def dataset_name(self):
        return "CIFAR10"

    def dataset_dim(self):
        return 32

    def num_classes(self):
        return 10

    def get_dataset(self, path, normalize=False, resize=None):
        os.makedirs(path, exist_ok=True)
        xform = []
        if resize is not None:
            xform.append(transforms.Resize(resize))
        xform.append(transforms.ToTensor())
        if normalize:
            xform.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(xform),
        )
