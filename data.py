import os
import torch
from torchvision import datasets, transforms


class DataLoader:
    def __init__(self):
        return

    @staticmethod
    def dataset_name():
        raise NotImplementedError

    @staticmethod
    def dataset_dim():
        raise NotImplementedError

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def get_dataset(*args, **kwargs):
        raise NotImplementedError


class MNISTDataLoader(DataLoader):
    def __init__(self):
        super(MNISTDataLoader, self).__init__()

    @staticmethod
    def dataset_name():
        return "MNIST"

    @staticmethod
    def dataset_dim():
        return 28

    @staticmethod
    def num_classes():
        return 10

    @staticmethod
    def get_dataset(path, normalize=False, resize=None):
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

    @staticmethod
    def dataset_name():
        return "CIFAR10"

    @staticmethod
    def dataset_dim(self):
        return 32

    @staticmethod
    def num_classes():
        return 10

    @staticmethod
    def get_dataset(path, normalize=False, resize=None):
        os.makedirs(path, exist_ok=True)
        xform = []
        if resize is not None:
            xform.append(transforms.Resize(resize))
        xform.append(transforms.ToTensor())
        if normalize:
            xform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

        return datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(xform),
        )
