import torch
import torch.nn as nn


def weights_init(m):
    """
    Custom weight init that everybody seems to do
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    # TODO: implement generator
    def __init__(self):
        super(Generator, self).__init__()
        raise NotImplementedError

    def forward(self, input, label):
        raise NotImplementedError


class Discriminator(nn.Module):
    # TODO: implement discriminator
    def __init__(self):
        super(Discriminator, self).__init__()
        raise NotImplementedError

    def forward(self, input, label):
        raise NotImplementedError


def load_pretrained_classifier(path: str) -> nn.Module:
    # TODO: implement loading of classifier
    raise NotImplementedError
