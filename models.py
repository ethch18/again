import os
import torch
import torch.nn as nn
import torch.nn.functional as F

num_gpu = 1 if torch.cuda.is_available() else 0


def weights_init(m):
    """
    Custom weight init that everybody seems to do
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Models:
    def __init__(self, dateset_name, model_name, latent_dim, path=None):
        self.model_list = {}

        self.import_model(dateset_name, model_name, latent_dim)
        if path == None:
            self.model_list["generator"].apply(weights_init)
            self.model_list["discriminator"].apply(weights_init)
        else:
            # path = os.path.join("./pretrained/", f"{dateset_name}")
            self.load_weights(path)

    def import_model(self, dateset_name, model_name, latent_dim):
        classifier_weight = f"./pretrained/{dateset_name}/classifier.pth"
        if dateset_name == "mnist":
            from references.csinva.mnist_classifier.lenet import (
                LeNet5 as pretrained_classifier,
            )
            self.model_list["classifier"] = pretrained_classifier()
            self.model_list["classifier"].load_state_dict(
                torch.load(classifier_weight)
            )

            channels = 1
        else:  # for CIFAR
            # TODO: actually add cifar
            import references.csinva.cifar10_classifier.model as pretrained_classifier
            self.model_list["classifier"] = pretrained_classifier.cifar10(128, pretrained=classifier_weight).eval()

            channels = 3

        if model_name == "dcgan":
            from references.malzantot.conditional_dcgan import (
                ModelD as Discriminator,
            )
            from references.malzantot.conditional_dcgan import (
                ModelG as Generator,
            )
            self.model_list["generator"] = Generator(latent_dim)
            self.model_list["discriminator"] = Discriminator()
        elif model_name == "dcgan_teeyo":
            self.model_list["generator"] = TeeyoGenerator(
                channels=channels, latent_dim=latent_dim
            )
            self.model_list["discriminator"] = TeeyoDiscriminator(
                channels=channels
            )
        else:  # vae
            # TODO: add vae?
            raise NotImplementedError

        self.model_list["classifier"].requires_grad = False

    def eval(self):
        for m in self.model_list:
            self.model_list[m].eval()

    def train(self):
        for m in self.model_list:
            if m != "classifier":
                self.model_list[m].train()
            else:
                self.model_list[m].eval()

    def load_weights(self, path):
        """
        @param path: <dataset_name>_run + <run#>/<epoch>
        E.g. MNIST 2nd run, 99 epoch:
            mnist_run2/99/discrimitor.pth
            mnist_run2/99/generator.pth
        """
        for m in self.model_list:
            if m != "classifier":
                self.model_list[m].load_state_dict(
                    torch.load(f"{path}/{m}.pth")
                )

    def save_weights(self, path, device):
        """
        @param path: <dataset_name>_run + <run#>/<epoch>
        E.g. MNIST 2nd run, 99 epoch:
            mnist_run2/99/discrimitor.pth
            mnist_run2/99/generator.pth
        """
        for m in self.model_list:
            if m != "classifier":
                self.model_list[m].cpu()
                torch.save(self.model_list[m].state_dict(), f"{path}/{m}.pth")
        self.choose_device(device)

    def choose_device(self, device):
        for m in self.model_list:
            self.model_list[m].to(device)


class TeeyoGenerator(nn.Module):
    def __init__(self, *, channels, dim=128, latent_dim=100):
        super(TeeyoGenerator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(latent_dim, dim * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(dim * 2)
        self.deconv1_2 = nn.ConvTranspose2d(10, dim * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(dim * 2)
        self.deconv2 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(dim * 2)
        self.deconv3 = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(dim)
        self.deconv4 = nn.ConvTranspose2d(dim, channels, 4, 2, 1)

    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x


class TeeyoDiscriminator(nn.Module):
    def __init__(self, *, channels, dim=128):
        super(TeeyoDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, dim // 2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, dim // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(dim * 4, 1, 4, 1, 0)

    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x
