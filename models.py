import os
import torch
import torch.nn as nn

num_gpu = 1 if torch.cuda.is_available() else 0

def weights_init(m):
    """
    Custom weight init that everybody seems to do
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)
class Models():
    def __init__(self, dateset_name, model_name, path=None):
        self.model_list = {}

        self.import_model(dateset_name, model_name)
        if path == None:
            path = os.path.join("./pretrained/", f"{dateset_name}")
        self.load_weights(path)

    def import_model(self, dateset_name, model_name):
        if dateset_name == "mnist":
            from references.csinva.mnist_classifier.lenet import LeNet5 as pretrained_classifier
            from references.csinva.mnist_dcgan.dcgan import Discriminator, Generator
        else: # for CIFAR
            # TODO: add cifar
            raise NotImplementedError

        if model_name == "dcgan":
            self.model_list["generator"] = Generator(num_gpu)
            self.model_list["discriminator"] = Discriminator(num_gpu)
        else: # vae
            # TODO: add cifar
            raise NotImplementedError
        self.model_list["classifier"] = pretrained_classifier()
        self.model_list["classifier"].load_state_dict(torch.load(f'./pretrained/{dateset_name}/classifier.pth'))
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
                self.model_list[m].load_state_dict(torch.load(f'{path}/{m}.pth'))
    
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
                torch.save(self.model_list[m].state_dict(),f'{path}/{m}.pth')
        self.choose_device(device)
    
    def choose_device(self, device):
        for m in self.model_list:
            self.model_list[m].to(device)
