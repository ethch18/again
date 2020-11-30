# AGAN: Adversarial Generative Adversarial Networks

## Resources:
[Original Conditional GAN Paper](https://arxiv.org/abs/1411.1784)
[Extension on CGAN](http://cs231n.stanford.edu/reports/2015/pdfs/jgauthie_final_report.pdf)
[DCGAN Paper](https://arxiv.org/pdf/1511.06434.pdf)

### Implementations
[malzantot/Pytorch-conditional-GANs](https://github.com/malzantot/Pytorch-conditional-GANs/blob/master/conditional_dcgan.py)
[pytorch/examples/dcgan](https://github.com/pytorch/examples/blob/master/dcgan/main.py)
[eriklindernoren/PyTorch-GAN/dcgan](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py)
[eriklindernoren/PyTorch-GAN/cgan](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py)
[TeeyoHuang/conditional-GAN/conditional_DCGAN](https://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_DCGAN.py)

### Other Links
https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6
https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

### Pretrained Class Discriminator
<ul>
<li>

**[aaron-xichen/pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)**: Contains pretrained classifiers for mnist, cifar, imagenet, etc. MLP with dropout for mnist, CNN with BatchNorm for cifar. 

<li>

**[csinva/gan-vae-pretrained-pytorch](https://github.com/csinva/gan-vae-pretrained-pytorch)**: Contains both pretrained DCGAN and classifier. The classifier weights are from above (aaron-xichen/pytorch-playground). 

<li>
Other MNIST Weights:
<ul>
<li> Use PyTorch Pretrained VGG16

<li>

[https://nextjournal.com/gkoehler/pytorch-mnist](https://nextjournal.com/gkoehler/pytorch-mnist)

<li>

[https://www.kaggle.com/tonysun94/pytorch-1-0-1-on-mnist-acc-99-8](https://www.kaggle.com/tonysun94/pytorch-1-0-1-on-mnist-acc-99-8)
</ul>

<li> Freeze weights: 

```
discriminator.requires_grad = False
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
 ```

</ul>
