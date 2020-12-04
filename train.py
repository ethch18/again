import argparse
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import data
import models
import util

parser = argparse.ArgumentParser()
parser.add_argument("--n-epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--latent-dim", type=int, default=100)
parser.add_argument("--sample-interval", type=int, default=400)
parser.add_argument("--checkpoint-interval", type=int, default=5)
parser.add_argument("--data-path", type=str, default="data/")
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"])
parser.add_argument("--normalize-data", action="store_true")
parser.add_argument("--image-size", type=str)
parser.add_argument("--output-path", type=str, default="output")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--target-class", type=int)
parser.add_argument('--resume', default=None, type=str, help='Resuming model path')
args = parser.parse_args()

device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

args.output_path = util.get_output_folder(args.output_path, args.dataset)

# Loss
generation_loss = nn.BCELoss()
classification_loss = nn.CrossEntropyLoss()

# Models
model_pack = models.Models("mnist", "dcgan", args.resume)
model_pack.choose_device(device)
generator = model_pack.model_list["generator"]
discriminator = model_pack.model_list["generator"]
classifier = model_pack.model_list["classifier"]

# Data
dataclass = (
    data.MNISTDataLoader()
    if args.dataset == "mnist"
    else data.CIFAR10DataLoader()
)
dataloader = torch.utils.data.DataLoader(
    dataclass.get_dataset(
        args.data_path, normalize=args.normalize_data, resize=args.image_size
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# Optimizers
generator_optim = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
)
discriminator_optim = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row ** 2, args.latent_dim)
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.LongTensor(
        [num for _ in range(n_row) for num in range(n_row)]
    )
    gen_imgs = generator(z, labels)
    save_image(
        gen_imgs.data,
        f"{args.output_path}/image_{batches_done}.png",
        nrow=n_row,
        normalize=True,
    )


# Training Loop
for epoch in range(args.n_epochs):
    for i, (imgs, labels) in tqdm(enumerate(dataloader)):
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size)
        fake = torch.zeros(batch_size)

        # Generate fake images according to the real labels
        z = torch.randn(batch_size, args.latent_dim)
        generated_images = generator(z, labels)

        # Train Discriminator
        discriminator_optim.zero_grad()
        discrim_real = discriminator(imgs, labels)
        discrim_real_loss = generation_loss(discrim_real, valid)

        discrim_fake = discriminator(generated_images.detach(), labels)
        discrim_fake_loss = generation_loss(discrim_fake, fake)

        discrim_loss = (discrim_real_loss + discrim_fake_loss) / 2
        discrim_loss.backward()
        discriminator_optim.step()

        # Train Generator
        generator_optim.zero_grad()
        discrim_generator = discriminator(generated_images, labels)
        discrim_generator_loss = generation_loss(discrim_generator, valid)

        classifier_logits = classifier(generated_images)
        if args.target_class:
            target = args.target_class * torch.ones(batch_size)
            classifier_loss = classification_loss(classifier_logits, target)
        else:
            classifier_loss = -classification_loss(classifier_logits, labels)

        gener_loss = (discrim_generator_loss + classifier_loss) / 2
        gener_loss.backward()
        generator_optim.step()

        if i % args.sample_interval == 0:
            print(
                f"[Epoch {epoch}] [Batch {i}] [D Loss: {discrim_loss.item()}] "
                f"[G Loss Full: {gener_loss.item()}] "
                f"[G Loss D: {discrim_generator_loss.item()}] "
                f"[G Loss C: {classifier_loss.item()}]"
            )
            sample_image(n_row=10, batches_done=(epoch * len(dataloader) + i))

    if epoch % args.checkpoint_interval == 0:
        save_path = os.path.join(f"{args.output_path}", f"{epoch}")
        os.makedirs(save_path, exist_ok=True)
        model_pack.save_weights(save_path, device)

model_pack.save_weights(f"{args.output_path}/final")
