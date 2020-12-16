import argparse
import math

import torch
import models
import util
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--output-path", type=str, default="output_dataset")
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"])
parser.add_argument("--latent-dim", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1000)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_pack = models.Models(
    args.dataset, "dcgan_teeyo", args.latent_dim, args.model_path
)
model_pack.choose_device(device)
generator = model_pack.model_list["generator"]
generator.eval()

N_CLASSES = 10
N_TRAIN = 50000
N_TEST = 10000

n_batches_train = int(math.ceil(N_TRAIN / args.batch_size))
n_batches_test = int(math.ceil(N_TRAIN / args.batch_size))
class_per_batch = int(math.ceil(args.batch_size / N_CLASSES))


def generate_batch():
    with torch.no_grad():
        z = torch.randn(args.batch_size, args.latent_dim, 1, 1).to(device)
        label_idx = torch.LongTensor(
            [num for num in range(N_CLASSES) for _ in range(class_per_batch)]
        )
        labels = torch.zeros(args.batch_size, N_CLASSES, 1, 1)
        labels[torch.arange(args.batch_size), label_idx, 0, 0] = 1.0
        labels = labels.to(device)

        gen_batch = generator(z, labels)
        del labels, z
        return gen_batch.cpu(), label_idx


logger.info("Generating train batches")
train_batches = []
train_labels = []
for i in tqdm(range(n_batches_train)):
    batch, labels = generate_batch()
    train_batches.append(batch)
    train_labels.append(labels)

train_batches = torch.cat(train_batches, dim=0)
train_labels = torch.cat(train_labels, dim=0)

logger.info("Generating test batches")
test_batches = []
test_labels = []
for i in tqdm(range(n_batches_test)):
    batch, labels = generate_batch()
    test_batches.append(batch)
    test_labels.append(labels)

test_batches = torch.cat(test_batches, dim=0)
test_labels = torch.cat(test_labels, dim=0)

logger.info(f"Saving batches to {args.output_path}")
torch.save(train_batches, f"{args.output_path}/train.pt")
torch.save(train_labels, f"{args.output_path}/train_labels.pt")
torch.save(test_batches, f"{args.output_path}/test.pt")
torch.save(test_labels, f"{args.output_path}/test_labels.pt")
