import argparse
import math
import data
import inception
import logging
import models
import torch
import util

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--dataset-path", type=str, required=True)
parser.add_argument("--raw-data-path", type=str)
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"])
parser.add_argument("--latent-dim", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1000)
parser.add_argument("--inception", action="store_true")
parser.add_argument("--clean-accuracy", action="store_true")
parser.add_argument("--adversarial-accuracy", action="store_true")
parser.add_argument("--mitigation", action="store_true")
parser.add_argument("--mitigation-epochs", type=int, default=-1)
parser.add_argument("--mitigation-lr", type=float, default=0.0002)
parser.add_argument("--mitigation-beta1", type=float, default=0.5)
parser.add_argument("--mitigation-beta2", type=float, default=0.999)
parser.add_argument("--normalize-data", action="store_true")
parser.add_argument("--image-size", type=int, default=32)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_pack = models.Models(
    args.dataset, "dcgan_teeyo", args.latent_dim, args.model_path
)
model_pack.choose_device(device)
classifier = model_pack.model_list["classifier"]

train, train_labels, test, test_labels = util.load_dataset(args.dataset_path)

if args.inception:
    inception_dataset = torch.cat((train, test), dim=0)
    incep_score = inception.score(inception_dataset, args.batch_size, device)
    logger.info(f"INCEPTION SCORE: {incep_score}")

if args.clean_accuracy and args.raw_data_path:
    dataclass = (
        data.MNISTDataLoader()
        if args.dataset == "mnist"
        else data.CIFAR10DataLoader()
    )

    dataloader = torch.utils.data.DataLoader(
        dataclass.get_dataset(
            args.raw_data_path,
            normalize=args.normalize_data,
            resize=args.image_size,
            train=False,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    logger.info(f"Evaluating clean accuracy")
    correct = 0
    total = 0
    classifier.eval()
    with torch.no_grad():
        for i, (imgs, labels) in tqdm(enumerate(dataloader)):
            predictions = classifier(imgs.to(device))
            correct += (predictions == labels).sum().item()
            total += imgs.size(0)

    logger.info(f"CLEAN ACCURACY: {correct / total}")

if args.adversarial_accuracy:
    n_batches = int(math.ceil(test.size(0) / args.batch_size))
    logger.info(f"Evaluating adversarial accuracy")
    correct = 0
    total = 0
    classifier.eval()
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            batch = test[i * args.batch_size : (i + 1) * args.batch_size]
            batch_labels = test_labels[
                i * args.batch_size : (i + 1) * args.batch_size
            ]
            predictions = classifier(batch.to(device))
            correct += (predictions == batch_labels).sum().item()
            total += batch.size(0)

    logger.info(f"ADVERSARIAL ACCURACY: {correct / total}")

if args.mitigation and args.mitigation_epochs > 0:
    train_dataset = torch.utils.data.TensorDataset(train, train_labels)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(
        classifier.parameters(),
        lr=args.mitigation_lr,
        betas=(args.mitigation_beta1, args.mitigation_beta2),
    )
    classifier.train()
    logger.info(f"Mitigation training for {args.mitigation_epochs} epochs")
    for epoch in range(args.mitigation_epochs):
        for i, (imgs, labels) in tqdm(enumerate(train_dataloader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.size(0)

            preds = classifier(imgs)
            loss = criterion(preds, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
    logger.info("Mitigation training done.  Evaluating adversarial accuracy")

    correct = 0
    total = 0
    classifier.eval()
    n_batches = int(math.ceil(test.size(0) / args.batch_size))
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            batch = test[i * args.batch_size : (i + 1) * args.batch_size].to(
                device
            )
            batch_labels = test_labels[
                i * args.batch_size : (i + 1) * args.batch_size
            ].to(device)
            predictions = classifier(batch)
            correct += (predictions == batch_labels).sum().item()
            total += batch.size(0)

    logger.info(f"ADVERSARIAL ACCURACY: {correct / total}")
