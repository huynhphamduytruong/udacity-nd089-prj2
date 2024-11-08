# train.py

import argparse
import json
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import get_device, get_transforms_and_loaders
from numpy import array as npArr
from PIL import Image
from torch import nn, optim
from torch import utils as TorchUtils
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, VGG19_Weights, resnet18, vgg19


def get_input_args():
    parser = argparse.ArgumentParser(description="Training the network")

    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory for image folder. E.g. `flowers`",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Output directory",
    )
    # parser.add_argument("--arch", default="vgg19")

    parser.add_argument(
        "--arch",
        choices=["vgg19", "resnet"],
        default="vgg19",
        help="Architecture of the image classifier",
    )

    parser.add_argument(
        "--learning_rate",
        metavar="rate",
        type=float,
        default=0.003,
        help="Learning rate for the training. E.g 0.003",
    )
    # parser.add_argument("--hidden_units", default=4096, type=int)
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Base number of hidden units for the classifier layers. E.g 512",
    )
    parser.add_argument("--output_features", default=102, type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for training. Note the epoch with the lowest validation loss will be saved, not the final epoch. E.g 20",
    )

    parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="Use GPU for training",
    )

    return parser.parse_args()


def build_model(args):
    # Use VGG11 model
    # model = models.vgg11(pretrained=True)
    # model = vgg11(weights = VGG11_Weights.IMAGENET1K_V1)

    if args.arch == "vgg19":
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif args.arch == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Only accept vgg19 or resnet")

    # Turn off gradient of model.features.parameters to train the classifier
    for para in model.features.parameters():
        para.requires_grad = False

    in_classes = 25088 if args.arch == "vgg19" else 512
    hidden_sz = args.hidden_units
    out_classes = 102  # There are 102 flower categories

    fcs = OrderedDict(
        [
            ("fc1", nn.Linear(in_features=in_classes, out_features=hidden_sz)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.2)),
            ("fc2", nn.Linear(hidden_sz, hidden_sz)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(0.2)),
            ("fc3", nn.Linear(in_features=hidden_sz, out_features=out_classes)),
            ("logsm", nn.LogSoftmax(dim=1)),
        ]
    )

    classifier = nn.Sequential(fcs)

    model.classifier = classifier

    return model


def get_optim_criterion(args, model):
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    return optimizer, criterion


def validate(model, criterion, loader):
    running_loss = 0
    accuracy = 0

    model.to(device)  # move model to device

    model.eval()  # set model to evaluation mode (without dropout)

    # Turn off gradient
    with torch.no_grad():
        model.eval()  # set model to evaluation mode (without dropout)

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            # count loss
            running_loss += loss.item()

            # compute accuracy
            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return running_loss / len(loader), accuracy / len(loader)


def train_model(args, device, model, loader, optim, criterion):
    train_loader = loader["train"]
    valid_loader = loader["valid"]

    epochs = args.epochs
    model.to(device)

    print_every = 10  # print result after every few steps
    train_losses, valid_losses = [], []  # loss after every epoch

    for e in range(epochs):
        train_loss = 0
        steps = 0
        model.train()

        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()  # clear gradient in optimizer

            output = model.forward(images)
            loss = criterion(output, labels)
            train_loss += loss.item()
            loss.backward()
            optim.step()

            if steps % print_every == 0 or steps == 1 or steps == len(train_loader):
                print(
                    "|  |- [epoch #{:02}/{:02}:  train  ] >> batch: {}/{}, train_loss items: {:.3f}, train_loss rate: {:.3f}".format(
                        e + 1,
                        epochs,
                        steps,
                        len(train_loader),
                        train_loss,
                        train_loss / steps,
                    )
                )

        print(
            "|  |- [epoch #{:02}/{:02}:  train  ] done. validating...".format(
                e + 1, epochs
            )
        )
        valid_loss, accuracy = validate(model, criterion, valid_loader)
        print(
            "|  |- [epoch #{:02}/{:02}: validate] >> valid_loss: {:.3f}, accuracy: {:.3f}".format(
                e + 1, epochs, valid_loss, accuracy * 100
            )
        )
        # save train and validation loss
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss)

    print("|  \\- train done")
    return train_losses, valid_losses


def save_checkpoint(args, model, dataset, optim, filename="checkpoint.pth"):
    """Save checkpoint."""
    epochs = args.epochs
    learning_rate = args.learning_rate
    save_dir = args.save_dir

    model.class_to_idx = image_datasets["train"].class_to_idx
    checkpoint = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "model": model.cpu(),
        "features": model.features,
        "classifier": model.classifier,
        "class_to_idx": model.class_to_idx,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        # 'criterion_state_dict': criterion.state_dict()
    }

    torch.save(checkpoint, "{}/{}".format(save_dir, filename))
    # print("Save successfully")
    return


def main():
    args = get_input_args()

    print("|- Build dataset, loaders", end="... ")
    data_transforms, image_datasets, dataloaders, labels = get_transforms_and_loaders(
        args.data_dir
    )
    print("done")
    print("|- Init device", end="... ")
    device = get_device(args.gpu)
    print("done")
    print("|- Build model", end="... ")
    model = build_model(args)
    print("done")
    print("|- Build optim, criterion", end="... ")
    optim, criterion = get_optim_criterion(args, model)
    print("done")
    print("|- Training...")

    # Train model
    train_model(args, device, model, dataloaders, optim, criterion)

    # Validate model
    print("|- Validating...")
    test_loader = dataloaders["test"]
    valid_loss, accuracy = validate(model, criterion, test_loader)
    print(
        "|  |- [Test result] >> valid loss: {:.3f}, accuracy: {:.3f}".format(
            valid_loss, accuracy * 100
        )
    )
    print("|  \\- validate done")

    # Save model
    print("|- Save checkpoint", end="... ")
    save_checkpoint(args, model, image_datasets, optim)
    print("|  \\- Save successfully")
    pritn("|")
    print("\\- Train finished")
    return 0


if __name__ == "__main__":
    main()
