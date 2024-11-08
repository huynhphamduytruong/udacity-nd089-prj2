# import argparse
import json
import warnings

import torch
from torch import utils as TorchUtils
from torchvision import datasets, models, transforms

# *=*=*= Const =*=*=*

meanArr = [0.485, 0.456, 0.406]
stdDevArr = [0.229, 0.224, 0.225]

genericTransforms = [
    transforms.ToTensor(),
    transforms.Normalize(meanArr, stdDevArr),
]

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224),
            *genericTransforms,
        ]
    ),
    "validTest": transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), *genericTransforms]
    ),
}

# *=*=*=*=*=*=*=*=*=*


def get_device(use_gpu=True):
    device_str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if use_gpu and device_str == "cpu":
        warnings.warn(
            "[WRN] Machine doesn't have supported GPU or torch with GPU-supported. Fallback to CPU."
        )

    return device


def get_data_paths(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    return [train_dir, valid_dir, test_dir]


def load_cat_to_name(cat_to_name_path="./"):
    with open(str(cat_to_name_path + "cat_to_name.json"), "r") as f:
        labels = json.load(f)

    return labels


def get_transforms_and_loaders(data_dir, label_dir="./", batch_size=102):
    train_dir, valid_dir, test_dir = get_data_paths(data_dir)

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(
            valid_dir, transform=data_transforms["validTest"]
        ),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["validTest"]),
    }

    # Using the image datasets and the transforms, define the dataloaders
    DataLoader = TorchUtils.data.DataLoader
    dataloaders = {
        "train": DataLoader(
            image_datasets["train"], batch_size=batch_size, shuffle=True
        ),
        "valid": DataLoader(image_datasets["valid"], batch_size=batch_size),
        "test": DataLoader(image_datasets["test"], batch_size=batch_size),
    }

    # with open(str(label_dir + "cat_to_name.json"), "r") as f:
    labels = load_cat_to_name(label_dir)

    return data_transforms, image_datasets, dataloaders, labels
