# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import config  # Import configuration


# Define transformations - resizing and normalization are crucial for VGG16

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((config.RESIZE_HEIGHT, config.RESIZE_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ]),
    'test': transforms.Compose([
        transforms.Resize((config.RESIZE_HEIGHT, config.RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ]),
}


github_data_path = '/scratch/ajinkyar/Project_Ranshur_Ajinkya_Nitinkumar/data'


# Load datasets using ImageFolder
train_dataset = torchvision.datasets.ImageFolder(
    root=f"{github_data_path}/train",  # Path to the 'train' subfolder
    transform=data_transforms['train']
)

test_dataset = torchvision.datasets.ImageFolder(
    root=f"{github_data_path}/test",  # Path to the 'test' subfolder
    transform=data_transforms['test']
)


train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE_DATA,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


# --- Assign loaders to expected names ---
your_train_loader = train_loader
your_valid_loader = test_loader
