
import os

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import numpy as np

num_workers = 2
#os.cpu_count() 
dataset_path = 'data_dir/eurosat/2750'
dataset = datasets.ImageFolder(dataset_path)
class EuroSAT(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
          x = self.transform(dataset[index][0])
        else:
          x = dataset[index][0]
        y = dataset[index][1]
        return x, y

    def __len__(self):
        return len(dataset)
    

def image_net_transforms(dataset):
    input_size = 224
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    train_data = EuroSAT(dataset, train_transform)
    test_data = EuroSAT(dataset, test_transform)
    return (train_data, test_data)

def sentinel_transforms(dataset):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.8,1.0)), # multilabel, avoid cropping out labels
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    train_data = EuroSAT(dataset, train_transforms)
    test_data = EuroSAT(dataset, test_transforms)
    return (train_data, test_data)




def data_split(train_data, test_data, dataset):

    # Randomly split the dataset into 80% train / 20% test
    # by subsetting the transformed train and test datasets
    train_size = 0.6
    val_size = 0.5
    indices = list(range(int(len(dataset))))
    train_split = int(train_size * len(dataset))
    val_split = int(val_size * (len(dataset)-train_split)) + train_split
    np.random.shuffle(indices)

    train_data = data.Subset(train_data, indices=indices[:train_split])
    val_data = data.Subset(test_data, indices=indices[train_split:val_split])
    test_data = data.Subset(test_data, indices=indices[val_split:])
    print("Train/val/test sizes: {}/{}/{}".format(len(train_data), len(val_data), len(test_data)))
    return (train_data, val_data, test_data)


def data_loaders(train_data, val_data, test_data, batch_size ):
    train_loader = data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
    )

    val_loader = data.DataLoader(
    val_data, batch_size=batch_size,  shuffle=False
    )

    test_loader = data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader
   
def create_dataloaders(dataset_path, pre_train_type, batch_size):
    dataset = datasets.ImageFolder(dataset_path)
    if pre_train_type == 'imagenet':
        image_net_train_data, image_net_test_data = image_net_transforms(dataset)
        train_data, val_data, test_data = data_split(image_net_train_data, image_net_test_data, dataset)
        train_loader, val_loader, test_loader = data_loaders(train_data, val_data, test_data, batch_size)
    else:
        sentinel_train_data, sentinel_test_data = sentinel_transforms(dataset)
        train_data, val_data, test_data = data_split(sentinel_train_data, sentinel_test_data, dataset)
        train_loader, val_loader, test_loader = data_loaders(train_data, val_data, test_data, batch_size)

    return(train_loader, val_loader, test_loader)