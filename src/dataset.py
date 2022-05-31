import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
import os
from torch.utils.data.dataset import Subset

class Dataset(object):
    def __init__(self, dataset_name):
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        if dataset_name == 'mnist' or 'MNIST':
            download_root = './MNIST_DATASET'
            self.train_dataset = datasets.MNIST(root=download_root, train=True, transform=transform, download=True)
            self.test_dataset = datasets.MNIST(root=download_root, train=True, transform=transform, download=True)

        elif dataset_name == 'cifar10' or 'CIFAR-10' or 'CIFAR10':
            download_root = './CIFAR10_DATASET'
            self.train_dataset = datasets.CIFAR10(root=download_root, train=True, transform=transform, download=True)
            self.test_dataset = datasets.CIFAR10(root=download_root, train=True, transform=transform, download=True)

    def split(self, num_clients):
        """split dataset for number of clients"""
        split_dataset = []
        length = len(self.train_dataset)
        subset = length // num_clients
        for i in range(num_clients):
            indices = range(subset*i, subset*(i+1))
            split_dataset.append(Subset(self.train_dataset, indices))

        return split_dataset, self.test_dataset


