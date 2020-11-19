import os
import random
from PIL import Image

from torch.utils import data

import torchvision
import torchvision.transforms as transforms


def create_dataloader(dataset='cifar10', batch_size=64, num_workers=1):
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)

        trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset == 'summer2winter':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        trainset = Summer2WinterDataset(train=True, transform=transform)
        testset = Summer2WinterDataset(train=False, transform=transform)

        trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


class Summer2WinterDataset(data.Dataset):
    def __init__(self, train: bool=True, transform=None):
        self.transform = transform
        dataset_dir = './data/summer2winter_yosemite/'

        # Implement the dataset for unpaired image-to-image translation.
        # Check the dataset directory and implement the proper dataset.
        # This dataset have to load the train or test files depending on the 'train' option.

        ### YOUR CODE HERE (~ 10 lines) #TODO usefull functions: sorted(), os.walk() os.listdir()
        phase = 'train' if train else 'test'
        self.dirA = os.path.join(dataset_dir, phase + 'A')
        self.dirB = os.path.join(dataset_dir, phase + 'B')
        assert os.path.isdir(self.dirA), 'please check %s is unknown dir' % self.dirA
        assert os.path.isdir(self.dirB), 'please check %s is unknown dir' % self.dirB

        image_A_dir = []
        for root_dir, _, file_names in sorted(os.walk(self.dirA)):
            for file_name in file_names:
                image_dir = os.path.join(root_dir, file_name)
                image_A_dir.append(image_dir)
        self.image_list_A = sorted(image_A_dir)

        image_B_dir = []
        for root_dir, _, file_names in sorted(os.walk(self.dirB)):
            for file_name in file_names:
                image_dir = os.path.join(root_dir, file_name)
                image_B_dir.append(image_dir)
        self.image_list_B = sorted(image_B_dir)

        self.A_size = len(self.image_list_A)
        self.B_size = len(self.image_list_B)
        ### END YOUR CODE

    def __getitem__(self, index):

        # The number of images in domain A and domain B are different.
        # You have to sample the index to load data from different pairs.

        ### YOUR CODE HERE (~ 2 lines)
        image_A = Image.open(self.image_list_A[index % self.A_size])
        image_B = Image.open(self.image_list_B[random.randint(0, self.B_size - 1)])
        ### END YOUR CODE

        return self.transform(image_A), self.transform(image_B)

    def __len__(self):
        return len(self.image_list_A)


class FolderDataset(data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_list = os.listdir(folder)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder, self.image_list[index]))
        return self.transform(image)

    def __len__(self):
        return len(self.image_list)
