from torch import optim
import torch.backends.mps as mps
import torch.nn as nn
import torch.utils.data as td
import torchvision.datasets as datasets
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image

# Function to load CIFAR10 dataset

def cifar_loader(batch_size, shuffle_test=False):
    # Normalization values for CIFAR10 dataset
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])

    # Loading training dataset with data augmentation techniques
    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
    transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize
    ]))

    test_dataset = datasets.CIFAR10('./data', train=False,
    transform=transforms.Compose([
    transforms.ToTensor(),
    normalize
    ]))
    
    def limit_per_class(dataset, num_per_class):
        class_counts = {i: 0 for i in range(10)}
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if class_counts[label] < num_per_class:
                indices.append(idx)
                class_counts[label] += 1
            if all(count >= num_per_class for count in class_counts.values()):
                break
        return td.Subset(dataset, indices)

    # Limit the number of images per class
    train_dataset = limit_per_class(train_dataset, 500)
    test_dataset = limit_per_class(test_dataset, 100)

    # Creating data loaders for training and testing
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=shuffle_test, pin_memory=True)
    return train_dataset, test_loader


class CNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        self.features = nn.Sequential(
            #layer 1:
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            #layer 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            #layer 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            #layer 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            
            #layer 5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True), 
            
            #layer 6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),   
            
            #layer 7
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  
            
            #layer 8
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)        
        )
        
        self.classifier= nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)

        )
       

    def forward(self, x):
        # pass through the 8 convolutional layers
        x = self.features(x)
        # flatten the output
        x = x.view(x.size(0), -1)
        #pass through the fully connected layers, classifier
        x = self.classifier(x)
        return x
