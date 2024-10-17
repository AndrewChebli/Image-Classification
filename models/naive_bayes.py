import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import resnet18_model

#load the features and labels
def load_labels_features():
    train_features = torch.load('data/extracted_data/features.pt')
    train_labels = torch.load('data/extracted_data/labels.pt')

