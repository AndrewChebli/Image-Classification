import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import resnet18_model

#load the features and labels
def load_train_labels_features():
    train_features = torch.load('data/extracted_data/features.pt', weights_only=True)
    train_labels = torch.load('data/extracted_data/labels.pt', weights_only=True)

    return train_features.numpy(), train_labels.numpy()

def load_test_labels_features():
    train_features = torch.load('data/extracted_data/features_test.pt', weights_only=True)
    train_labels = torch.load('data/extracted_data/labels_test.pt', weights_only=True)

    return train_features.numpy(), train_labels.numpy()

