import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms

def get_resnet():
    #load pre trained model resnet 18
    resnet18_model = resnet18(weights= ResNet18_Weights.IMAGENET1K_V1)

    #remove last layer to only use feature extraction from the model
    newmodel = nn.Sequential(*(list(resnet18_model.children())[:-1]))
    return newmodel

if __name__ == "__main__":
    resnet_18 = get_resnet()
    print (resnet_18)
