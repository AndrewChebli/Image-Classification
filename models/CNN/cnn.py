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

if __name__ == '__main__':

    batch_size = 32
    test_batch_size = 32
    input_size = 3 * 32 * 32  # 3 channels, 32x32 image size
    # hidden_size = 50  # Number of hidden units
    output_size = 10  # Number of output classes (CIFAR-10 has 10 classes)
    num_epochs = 50
    all_models=[]


    train_dataset, test_loader = cifar_loader(batch_size)
    _, test_loader = cifar_loader(test_batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(input_size, 10, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('path'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    BestACC=0
    
    # Split the training dataset into 90% training and 10% evaluation
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])


    
    for epoch in range(num_epochs):
        train_loader = td.DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        eval_loader = td.DataLoader(eval_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()

            output = model(instances.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            allsamps=0
            rightPred=0

            for instances, labels in eval_loader:
                output = model(instances.to(device))
                predictedClass=torch.max(output,1)
                allsamps+=output.size(0)
                rightPred+=(torch.max(output,1)[1]==labels.to(device)).sum()

            ACC=float(rightPred)/float(allsamps)
            print(f'Evaluation Accuracy is={ACC*100}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            if ACC>BestACC:
                BestACC=ACC
                all_models.append(model.state_dict())
        model.train()
    print(f'best accuracy for cnn is {BestACC*100}')
    torch.save(all_models[-1], './output/cnn_model.pth')

