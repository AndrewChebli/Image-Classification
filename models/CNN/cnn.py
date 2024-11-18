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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    return train_dataset, test_dataset


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
    
    @staticmethod
    def load_model(filename, input_size=3 * 32 * 32, hidden_size=10, num_classes=10):
        # Initialize a new instance of the MLP model
        model = CNN(input_size, hidden_size, num_classes)
        
        # Load the model parameters from the file
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
        return model
    
    def train_model(self, train_loader, eval_loader, device, optimizer, criterion, num_epochs, save_path):
        best_acc = 0
        best_model = None

        for epoch in range(num_epochs):
            self.train() # set the model to train mode
            running_loss = 0
            for instances, labels in train_loader:
                optimizer.zero_grad()
                output = self(instances.to(device))
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.eval() # set the model to evaluate mode
            correct = 0
            total = 0
            with torch.no_grad():
                for instances, labels in eval_loader:
                    output = self(instances.to(device))
                    predictions = torch.max(output, 1)[1]
                    total += labels.size(0)
                    correct += (predictions == labels.to(device)).sum().item()

            acc = correct / total * 100
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_model = self.state_dict()

        # Save the best model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model, save_path)
        print(f"Best model saved with accuracy {best_acc:.2f}%")
    
    def test_model(self, test_loader, device, criterion):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():  # Disable gradient computations
            for instances, labels in test_loader:
                instances, labels = instances.to(device), labels.to(device)
                output = self(instances)  # Forward pass
                loss = criterion(output, labels)  # Compute loss
                test_loss += loss.item()

                predictions = torch.max(output, 1)[1]  # Get the predicted class
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # Calculate and print test accuracy and loss
        test_loss /= len(test_loader)
        test_accuracy = correct / total * 100
        print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

if __name__ == '__main__':

    batch_size = 32
    test_batch_size = 32
    input_size = 3 * 32 * 32  # 3 channels, 32x32 image size
    # hidden_size = 50  # Number of hidden units
    output_size = 10  # Number of output classes (CIFAR-10 has 10 classes)
    num_epochs = 50


    train_dataset, test_dataset = cifar_loader(batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(input_size, 10, output_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    
    # Split the training dataset into 90% training and 10% evaluation
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    #to get the data in batches
    train_loader = td.DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    eval_loader = td.DataLoader(eval_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    action = input("Enter 'train' or 'test': ").strip().lower()

    if action == "train":
        model.train_model(train_loader, eval_loader, device, optimizer, criterion, num_epochs, './output/cnn_model.pth')
    elif action == "test":
        model.load_state_dict(torch.load('./output/cnn_model.pth'))
        model.test_model(test_loader, device, criterion)
