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
    def __init__(self,input_size, output_size, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            #layer 1:
            nn.Conv2d(3, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            #layer 2
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding =padding),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            #layer 3
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            #layer 4
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 
            
            #layer 5
            nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True), 
            
            #layer 6
            nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),   
            
            #layer 7
            nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  
            
            #layer 8
            nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding),
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
    def load_model(filename, input_size=3 * 32 * 32, kernel_size=3, num_classes=10):
        # Initialize a new instance of the MLP model
        model = CNN(input_size=input_size, kernel_size=kernel_size, output_size=num_classes)
        
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

    def predict(self, test_loader, device):
        self.eval()  # Set the model to evaluation mode
        all_predictions = []
        with torch.no_grad():  # Disable gradient computations
            for instances, _ in test_loader:
                instances = instances.to(device)
                output = self(instances)  # Forward pass
                predictions = torch.max(output, 1)[1]  # Get the predicted class
                all_predictions.extend(predictions.cpu().numpy())
        return all_predictions

def remove_last_layer(model):
    layers = list(model.features.children())
    for i in range(len(layers) - 1, -1, -1):
        if isinstance(layers[i], nn.Conv2d):
            break
    model.features = nn.Sequential(*layers[:i])
    
    # Calculate the new input size for the first fully connected layer
    num_maxpool_layers = sum(1 for layer in model.features if isinstance(layer, nn.MaxPool2d))
    new_input_size = 512 * (32 // (2 ** num_maxpool_layers)) * (32 // (2 ** num_maxpool_layers))
    print(f"new input size is : {new_input_size}")
    
    model.classifier = nn.Sequential(
        nn.Linear(new_input_size, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
    return model

def add_extra_conv_layer(model, kernel_size=3):
    padding = kernel_size // 2
    # Add a convolutional layer with the same input/output channels as the last Conv2d in the model
    new_layer = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=padding)
    model.features = nn.Sequential(
        *list(model.features.children()),  # Existing layers
        new_layer,                         # New layer
        nn.BatchNorm2d(512),               # Add BatchNorm2d for consistency
        nn.ReLU(True)                      # Activation function
    )
    print(f"Added an extra convolutional layer with kernel size {kernel_size}")
    return model

if __name__ == '__main__':

    batch_size = 32
    test_batch_size = 32
    input_size = 3 * 32 * 32  # 3 channels, 32x32 image size
    # hidden_size = 50  # Number of hidden units
    output_size = 10  # Number of output classes (CIFAR-10 has 10 classes)
    num_epochs = 50


    train_dataset, test_dataset = cifar_loader(batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    # model = CNN(input_size, output_size)
    # model.to(device)

    
    action = input("Enter 'train', 'test': ").strip().lower()

    if action in ["train", "test"]:
        modify_action = input("Do you want to 'remove_last_layer' or 'add_extra_layer'  or 'adjust_kernel_size'? (leave blank for none): ").strip().lower()
        if modify_action == "remove_last_layer":
            model = CNN(input_size, output_size).to(device)
            model = remove_last_layer(model)
        elif modify_action == "adjust_kernel_size":
            new_kernel_size = int(input("Enter new kernel size: "))
            model = CNN(input_size, output_size, new_kernel_size).to(device)
        elif modify_action == "add_extra_layer":
            model = CNN(input_size, output_size).to(device)
            model = add_extra_conv_layer(model)

        if action == "train":
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
            
            # Split the training dataset into 90% training and 10% evaluation
            train_size = int(0.9 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_subset, eval_subset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
            
            #to get the data in batches
            train_loader = td.DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            eval_loader = td.DataLoader(eval_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
            model.train_model(train_loader, eval_loader, device, optimizer, criterion, num_epochs, './output/cnn_model.pth')
        elif action == "test":
            
            #to get the data in batches
            test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

            # Load the state dict
            state_dict = torch.load('./output/cnn_model.pth')

            # Create a new state dict with matching keys
            new_state_dict = model.state_dict()
            for key in state_dict:
                if key in new_state_dict and state_dict[key].shape == new_state_dict[key].shape:
                    new_state_dict[key] = state_dict[key]

            # Load the new state dict into the model
            model.load_state_dict(new_state_dict, strict=False)

            model.to(device)
            model.test_model(test_loader, device, criterion)