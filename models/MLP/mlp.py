import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    def save_model(self, filename):
        # Ensure the output folder exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the model parameters to the specified file
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename, input_size=50, hidden_size=512, num_classes=10):
        # Initialize a new instance of the MLP model
        model = MLP(input_size, hidden_size, num_classes)
        
        # Load the model parameters from the file
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
        return model



    # Train the MLP
    def train_model(self,model, train_features, train_labels, criterion, optimizer, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the MLP
    def evaluate_model(self,model, test_features, test_labels):
        model.eval()
        with torch.no_grad():
            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_labels).sum().item() / len(test_labels)
        return accuracy * 100
    
    def predict(self, features):
        self.eval()
        with torch.no_grad():
            outputs = self(features)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    # Load data
def load_data(file_path):
    data = torch.load(file_path)
    features = data['features']
    labels = data['labels']
    return features, labels

def set_random_seeds(seed):
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #gpu seed
    
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed
    random.seed(seed)
    
    # Set the random seed for Scikit-learn
    from sklearn.utils import check_random_state
    check_random_state(seed)

if __name__ == "__main__":
    set_random_seeds(seed=88)
    # Load training and testing data
    train_features, train_labels = load_data('data/extracted_data/train_data.pt')
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    test_features, test_labels = test_features.to(device), test_labels.to(device)

    # Define model, loss, and optimizer
    mlp = MLP(input_size=50, hidden_size=512, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    print("Training the model...")
    mlp.train_model(mlp, train_features, train_labels, criterion, optimizer, num_epochs=20)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = mlp.evaluate_model(mlp, test_features, test_labels)
    print(f'Accuracy of the MLP model: {accuracy:.2f}%')

    # Save the trained model
    mlp.save_model('./output/mlp_model.pth')

   