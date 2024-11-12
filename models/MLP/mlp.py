import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# Load data


def load_data(file_path):
    data = torch.load(file_path)
    features = data['features']
    labels = data['labels']
    return features, labels

# Train the MLP


def train_model(model, train_features, train_labels, criterion, optimizer, num_epochs=10):
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


def evaluate_model(model, test_features, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    return accuracy * 100


# Main function to set up data, model, and training
if __name__ == "__main__":
    # Load training and testing data
    train_features, train_labels = load_data(
        'data/extracted_data/train_data.pt')
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features, train_labels = train_features.to(
        device), train_labels.to(device)
    test_features, test_labels = test_features.to(
        device), test_labels.to(device)

    # Define model, loss, and optimizer
    model = MLP(input_size=50, hidden_size=512, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    print("Training the model...")
    train_model(model, train_features, train_labels,
                criterion, optimizer, num_epochs=20)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = evaluate_model(model, test_features, test_labels)
    print(f'Accuracy of the MLP model: {accuracy:.2f}%')

    # Experiment with changing depth and hidden layer sizes
    # Modify the MLP class to add/remove layers or change hidden_size
