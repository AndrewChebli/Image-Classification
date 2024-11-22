import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_sizes=[512, 512], num_classes=10):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def predict(self, test_features):
        self.eval() 
        with torch.no_grad():
            outputs = self(test_features) 
            _, predicted = torch.max(outputs, 1)  
        return predicted
    

    @staticmethod
    def load_model(filename, input_size=50, hidden_sizes=[512, 512], num_classes=10):
        model = MLP(input_size, hidden_sizes, num_classes)
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
        return model

    @staticmethod
    def train_model(model, train_features, train_labels, criterion, optimizer, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (epoch + 1) % 2 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    @staticmethod
    def evaluate_model(model, test_features, test_labels):
        model.eval()
        with torch.no_grad():
            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_labels).sum().item() / len(test_labels)
        return accuracy * 100


# Load data
def load_data(file_path):
    data = torch.load(file_path)
    features = data['features']
    labels = data['labels']
    return features, labels

def set_random_seeds(seed=88):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_random_seeds()

    # Load data
    train_features, train_labels = load_data('data/extracted_data/train_data.pt')
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_features, train_labels = train_features.to(device), train_labels.to(device)
    test_features, test_labels = test_features.to(device), test_labels.to(device)

    # Define different configurations for experiments
    experiments = [
        {"hidden_sizes": [512, 512], "description": "2-layer"},
        {"hidden_sizes": [512], "description": "1 hidden layer"},
        {"hidden_sizes": [512, 512, 512], "description": "3-layer MLP"},
        {"hidden_sizes": [256, 256], "description": "Smaller hidden layers"},
        {"hidden_sizes": [1024, 1024], "description": "Larger hidden layers"}
    ]

    criterion = nn.CrossEntropyLoss()
    for experiment in experiments:
        print(f"\nRunning Experiment: {experiment['description']}")
        mlp = MLP(input_size=50, hidden_sizes=experiment['hidden_sizes'], num_classes=10).to(device)
        optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

        print("Training the model...")
        MLP.train_model(mlp, train_features, train_labels, criterion, optimizer, num_epochs=20)

        print("Evaluating the model...")
        accuracy = MLP.evaluate_model(mlp, test_features, test_labels)
        print(f"Accuracy of the MLP model ({experiment['description']}): {accuracy:.2f}%")

        # Save model
        filename = f'./output/mlp_{experiment["description"].replace(" ", "_").lower()}.pth'
        mlp.save_model(filename)

if __name__ == "__main__":
    main()
