from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import numpy


def load_train_labels_features():
    data = torch.load('data/extracted_data/train_data.pt', weights_only=True)
    train_features = data['features'].numpy()
    train_labels = data['labels'].numpy()

    return train_features, train_labels

def load_test_labels_features():
    data = torch.load('data/extracted_data/test_data.pt', weights_only=True)
    test_features = data['features'].numpy()
    test_labels = data['labels'].numpy()

    return test_features, test_labels

# Load train and test data
train_features, train_labels = load_train_labels_features()
test_features, test_labels = load_test_labels_features()

# Train and test using the Scikit-Learn's GaussianNB
gnb = GaussianNB()
y_prediction = gnb.fit(train_features, train_labels).predict(test_features)
scikit_accuracy = accuracy_score(y_prediction, test_labels)
print(f"Scikit-Learn GaussianNB Accuracy: {scikit_accuracy * 100:.2f}%")
