import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import resnet18_model
import math

#load the features and labels
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

def calculate_probability_of_labels(labels):
    label_counts={}
    # keep track of the occurrences of each label 
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # calculate the probabilities
    total_labels = len(labels)
    probabilities={label: count/ total_labels for label, count in label_counts.items()}
    return probabilities


def calculate_parameters():
    features, labels = load_train_labels_features()
    mean_std={}
    label_feature_map = {}

    # Group features and their labels
    for feature_vector, label in zip(features, labels):
        if label not in label_feature_map:
            label_feature_map[label] = []
        label_feature_map[label].append(feature_vector)
    
    # Calculate the mean "mu" and standard deviation "sigma" 
    for label, feature_vectors in label_feature_map.items():
        feature_vectors = np.array(feature_vectors)
        std_dev = np.std(feature_vectors, axis=0)
        mean = np.mean(feature_vectors, axis=0)
        mean_std[label] = (mean, std_dev)
    return mean_std

def predict(feature_vector, label_probabilities, mean_std):
    max_prob = -1
    best_label = None
    for label in label_probabilities:  # Loop over each class
        prob = label_probabilities[label]  # Start with the prior probability of the class
        
        # Retrieve mean and std_dev for this class
        mean, std_dev = mean_std[label]
        
        # Calculate the likelihood of this feature_vector under the current class
        for i, feature in enumerate(feature_vector):
            # Calculate the likelihood for feature i given the class
            prob *= (1 / math.sqrt(2 * math.pi * std_dev[i] ** 2)) * math.exp(-((feature - mean[i]) ** 2) / (2 * std_dev[i] ** 2))

        # Check if this class has the highest probability so far
        if prob > max_prob:
            max_prob = prob
            best_label = label
    
    return best_label


def evaluate_model(mean_std, priors):
    # Load the test data
    test_features, test_labels = load_test_labels_features()
    
    correct_predictions = 0
    
    # Loop through each test sample
    for feature_vector, true_label in zip(test_features, test_labels):
        # Predict the label using the trained model parameters
        predicted_label = predict(feature_vector, priors, mean_std)
        print(f"predicted label: {predicted_label} vs true one was {true_label}")
        
        # Compare prediction with the true label
        if predicted_label == true_label:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / len(test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    

if __name__ == "__main__":
    # Load training data
    train_features, train_labels = load_train_labels_features()
    
    # Calculate priors (class probabilities)
    priors = calculate_probability_of_labels(train_labels)
    
    # Calculate mean and standard deviation for each class
    mean_std = calculate_parameters()
    
    # Evaluate the model on test data
    evaluate_model(mean_std, priors)

