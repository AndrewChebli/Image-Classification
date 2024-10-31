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
    _, labels = load_train_labels_features()
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


def calculate_likelihoods():
    features, labels = load_train_labels_features()
    likelihoods={}
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
        print(f"{mean_std} \n")
    
    # Calculate the likelihood for each image(feature_vector)
    for label, (mean, std_dev) in mean_std.items():
        likelihoods[label] = []
        for feature_vector in features:
            likelihood = 1
            for i, feature in enumerate(feature_vector):
                likelihood *= (1/math.sqrt(2*math.pi*std_dev[i]**2)) * math.exp(-((feature - mean[i])**2) / (2*std_dev[i]**2))
            likelihoods[label].append(likelihood)
    

if __name__ == "__main__":
    # Your main function code here
    _, labels = load_train_labels_features()
    calculate_likelihoods()

