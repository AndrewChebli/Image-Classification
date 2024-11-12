import torch
import os,sys,pickle
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import math

class NaiveBayesModel:
    
    #load the features and labels
    def load_train_labels_features(self):
        data = torch.load('data/extracted_data/train_data.pt', weights_only=True)
        train_features = data['features'].numpy()
        train_labels = data['labels'].numpy()

        return train_features, train_labels

    def load_test_labels_features(self):
        data = torch.load('data/extracted_data/test_data.pt', weights_only=True)
        test_features = data['features'].numpy()
        test_labels = data['labels'].numpy()

        return test_features, test_labels

    def calculate_probability_of_labels(self,labels):
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


    def calculate_parameters(self, training_features, training_labels):
        mean_std={}
        label_feature_map = {}

        # Group features and their labels
        for feature_vector, label in zip(training_features, training_labels):
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

   
    def predict(self, test_features):
        predictions = []
        for feature_vector in test_features:
            max_prob = -1
            best_label = None
            for label in self.priors:  # Loop over each class
                prob = self.priors[label]  # Start with the prior probability of the class

                # Retrieve mean and std_dev for this class
                mean, std_dev = self.mean_std[label]
                
                # Calculate the likelihood of this feature_vector under the current class
                for i, feature in enumerate(feature_vector):
                    # Calculate the likelihood for feature i given the class
                    prob *= (1 / math.sqrt(2 * math.pi * std_dev[i] ** 2)) * math.exp(-((feature - mean[i]) ** 2) / (2 * std_dev[i] ** 2))

                # Check if this class has the highest probability so far
                if prob > max_prob:
                    max_prob = prob
                    best_label = label
            
            predictions.append(best_label)
        
        return predictions

    def evaluate_model(self,test_features, test_labels, mean_std, priors):

        correct_predictions = 0
        
        # Loop through each test sample
        for feature_vector, true_label in zip(test_features, test_labels):
            # Predict the label using the trained model parameters
            predicted_label = self.predict(feature_vector, priors, mean_std)
                # print(f"predicted label: {predicted_label} vs true one was {true_label}")
            
            # Compare prediction with the true label
            if predicted_label == true_label:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / len(test_labels)
        return accuracy * 100
    

    def save_model(self, filename):
        # Ensure the output folder exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the model parameters (priors and mean_std) to a file
        with open(filename, 'wb') as f:
            pickle.dump({'priors': self.priors, 'mean_std': self.mean_std}, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        # Load the model parameters from the file
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new NaiveBayesModel object and populate it with the loaded parameters
        model = NaiveBayesModel()
        model.priors = model_data['priors']
        model.mean_std = model_data['mean_std']
        print(f"Model loaded from {filename}")
        return model

if __name__ == "__main__":
    naive_Bayes = NaiveBayesModel()

    # Load training data
    train_features, train_labels = naive_Bayes.load_train_labels_features()
    test_features, test_labels = naive_Bayes.load_test_labels_features()
    
    # Calculate priors (class probabilities)
    priors = naive_Bayes.calculate_probability_of_labels(train_labels)
    naive_Bayes.priors = priors
    # Calculate mean and standard deviation for each class
    mean_std = naive_Bayes.calculate_parameters(train_features, train_labels)
    naive_Bayes.mean_std = mean_std
    
    # Evaluate the model on test data
    accuracy = naive_Bayes.evaluate_model(test_features, test_labels, mean_std, priors)
    print(f"Naive Bayes Accuracy: {accuracy:.2f}%")

     # Save the trained model
    naive_Bayes.save_model('./output/naive_bayes_model.pkl')

