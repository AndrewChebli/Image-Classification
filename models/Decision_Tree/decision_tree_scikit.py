import random
import torch
import pickle
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree


class DecisionTreeModelSklearn:
    def __init__(self, max_depth=50):
        self.model = SklearnDecisionTree(criterion="gini", max_depth=max_depth)

    def load_train_labels_features(self):
        data = torch.load(
            'data/extracted_data/train_data.pt', weights_only=True)
        train_features = data['features'].numpy()
        train_labels = data['labels'].numpy()
        return train_features, train_labels

    def load_test_labels_features(self):
        data = torch.load('data/extracted_data/test_data.pt',
                          weights_only=True)
        test_features = data['features'].numpy()
        test_labels = data['labels'].numpy()
        return test_features, test_labels

    def fit(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)

    def evaluate_model(self, test_features, test_labels):
        accuracy = self.model.score(test_features, test_labels) * 100
        return accuracy

    def predict(self, features):
        return self.model.predict(features)
    
    def save_model(self, filename):
        # Ensure the output folder exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the model to the specified file, overwriting it if it exists
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

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
    sklearn_tree = DecisionTreeModelSklearn(max_depth=50)

    # Load data
    train_features, train_labels = sklearn_tree.load_train_labels_features()
    test_features, test_labels = sklearn_tree.load_test_labels_features()

    # Train model
    sklearn_tree.fit(train_features, train_labels)

    # Evaluate model
    accuracy = sklearn_tree.evaluate_model(test_features, test_labels)
    print(f"Scikit-learn Decision Tree Accuracy: {accuracy:.2f}%")

    # Save model to file
    sklearn_tree.save_model('./output/decision_tree_sklearn_model.pkl')
