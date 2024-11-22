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
        data = torch.load('data/extracted_data/train_data.pt', weights_only=True)
        train_features = data['features'].numpy()
        train_labels = data['labels'].numpy()
        return train_features, train_labels

    def load_test_labels_features(self):
        data = torch.load('data/extracted_data/test_data.pt', weights_only=True)
        test_features = data['features'].numpy()
        test_labels = data['labels'].numpy()
        return test_features, test_labels

    def fit(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)

    def evaluate_model(self, test_features, test_labels):
        accuracy = self.model.score(test_features, test_labels) * 100
        return accuracy
    
    def predict(self, test_features):
        return self.model.predict(test_features)

    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model


def set_random_seeds(seed=88):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    set_random_seeds()
    max_depths = [10, 20, 50]
    for max_depth in max_depths:
        sklearn_tree = DecisionTreeModelSklearn(max_depth=max_depth)
        train_features, train_labels = sklearn_tree.load_train_labels_features()
        test_features, test_labels = sklearn_tree.load_test_labels_features()
        sklearn_tree.fit(train_features, train_labels)
        accuracy = sklearn_tree.evaluate_model(test_features, test_labels)
        sklearn_tree.save_model(f'./output/decision_tree_sklearn_model_{max_depth}.pkl')
        print(f"Scikit-learn Decision Tree with max_depth={max_depth} Accuracy: {accuracy:.2f}%")
    