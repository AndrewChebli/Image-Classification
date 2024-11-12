import torch
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


if __name__ == "__main__":
    sklearn_tree = DecisionTreeModelSklearn(max_depth=50)

    # Load data
    train_features, train_labels = sklearn_tree.load_train_labels_features()
    test_features, test_labels = sklearn_tree.load_test_labels_features()

    # Train model
    sklearn_tree.fit(train_features, train_labels)

    # Evaluate model
    accuracy = sklearn_tree.evaluate_model(test_features, test_labels)
    print(f"Scikit-learn Decision Tree Accuracy: {accuracy:.2f}%")
