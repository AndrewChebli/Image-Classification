import torch
import os
import pickle
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

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

    def gini_index(self, groups, classes):
        n_instances = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            # Use NumPy for efficient Gini calculation
            counts = np.bincount([row[-1]
                                 for row in group], minlength=len(classes))
            probabilities = counts / size
            gini += (1.0 - np.sum(probabilities ** 2)) * (size / n_instances)
        return gini

    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            # Sample a limited number of unique values for the feature to reduce complexity
            unique_values = np.unique([row[index] for row in dataset])
            sampled_values = np.random.choice(
                unique_values, size=min(10, len(unique_values)), replace=False)
            for value in sampled_values:
                groups = self.test_split(index, value, dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, value, gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(
                left), self.to_terminal(right)
            return
        if len(left) <= 1:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)
        if len(right) <= 1:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    def build_tree(self, train):
        root = self.get_split(train)
        self.split(root, 1)
        return root

    def fit(self, train_features, train_labels):
        train_data = np.column_stack((train_features, train_labels))
        self.tree = self.build_tree(train_data)

    def predict_row(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']

    def predict(self, test_features):
        predictions = [self.predict_row(self.tree, row)
                       for row in test_features]
        return predictions

    def evaluate_model(self, test_features, test_labels):

        predictions = self.predict(test_features)
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        return accuracy * 100

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


if __name__ == "__main__":
    tree_classifier = DecisionTreeClassifier(max_depth=12)

    # Load data
    print("loading")
    train_features, train_labels = tree_classifier.load_train_labels_features()
    test_features, test_labels = tree_classifier.load_test_labels_features()

    # Train model
    print("train model - fitting")
    tree_classifier.fit(train_features, train_labels)

    # Evaluate model
    print("evaluation")
    accuracy = tree_classifier.evaluate_model(test_features, test_labels)
    print(f"Basic Python & NumPy Decision Tree Accuracy: {accuracy:.2f}%")

    # Save model to file
    tree_classifier.save_model('./output/decision_tree_model.pkl')
