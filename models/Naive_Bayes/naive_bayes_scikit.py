from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import numpy

class ScikitNaiveBayesModel():
    def __init__(self):
        self.model = GaussianNB()

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

    def train(self, train_features, train_labels):
        # train using the Scikit-Learn's GaussianNB
        self.model.fit(train_features, train_labels)

    def predict(self, test_features):
        # predict using the trained model
        return self.model.predict(test_features)

    def evaluate_model(self, y_prediction, test_labels):
        return accuracy_score(y_prediction, test_labels) * 100

if __name__ == "__main__":

    skicit_model = ScikitNaiveBayesModel()

    # Load the saved training data and testing data
    train_features, train_labels = skicit_model.load_train_labels_features()
    test_features, test_labels = skicit_model.load_test_labels_features()

    # Train the model with the data we have
    skicit_model.train(train_features, train_labels)

    # Predict on the test data
    y_prediction = skicit_model.predict(test_features)

    # Evaluate accuracy
    scikit_accuracy = skicit_model.evaluate_model(y_prediction, test_labels)
    print(f"Scikit-Learn GaussianNB Accuracy: {scikit_accuracy:.2f}%")
