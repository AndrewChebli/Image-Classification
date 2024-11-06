import torch
from models.Naive_Bayes.naive_bayes import NaiveBayesModel
from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel

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

def Naive_Bayes():
    custom_NaiveBayes_model = NaiveBayesModel()
    scikit_NaiveBayes_model = ScikitNaiveBayesModel()

    # Load train and test data
    train_features, train_labels = load_train_labels_features()
    test_features, test_labels = load_test_labels_features()

    # Train and evaluate custom Naive Bayes model
    priors = custom_NaiveBayes_model.calculate_probability_of_labels(train_labels)
    mean_std = custom_NaiveBayes_model.calculate_parameters(train_features, train_labels)
    accuracy = custom_NaiveBayes_model.evaluate_model(test_features, test_labels, mean_std, priors)
    print(f"custom Naive Bayes Accuracy: {accuracy:.2f}%")

    # Train and evaluate Scikit-Learn GaussianNB model
    scikit_NaiveBayes_model.train(train_features, train_labels)
    y_prediction = scikit_NaiveBayes_model.predict(test_features)
    scikit_accuracy = scikit_NaiveBayes_model.evaluate_model(y_prediction, test_labels)
    print(f"Scikit-Learn GaussianNB Accuracy: {scikit_accuracy:.2f}%")

if __name__ == "__main__":
    Naive_Bayes()
