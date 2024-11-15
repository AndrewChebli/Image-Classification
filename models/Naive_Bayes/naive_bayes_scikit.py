from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import torch
import numpy,os,random,pickle

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

    def save_model(self, filename):
        # Ensure the output folder exists, if not, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the model to a file
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        # Load the model from the file
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        
        # Create a new ScikitNaiveBayesModel object and assign the loaded model
        sk_model = ScikitNaiveBayesModel()
        sk_model.model = model
        print(f"Model loaded from {filename}")
        return sk_model
def set_random_seeds(seed):
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #gpu seed
    
    # Set the random seed for NumPy
    numpy.random.seed(seed)
    
    # Set the random seed
    random.seed(seed)
    
    # Set the random seed for Scikit-learn
    from sklearn.utils import check_random_state
    check_random_state(seed)

if __name__ == "__main__":
    set_random_seeds(seed=88)
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


    # Save the trained model
    skicit_model.save_model('./output/scikit_naive_bayes_model.pkl')
