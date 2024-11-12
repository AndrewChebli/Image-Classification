import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.MLP.mlp import MLP, load_data
from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel
from models.Decision_Tree.decision_tree import DecisionTreeClassifier
from models.Decision_Tree.decision_tree_scikit import DecisionTreeModelSklearn
from models.Naive_Bayes.naive_bayes import NaiveBayesModel


def evaluate_naives_bayes_model():
    loaded_model = NaiveBayesModel.load_model('./output/naive_bayes_model.pkl')
    test_features, test_labels = loaded_model.load_test_labels_features()
    accuracy = loaded_model.evaluate_model(test_features, test_labels, loaded_model.mean_std, loaded_model.priors)
    print(f"Naive Bayes Accuracy: {accuracy:.2f}%")


def evaluate_naives_bayes_scikit_model():
    loaded_model = ScikitNaiveBayesModel.load_model('./output/scikit_naive_bayes_model.pkl')
    test_features, test_labels = loaded_model.load_test_labels_features()
    y_prediction_loaded = loaded_model.predict(test_features)
    scikit_accuracy_loaded = loaded_model.evaluate_model(y_prediction_loaded, test_labels)
    print(f"Naive Bayes Scikit Accuracy: {scikit_accuracy_loaded:.2f}%")

def evaluate_decision_tree_model():
    model = DecisionTreeClassifier.load_model("./output/decision_tree_model.pkl")
    test_features, test_labels = model.load_test_labels_features()
    accuracy = model.evaluate_model(test_features, test_labels)
    print(f"Decision Tree Accuracy: {accuracy:.2f}%")



def evaluate_decision_tree_sklearn_model():
    model = DecisionTreeModelSklearn.load_model("./output/decision_tree_sklearn_model.pkl")
    test_features, test_labels = model.load_test_labels_features()
    accuracy = model.evaluate_model(test_features, test_labels)
    
    print(f"Scikit learn Decision Tree Accuracy: {accuracy:.2f}%")

def evaluate_mlp_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = MLP.load_model('./output/mlp_model.pth', input_size=50, hidden_size=512, num_classes=10).to(device)

    test_features, test_labels = load_data('data/extracted_data/test_data.pt')
    test_features, test_labels = test_features.to(device), test_labels.to(device)
   
    loaded_model_accuracy = loaded_model.evaluate_model(loaded_model, test_features, test_labels)
    print(f'MLP Accuracy: {loaded_model_accuracy:.2f}%')


if __name__ == "__main__":
    print("--------------------")
    print("NAIVE BAYES")
    evaluate_naives_bayes_model()

    print("--------------------")
    print("NAIVE BAYES SCIKIT")
    evaluate_naives_bayes_scikit_model()


    print("--------------------")
    print("DECISION TREE")
    evaluate_decision_tree_model()


    print("--------------------")
    print("DECISION TREE - SCIKIT")
    evaluate_decision_tree_sklearn_model()

    print("--------------------")
    print("MLP")
    evaluate_mlp_model()




