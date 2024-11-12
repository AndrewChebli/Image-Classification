import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Decision_Tree.decision_tree import DecisionTreeClassifier
from models.Decision_Tree.decision_tree_scikit import DecisionTreeModelSklearn


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



# This would be the starting point when running the evaluation script
if __name__ == "__main__":
    print("--------------------")
    print("DECISION TREE")
    evaluate_decision_tree_model()


    print("--------------------")
    print("DECISION TREE - SCIKIT")
    evaluate_decision_tree_sklearn_model()

