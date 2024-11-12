import os
import sys
import tkinter
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.MLP.mlp import MLP, load_data
from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel
from models.Decision_Tree.decision_tree import DecisionTreeClassifier
from models.Decision_Tree.decision_tree_scikit import DecisionTreeModelSklearn
from models.Naive_Bayes.naive_bayes import NaiveBayesModel


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1


def evaluate_model(model, test_features, test_labels, model_name):
    predictions = model.predict(test_features)
    accuracy, precision, recall, f1 = calculate_metrics(test_labels, predictions)
    plot_confusion_matrix(test_labels, predictions, model_name)
    return accuracy, precision, recall, f1


def evaluate_all_models():
    results = []

    # Naive Bayes
    naive_bayes_model = NaiveBayesModel.load_model('./output/naive_bayes_model.pkl')
    test_features, test_labels = naive_bayes_model.load_test_labels_features()
    accuracy, precision, recall, f1 = evaluate_model(naive_bayes_model, test_features, test_labels, "Naive Bayes")
    results.append(["Naive Bayes", accuracy, precision, recall, f1])

    # Scikit Naive Bayes
    naive_bayes_scikit_model = ScikitNaiveBayesModel.load_model('./output/scikit_naive_bayes_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(naive_bayes_scikit_model, test_features, test_labels, "Naive Bayes Scikit")
    results.append(["Naive Bayes Scikit", accuracy, precision, recall, f1])

    # Decision Tree
    decision_tree_model = DecisionTreeClassifier.load_model('./output/decision_tree_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(decision_tree_model, test_features, test_labels, "Decision Tree")
    results.append(["Decision Tree", accuracy, precision, recall, f1])

    # Scikit Decision Tree
    decision_tree_sklearn_model = DecisionTreeModelSklearn.load_model('./output/decision_tree_sklearn_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(decision_tree_sklearn_model, test_features, test_labels, "Decision Tree Scikit")
    results.append(["Decision Tree Scikit", accuracy, precision, recall, f1])

    # MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = MLP.load_model('./output/mlp_model.pth', input_size=50, hidden_size=512, num_classes=10).to(device)
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')
    test_features, test_labels = test_features.to(device), test_labels.to(device)
    accuracy, precision, recall, f1 = evaluate_model(mlp_model, test_features, test_labels, "MLP")
    results.append(["MLP", accuracy, precision, recall, f1])

    # Create summary table
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    print("\nSummary of Model Performance:")
    print(results_df)


if __name__ == "__main__":
    evaluate_all_models()
