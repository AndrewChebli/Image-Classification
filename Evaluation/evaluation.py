import os
import sys
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.MLP.mlp import MLP, load_data
from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel
from models.Decision_Tree.decision_tree import DecisionTreeClassifier
from models.Decision_Tree.decision_tree_scikit import DecisionTreeModelSklearn
from models.Naive_Bayes.naive_bayes import NaiveBayesModel


def plot_confusion_matrix(y_true, y_pred, model_name, pdf):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    
    # Save the confusion matrix plot to the PDF
    pdf.savefig()
    plt.close()


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1


def evaluate_model(model, test_features, test_labels, model_name, pdf):
    predictions = model.predict(test_features)
    accuracy, precision, recall, f1 = calculate_metrics(test_labels, predictions)
    plot_confusion_matrix(test_labels, predictions, model_name, pdf)
    return accuracy, precision, recall, f1

def add_summary_to_pdf_report(results_df, pdf):
    # Format values for display in the PDF table (3 decimal places and percentages)
    results_df_display = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        results_df_display[col] = (results_df_display[col]).round(3)

    # Create a figure to render the table as an image for the PDF
    fig, ax = plt.subplots(figsize=(8, 2 + len(results_df_display) * 0.4))
    ax.axis('tight')
    ax.axis('off')

    # Create the table from the formatted DataFrame
    table = ax.table(cellText=results_df_display.values,
                     colLabels=results_df_display.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Style the table headers
    for j in range(len(results_df_display.columns)):
        header_cell = table[(0, j)]
        header_cell.set_text_props(weight='bold', color="white", backgroundcolor="gray")

    # Add the table to the PDF
    pdf.savefig(fig)
    plt.close(fig)
    print("Summary table added to PDF.")



def evaluate_all_models():
    results = []
    
    # Get current timestamp for the output filename
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Evaluation_Output"

    # Ensure the evaluation-output folder exists
    if not os.path.exists(f"evaluation-output/{current_timestamp}"):
        os.makedirs(f"evaluation-output/{current_timestamp}")

    # Prepare PDF for plots
    pdf_plots_filename = f"evaluation-output/{current_timestamp}/{output_file}_Confusion_Matrices_And_Summary.pdf"
    pdf = PdfPages(pdf_plots_filename)
    
    # Naive Bayes
    naive_bayes_model = NaiveBayesModel.load_model('./output/naive_bayes_model.pkl')
    test_features, test_labels = naive_bayes_model.load_test_labels_features()
    accuracy, precision, recall, f1 = evaluate_model(naive_bayes_model, test_features, test_labels, "Naive Bayes", pdf)
    results.append(["Naive Bayes", accuracy, precision, recall, f1])

    # Scikit Naive Bayes
    naive_bayes_scikit_model = ScikitNaiveBayesModel.load_model('./output/scikit_naive_bayes_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(naive_bayes_scikit_model, test_features, test_labels, "Naive Bayes Scikit", pdf)
    results.append(["Naive Bayes Scikit", accuracy, precision, recall, f1])

    # Decision Tree
    decision_tree_model = DecisionTreeClassifier.load_model('./output/decision_tree_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(decision_tree_model, test_features, test_labels, "Decision Tree", pdf)
    results.append(["Decision Tree", accuracy, precision, recall, f1])

    # Scikit Decision Tree
    decision_tree_sklearn_model = DecisionTreeModelSklearn.load_model('./output/decision_tree_sklearn_model.pkl')
    accuracy, precision, recall, f1 = evaluate_model(decision_tree_sklearn_model, test_features, test_labels, "Decision Tree Scikit", pdf)
    results.append(["Decision Tree Scikit", accuracy, precision, recall, f1])

    # MLP
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    mlp_model = MLP.load_model('./output/mlp_model.pth', input_size=50, hidden_size=512, num_classes=10).to(device)
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')
    test_features, test_labels = test_features.to(device), test_labels.to(device)
    accuracy, precision, recall, f1 = evaluate_model(mlp_model, test_features, test_labels, "MLP", pdf)
    results.append(["MLP", accuracy, precision, recall, f1])

    #CNN
    

    # Create summary table
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

      # Create summary table with 5 decimal places for CSV
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    results_df_csv = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        results_df_csv[col] = results_df_csv[col].round(5)

    # Save summary results to CSV with 5 decimal places
    results_filename = f"evaluation-output/{current_timestamp}/{output_file}_summary.csv"
    results_df_csv.to_csv(results_filename, index=False)

    
    
    # Create the final report PDF
    add_summary_to_pdf_report(results_df,pdf)

    # Finalize the PDF with the plots and the table
    pdf.close()

    print("\nSummary of Model Performance:")
    print(results_df)


if __name__ == "__main__":
    evaluate_all_models()
