import os
import sys
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
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


def add_summary_to_pdf_report(results_df, output_file, pdf_plots_filename,pdf_filename):
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Ensure the directory exists for saving the report
    output_dir = f'evaluation-output/{current_timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_filename = f"{output_dir}/{output_file}_summary.pdf"
    document = SimpleDocTemplate(pdf_filename, pagesize=letter)
    
    # Create a list of elements to add to the document
    elements = []
    
    # Add Confusion Matrices Section Title
    plot_section_title = Paragraph("<b>Model Confusion Matrices:</b>", getSampleStyleSheet()['Heading2'])
    elements.append(plot_section_title)
    
    # Add Confusion Matrix Plots
    plot_description = Paragraph(f"Confusion matrices saved in the file: {pdf_plots_filename}", getSampleStyleSheet()['Normal'])
    elements.append(plot_description)
    
    # Embed the plot images in the report (using the PdfPages for now)
    try:
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        c.drawString(100, 750, f"Confusion matrix plots are embedded as {pdf_plots_filename}.")
        c.showPage()
        c.save()
    except Exception as e:
        print(f"Error embedding plots: {e}")
    
    # Add a page for summary table after confusion matrices
    elements.append(Paragraph("<b>Summary of Model Performance:</b>", getSampleStyleSheet()['Heading2']))
    
    # Prepare the data for the table (summary of results)
    table_data = [["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
    for row in results_df.values:
        table_data.append([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])
    
    # Create the summary table
    table = Table(table_data)
    
    # Add table styling
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    # Add the table to the elements
    elements.append(table)
    
    # Build and save the PDF
    document.build(elements)
    print(f"PDF report generated: {pdf_filename}")


def evaluate_all_models():
    results = []
    
    # Get current timestamp for the output filename
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"Evaluation_Output"

    # Ensure the evaluation-output folder exists
    if not os.path.exists("evaluation-output"):
        os.makedirs("evaluation-output")

    # Prepare PDF for plots
    pdf_plots_filename = f"evaluation-output/{output_file}_Confusion_Matrices.pdf"
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = MLP.load_model('./output/mlp_model.pth', input_size=50, hidden_size=512, num_classes=10).to(device)
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')
    test_features, test_labels = test_features.to(device), test_labels.to(device)
    accuracy, precision, recall, f1 = evaluate_model(mlp_model, test_features, test_labels, "MLP", pdf)
    results.append(["MLP", accuracy, precision, recall, f1])

    # Create summary table
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

    # Save summary results to a CSV file
    results_filename = f"evaluation-output/{output_file}_summary.csv"
    results_df.to_csv(results_filename, index=False)

    # Finalize the PDF with the plots and the table
    pdf.close()
    
    # Create the final report PDF
    add_summary_to_pdf_report(results_df, output_file, pdf_plots_filename)

    print("\nSummary of Model Performance:")
    print(results_df)


if __name__ == "__main__":
    evaluate_all_models()
