import os
import random
import sys
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import torch.utils.data as td
import torch.nn as nn
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.MLP.mlp import MLP, load_data
from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel
from models.Decision_Tree.decision_tree import DecisionTreeClassifier
from models.Decision_Tree.decision_tree_scikit import DecisionTreeModelSklearn
from models.Naive_Bayes.naive_bayes import NaiveBayesModel
from models.CNN.cnn import CNN, cifar_loader, remove_last_layer, add_extra_conv_layer


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
    if hasattr(model, 'predict'):
        predictions = model.predict(test_features)  # Call the predict method
    else:
        raise TypeError(f"The model '{model_name}' does not have a 'predict' method.")
    
    # Ensure test_labels and predictions are compatible (e.g., both NumPy arrays)
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    accuracy, precision, recall, f1 = calculate_metrics(test_labels, predictions)
    plot_confusion_matrix(test_labels, predictions, model_name, pdf)
    return accuracy, precision, recall, f1

def evaluate_cnn_model(model, test_loader, device, model_name, pdf):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    y_true, y_pred = [], []
    with torch.no_grad():  # Disable gradient computations
        for instances, labels in test_loader:
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)  # Forward pass
            predictions = torch.max(output, 1)[1]  # Get the predicted class
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, model_name, pdf)
    return accuracy, precision, recall, f1

def add_summary_to_pdf_report(results_df, pdf):
    # Format values for display in the PDF table (3 decimal places and percentages)
    results_df_display = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        results_df_display[col] = (results_df_display[col]).round(3)

    # Create a figure to render the table as an image for the PDF
    fig, ax = plt.subplots(figsize=(10, 2 + len(results_df_display) * 0.8))
    ax.axis('tight')
    ax.axis('off')

    # Create the table from the formatted DataFrame
    table = ax.table(cellText=results_df_display.values,
                     colLabels=results_df_display.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

     # Adjust the width of the first column
    for i in range(len(results_df_display) + 1):  # +1 to include the header
        cell = table[(i, 0)]  # (row, column), 0 for the first column
        cell.set_width(0.399)  # Set a wider width for the first column

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
    max_depths = [10, 20, 50]
    for max_depth in max_depths:
        decision_tree_model = DecisionTreeClassifier.load_model(f'./output/decision_tree_model_{max_depth}.pkl')
        accuracy, precision, recall, f1 = evaluate_model(decision_tree_model, test_features, test_labels, f"Decision Tree - Max Depth: {max_depth}", pdf)
        results.append([f"Decision Tree - Max Depth: {max_depth}", accuracy, precision, recall, f1])

    # Scikit Decision Tree
    max_depths = [10, 20, 50]
    for max_depth in max_depths:
        decision_tree_sklearn_model = DecisionTreeModelSklearn.load_model(f'./output/decision_tree_sklearn_model_{max_depth}.pkl')
        accuracy, precision, recall, f1 = evaluate_model(decision_tree_sklearn_model, test_features, test_labels, f"Decision Tree Scikit - Max Depth: {max_depth}", pdf)
        results.append([f"Decision Tree Scikit - Max Depth: {max_depth}", accuracy, precision, recall, f1])

    # MLP
    experiments = [
        {"hidden_sizes": [512, 512], "description": "2-layer"},
        {"hidden_sizes": [512], "description": "1 hidden layer"},
        {"hidden_sizes": [512, 512, 512], "description": "3-layer MLP"},
        {"hidden_sizes": [256, 256], "description": "Smaller hidden layers"},
        {"hidden_sizes": [1024, 1024], "description": "Larger hidden layers"}
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
   
    # Define different configurations for experiments
    
    test_features, test_labels = load_data('data/extracted_data/test_data.pt')
    test_features, test_labels = test_features.to(device), test_labels.to(device)
    for experiment in experiments:
        model_path = f'./output/mlp_{experiment["description"].replace(" ", "_").lower()}.pth'
        mlp_model = MLP.load_model(model_path, input_size=50, hidden_sizes=experiment["hidden_sizes"], num_classes=10).to(device)
        accuracy, precision, recall, f1 = evaluate_model(mlp_model, test_features, test_labels, f'MLP - Network Depth: {experiment["description"]}',pdf)
        results.append([f'MLP - Network Depth: {experiment["description"]}', accuracy, precision, recall, f1])

    #CNN
    _, test_dataset = cifar_loader(batch_size = 32)
    test_loader = td.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    models_dir = './output/'  # Directory where CNN models are saved
    cnn_files = [f for f in os.listdir(models_dir) if f.startswith('cnn') and f.endswith('.pth')]
    for cnn_file in cnn_files:
        kernel_size = int(cnn_file.split('_')[3])
        model = CNN(3 * 32 * 32, 10, kernel_size).to(device)

        # Apply modifications based on filename
        if "remove_last_layer" in cnn_file:
            model = remove_last_layer(model)
        elif "add_extra_layer" in cnn_file:
            model = add_extra_conv_layer(model)

        # Load model weights
        model.load_state_dict(torch.load(os.path.join(models_dir, cnn_file ),map_location="cpu"))
        
        #move model to the correct device
        model = model.to(device)

        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_cnn_model(model, test_loader, device, cnn_file, pdf)
        results.append([cnn_file, accuracy, precision, recall, f1])

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


def set_random_seeds(seed=88):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_random_seeds()
    evaluate_all_models()

if __name__ == "__main__":
   main()
