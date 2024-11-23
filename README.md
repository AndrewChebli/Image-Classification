# Image Classification System

## Project Overview

This project implements an Image Classification System using various machine learning models including Naive Bayes, Decision Trees, Multilayer Perceptrons (MLP), and Convolutional Neural Networks (CNNs). The system supports training, evaluation, and prediction using these models on image datasets.

## Project Structure

	|-- main.py: Entry point for the project. Provides a menu to select different models or the evaluation script.
    |-- data/
        |-- CIFAR-10 datasets and the necessary preprocessing scripts.
        |-- data_loader.py: Script for loading and preprocessing the dataset.
    |-- models/
        |-- Naive_Bayes/
            |-- naive_bayes.py: Custom implementation of Naive Bayes.
            |-- naive_bayes_scikit.py: Scikit-learn implementation of Naive Bayes.
            |-- resnet18_model.py: ResNet-18 feature extractor for advanced feature extraction.
        |-- Decision_Tree/
            |-- decision_tree.py: Custom implementation of Decision Tree.
            |-- decision_tree_scikit.py: Scikit-learn implementation of Decision Tree.
        |-- MLP/
            |-- mlp.py: Implementation of Multilayer Perceptron (MLP) with customizable architecture.
        |-- CNN/
            |-- cnn.py: Convolutional Neural Network (CNN) implementation for image classification.
    |-- Evaluation/
        |-- evaluation.py: Evaluates all models and generates confusion matrices and metrics reports.
    |-- output/
        |-- Trained model files and evaluation reports.
        |-- *.pth, *.pkl: Saved model weights and parameters.
    |-- requirements.txt: Dependencies required to run the project.
    |-- README.md: Project documentation.
    |-- report/: Contains additional documentation and reporting files (if any).

Setup Instructions

Prerequisites

    Python 3.8 or higher installed along with pip.

Steps

	1.	Clone the repository to your local machine.

git clone <repository_url>
cd Image-Classification


	2.	Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate (On Mac)


	3.	Install the required dependencies:

pip install -r requirements.txt


	4.	Download and preprocess the dataset (CIFAR-10 is used):
	5.	The data_loader.py script handles the preprocessing.


## 1. Running the Main Menu

Run the project entry point:

python main.py

You will see a menu with the following options:
	•	Run Naive Bayes Model (Custom/Scikit-Learn)
	•	Run Decision Tree Model (Custom/Scikit-Learn)
	•	Run MLP Model
	•	Run CNN Model
	•	Run Evaluation Script
	•	Exit

Follow the prompts to select a model and perform actions like training, evaluation, or testing.

## 2. Running Specific Models

### Naive Bayes (Custom):
    python3 models/Naive_Bayes/naive_bayes.py
### Naive Bayes (Scikit-Learn):
    python3 models/Naive_Bayes/naive_bayes_scikit.py
### Decision Tree (Custom):
    python3 models/Decision_Tree/decision_tree.py
### Decision Tree (Scikit-Learn):
    python models/Decision_Tree/decision_tree_scikit.py
### MLP:
    python3 models/MLP/mlp.py
### CNN:
    python3 models/CNN/cnn.py



## 3. Evaluating Models

To generate evaluation reports:

    python Evaluation/evaluation.py

## 4. Output and Reports

	•	Model weights are saved in the output/ directory with filenames reflecting their architecture and configurations.
	•	Evaluation reports (confusion matrices and metrics summaries) are stored in evaluation-output/.

## Customization Notes:
    - As seen in the MLP.py and decision tree models files, certain parameters like maximum depth or network architecture need to be updated manually in the code if changes are required. Be sure to modify these parameters directly in the respective files before execution. (ie: if you need to change the depth in the mlp file, and need that in the report, you will need to change it in the evaluation.py file as well.)
    - Hyperparameter Tuning: Models like MLP and CNN allow tuning of hyperparameters like learning rate, batch size, and number of epochs. Update these values in their respective training functions if different hyperparameter settings are required. 
Project Features

	1.	Preprocessing: Automates loading and preprocessing of CIFAR-10 datasets.
	2.	Model Training and Testing: Flexible training for Naive Bayes, Decision Trees, CNNs, and MLPs.
	3.	Evaluation: Comprehensive performance evaluation using confusion matrices, accuracy, precision, recall, and F1 score.
	4.	Modular Design: Each model is implemented in its own module for clarity and scalability.
	5.	Reports: Automatically generated evaluation reports in PDF and CSV formats.

### Instructions for Dataset

    - The CIFAR-10 dataset is automatically downloaded when the models are executed. Ensure your system has internet access during the first run.

## Example Commands

- To train and evaluate a Decision Tree model with a maximum depth of 10:

        python models/Decision_Tree/decision_tree.py

- To test an already trained CNN model:

        python models/CNN/cnn.py

- Customizing Max depth for Decision Tree(Scikit-Learn): if you want to experiment with different maximum depths, you should modify the max_depths list

    ```python
    def main():
    set_random_seeds()
    max_depths = [10, 20, 50]  # Predefined maximum depths for the Decision Tree
    for max_depth in max_depths:
        sklearn_tree = DecisionTreeModelSklearn(max_depth=max_depth)
        train_features, train_labels = sklearn_tree.load_train_labels_features()
        test_features, test_labels = sklearn_tree.load_test_labels_features()
        sklearn_tree.fit(train_features, train_labels)
        accuracy = sklearn_tree.evaluate_model(test_features, test_labels)
        sklearn_tree.save_model(f'./output/decision_tree_sklearn_model_{max_depth}.pkl')
        print(f"Scikit-learn Decision Tree with max_depth={max_depth} Accuracy: {accuracy:.2f}%")```

Notes

	•	File Locations: Ensure all files remain in their respective directories for smooth execution.
