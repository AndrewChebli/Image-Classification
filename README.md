# Image Classification on CIFAR-10 Dataset

## Project Overview

This project involves applying various machine learning models to classify images from the CIFAR-10 dataset. The models implemented include **Naive Bayes**, **Decision Tree**, **MLP (Multi-Layer Perceptron)**, and **CNN (Convolutional Neural Network)**. The goal is to explore the effectiveness of these models for image classification by evaluating their performance using accuracy, precision, recall, and F1-score metrics.

---

## Table of Contents

1. [Project Files](#project-files)
2. [Data Preprocessing](#data-preprocessing)
3. [Training and Evaluating Models](#training-and-evaluating-models)
4. [Running the Code](#running-the-code)
5. [Model Evaluation](#model-evaluation)
6. [Requirements](#requirements)
7. [License](#license)

---

## Project Files

This project is structured as follows:

### `main_menu.py`
- **Purpose**: This is the main script that runs the interactive menu for the user. It allows the user to choose between different models (Naive Bayes, Decision Tree, MLP, CNN) or run the evaluation script.
- **Execution**: When executed, this script will display a menu for the user to select a model to run, or to run the evaluation script. It invokes the appropriate model training and evaluation scripts based on the user's input.

### `naive_bayes.py`
- **Purpose**: This file implements a custom Naive Bayes model for classification.
- **Functionality**: It includes functions to calculate the prior probabilities of labels and the conditional probabilities for the features.

### `naive_bayes_scikit.py`
- **Purpose**: This file implements the Naive Bayes model using the Scikit-Learn library.
- **Functionality**: It uses Scikit-Learn's `GaussianNB` to train and predict on the CIFAR-10 dataset.

### `decision_tree.py`
- **Purpose**: This file contains a custom implementation of a Decision Tree model for classification.
- **Functionality**: It implements a basic decision tree using a recursive approach to build the tree based on Gini impurity.

### `decision_tree_scikit.py`
- **Purpose**: This file implements a Decision Tree model using Scikit-Learn's `DecisionTreeClassifier`.
- **Functionality**: It leverages Scikit-Learn's decision tree algorithms to train and predict on the CIFAR-10 dataset.

### `mlp.py`
- **Purpose**: This file defines a Multi-Layer Perceptron (MLP) for image classification.
- **Functionality**: The MLP is constructed with multiple hidden layers, and it is trained using stochastic gradient descent (SGD).

### `cnn.py`
- **Purpose**: This file defines a Convolutional Neural Network (CNN) for image classification.
- **Functionality**: It implements a VGG11 architecture for image classification, leveraging convolutional layers for feature extraction.

### `evaluation.py`
- **Purpose**: This file contains functions to evaluate the performance of the models.
- **Functionality**: It computes various evaluation metrics such as accuracy, precision, recall, and F1-score. It also includes confusion matrix generation.

---

## Data Preprocessing

Before training the models, the CIFAR-10 dataset must be pre-processed to extract features and prepare the data.

### Steps to Preprocess the Data

1. **Load CIFAR-10 Dataset**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes. First, ensure that the dataset is downloaded and placed in the correct directory.

2. **Resizing Images**: To use the pre-trained ResNet-18 model for feature extraction, images are resized from 32x32 pixels to 224x224 pixels.

3. **Feature Extraction with ResNet-18**:
    - Use the pre-trained ResNet-18 model (pre-trained on ImageNet) to extract feature vectors of size 512x1 for each image.
    - These feature vectors are then reduced to 50x1 dimensions using **Principal Component Analysis (PCA)** for better performance with traditional machine learning models.

4. **Saving Preprocessed Data**: The extracted features and labels are saved in `.pt` files for both the training and testing datasets.

---

## Training and Evaluating Models

### Step 1: Model Training

To train the models, simply run the corresponding script via the menu interface or by invoking them directly.

1. **Naive Bayes**:
   - Custom implementation (`naive_bayes.py`).
   - Scikit-Learn implementation (`naive_bayes_scikit.py`).

2. **Decision Tree**:
   - Custom implementation (`decision_tree.py`).
   - Scikit-Learn implementation (`decision_tree_scikit.py`).

3. **MLP**:
   - Defined and trained in `mlp.py`.

4. **CNN**:
   - VGG11 architecture defined and trained in `cnn.py`.

### Step 2: Model Evaluation

Once models are trained, you can evaluate their performance by running the evaluation script:
- **Evaluation Script** (`evaluation.py`): This will compute metrics such as accuracy, precision, recall, and F1-score, and display confusion matrices for each model.

---

## Running the Code

To run the interactive menu and invoke model training, evaluation, and predictions, execute the following script:

```bash
python main_menu.py
```

This will display the following menu options:

1. **Run Naive Bayes Model (Custom)**: Runs the custom implementation of Naive Bayes.
2. **Run Naive Bayes Model (Scikit-Learn)**: Runs the Scikit-Learn Naive Bayes model.
3. **Run Decision Tree Model (Custom)**: Runs the custom implementation of Decision Tree.
4. **Run Decision Tree Model (Scikit-Learn)**: Runs the Scikit-Learn Decision Tree model.
5. **Run MLP Model**: Runs the MLP model.
6. **Run CNN Model**: Runs the CNN (VGG11) model.
7. **Run Evaluation Script**: Evaluates the models' performance.
8. **Exit**: Exits the program.

### Example Flow:

1. Select "1" to run the **Naive Bayes** model (custom implementation).
2. Wait for the model to train and evaluate, printing the results.
3. Select "5" to run the **Evaluation Script** and review the results.

---

## Model Evaluation

The models are evaluated using the following metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: The proportion of positive predictions that are correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.

Confusion matrices are also generated for each model to visualize misclassifications.

---

## Requirements

To run the code, ensure that the following dependencies are installed:

```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

Additionally, the project requires access to the CIFAR-10 dataset, which can be downloaded via the `torchvision.datasets` module.
