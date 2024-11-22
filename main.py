# import torch
# from models.Naive_Bayes.naive_bayes import NaiveBayesModel
# from models.Naive_Bayes.naive_bayes_scikit import ScikitNaiveBayesModel

# def load_train_labels_features():
#     data = torch.load('data/extracted_data/train_data.pt', weights_only=True)
#     train_features = data['features'].numpy()
#     train_labels = data['labels'].numpy()
#     return train_features, train_labels

# def load_test_labels_features():
#     data = torch.load('data/extracted_data/test_data.pt', weights_only=True)
#     test_features = data['features'].numpy()
#     test_labels = data['labels'].numpy()
#     return test_features, test_labels

# def Naive_Bayes():
#     custom_NaiveBayes_model = NaiveBayesModel()
#     scikit_NaiveBayes_model = ScikitNaiveBayesModel()

#     # Load train and test data
#     train_features, train_labels = load_train_labels_features()
#     test_features, test_labels = load_test_labels_features()

#     # Train and evaluate custom Naive Bayes model
#     priors = custom_NaiveBayes_model.calculate_probability_of_labels(train_labels)
#     mean_std = custom_NaiveBayes_model.calculate_parameters(train_features, train_labels)
#     accuracy = custom_NaiveBayes_model.evaluate_model(test_features, test_labels, mean_std, priors)
#     print(f"custom Naive Bayes Accuracy: {accuracy:.2f}%")

#     # Train and evaluate Scikit-Learn GaussianNB model
#     scikit_NaiveBayes_model.train(train_features, train_labels)
#     y_prediction = scikit_NaiveBayes_model.predict(test_features)
#     scikit_accuracy = scikit_NaiveBayes_model.evaluate_model(y_prediction, test_labels)
#     print(f"Scikit-Learn GaussianNB Accuracy: {scikit_accuracy:.2f}%")

# if __name__ == "__main__":
#     Naive_Bayes()


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Naive_Bayes.naive_bayes import main as naive_bayes_main
from models.Naive_Bayes.naive_bayes_scikit import main as naive_bayes_scikit_main
from models.Decision_Tree.decision_tree import main as decision_tree_main
from models.Decision_Tree.decision_tree_scikit import main as decision_tree_scikit_main
from models.MLP.mlp import main as mlp_main
# from models.CNN.cnn import main as cnn_main
from Evaluation.evaluation import main as evaluate_all_models

def display_menu():
    """Display the main menu options to the user."""
    print("===========================================")
    print("Welcome to the Image Classification System")
    print("===========================================")
    print("Please choose an option:")
    print("1. Run Naive Bayes Model (Custom) ")
    print("2. Run Naive Bayes Model (Scikit-Learn)")
    print("3. Run Decision Tree Model (Custom)")
    print("4. Run Decision Tree Model (Scikit-Learn)")
    print("5. Run MLP Model")
    print("6. Run CNN Model")
    print("7. Run Evaluation Script")
    print("8. Exit")

def handle_user_input(choice):
    """Handle user input based on their selection."""
    if choice == "1":
        print("\nRunning Naive Bayes Model (Custom)...\n")
        naive_bayes_main()  # Calls the main function of the Custom Naive Bayes model
    elif choice == "2":
        print("\nRunning Naive Bayes Model (Scikit-Learn)...\n")
        naive_bayes_scikit_main()  # Calls the main function of the Scikit-Learn Naive Bayes model
    elif choice == "3":
        print("\nRunning Decision Tree Model (Custom)...\n")
        decision_tree_main()  # Calls the main function of the Custom Decision Tree model
    elif choice == "4":
        print("\nRunning Decision Tree Model (Scikit-Learn)...\n")
        decision_tree_scikit_main()  # Calls the main function of the Scikit-Learn Decision Tree model
    elif choice == "5":
        print("\nRunning MLP Model...\n")
        mlp_main()  # Calls the main function of the MLP model
    elif choice == "6":
        print("\nRunning CNN Model...\n")
        # cnn_main()  # Calls the main function of the CNN model
    elif choice == "7":
        print("\nRunning Evaluation Script...\n")
        print(f"Current Working Directory: {os.getcwd()}")
        evaluate_all_models()  # Calls the evaluation script
    elif choice == "8":
        print("\nExiting the program...")
        sys.exit()  # Exits the program
    else:
        print("\nInvalid option! Please choose a valid option from 1 to 8.")

def main():
    """Main function that runs the menu loop."""
    while True:
        display_menu()  # Displays the menu
        choice = input("Enter your choice (1-8): ")
        handle_user_input(choice)  # Processes user input

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()


