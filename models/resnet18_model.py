import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet():
    #load pre trained model resnet 18
    resnet18_model = resnet18(weights= ResNet18_Weights.IMAGENET1K_V1)

    #remove last layer to only use feature extraction from the model
    newmodel = nn.Sequential(*(list(resnet18_model.children())[:-1]))
    return newmodel

# Function to extract features from a batch of images using ResNet-18
def extract_features(images):
    model = get_resnet()  # Load the modified ResNet-18 model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for faster inference
        features = model(images)  # Pass the images through ResNet-18
        return features.flatten(start_dim=1)  # Flatten the output to (batch_size, 512)


# Function to extract features directly from the images stored in the dictionary
def extract_features_from_dict(image_label_dict):
    all_features = []
    all_labels = []

    for label, images in image_label_dict.items():
        print(f"Extracting features for label: {label}")
        
        # The images are already pre-processed, just stack them
        images = torch.stack(images)  # Combine list of images into a batch
        
        # Extract features using ResNet-18
        features = extract_features(images)

        # Append the features and corresponding labels
        all_features.append(features)
        all_labels.extend([label] * len(images))  # Append the label for each image
    
    # Combine all features and labels into single tensors
    all_features = torch.vstack(all_features)
    all_labels = torch.tensor(all_labels)

     # where we save the data
    save_folder = os.path.join('data', 'extracted_data')

    # Create the folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Path for saving the features and labels
    features_save_path = os.path.join(save_folder, 'features.pt')
    labels_save_path = os.path.join(save_folder, 'labels.pt')   

    # Save the extraction to not re computate them every time
    torch.save(all_features, features_save_path)
    torch.save(all_labels, labels_save_path)


    return all_features, all_labels
