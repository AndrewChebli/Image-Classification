import torch
from torchvision import datasets, transforms
from models.resnet18_model import extract_features_from_dict

def get_dataset(batch_size = 500):
    
    transform = transform_dataset()
    # Download and load the training data
    raw_data = datasets.CIFAR10(root="data/images/transformed_data",train=True,download=True, transform=transform)
    print(f"Total number of images in dataset: {len(raw_data)}")
    print("downloading training images")
    # Create a DataLoader
    data_loader = torch.utils.data.DataLoader(raw_data,batch_size=batch_size)

    
    return data_loader

def transform_dataset():
    # Transform image size to 224 x 224 x 3
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], #mean and std based on ImageNet
                                                        std=[0.229, 0.224, 0.225])])
    return transform

def get_dataset(batch_size = 64):
    
    transform = transform_dataset()
    # Download and load the training data
    train_raw_data = datasets.CIFAR10(root="data/images/transformed_data/train_data",train=True,download=True, transform=transform)
    test_raw_data = datasets.CIFAR10(root="data/images/transformed_data/test_data",train=False,download=True, transform=transform)
    print(f"Total number of images in dataset: {len(train_raw_data)}")
    print(f"Total number of images in dataset: {len(test_raw_data)}")
    print("downloading training & testing images")

    # Create a DataLoader
    train_data_loader = torch.utils.data.DataLoader(train_raw_data,batch_size=batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_raw_data,batch_size=batch_size)

    
    return train_data_loader, test_data_loader

def build_label_image_dictionary(data_loader, image_limit):
    image_label_dictionary={}
    image_count = {i: 0 for i in range(10)}
    for images, labels in data_loader:
        for image, label in zip(images,labels):
            label = label.item()
            if image_count[label] < image_limit:
                if label not in image_label_dictionary:
                    image_label_dictionary[label] = []
                image_label_dictionary[label].append(image)
                image_count[label] += 1
        #stop when we reach limit for each class
        if all(count >= image_limit for count in image_count.values()):
            break
    return image_label_dictionary


# Testing the functions
if __name__ == "__main__":
    train_image_label_dictionary={}
    test_image_label_dictionary={}

    train_data_loader, test_data_loader = get_dataset()

    train_image_label_dictionary = build_label_image_dictionary(train_data_loader,500)
    test_image_label_dictionary= build_label_image_dictionary(test_data_loader,100)

# Print the number of images for each label
    print("---------------------- training data ----------------------")

    for label, images in train_image_label_dictionary.items():
        print(f"Label {label}: {len(images)} images")

    print("---------------------- testing data ----------------------")

    build_label_image_dictionary(test_data_loader, 100)

# Print the number of images for each label
    for label, images in test_image_label_dictionary.items():
        print(f"Label {label}: {len(images)} images")

    print("----------------------------------------------------------")

    _, _ = extract_features_from_dict(train_image_label_dictionary)
