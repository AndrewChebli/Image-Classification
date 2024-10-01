# Image-Classification Project


# Concordia University COMP472 Fall 2024 Due: 11:59 PM (EST), Nov 15, 2024, submit on Moodle.  
Qian Yi Wang
Andrew Chebli

---

### 1: Project Introduction: Image Classification
In this project, your task is to perform image classification on the CIFAR-10 dataset using some of the AI models learned in this class. You will first analyze the CIFAR-10 dataset, and pre-process the images. You will then build four AI models: Naive Bayes, a decision tree, a multi-layer perceptron, and a convolutional neural network (CNN) using Python and Pytorch to detect the 10 object classes in the CIFAR-10 dataset. You will then apply basic evaluation metrics: accuracy, confusion matrix, precision, and recall.

**Why these tasks matter:** Your pathway to AI Mastery. By undertaking this project, you’re immersing yourself in the world of different traditional and modern AI models that have applications in autonomous vehicles to medical imaging. Mastering image classification with Python and PyTorch is not just an academic exercise; it reflects the real-world demands of industries in pursuit of cutting-edge AI solutions. The tasks you engage with, from designing custom AI model architectures to conducting thorough evaluations, echo the challenges faced by AI professionals deploying solutions in ever-changing real-world settings, such as object detection in autonomous vehicles, obstacle avoidance in robotics, etc. Navigating through this project, understand that each skill and insight you acquire not only deepens your understanding but also strategically positions you for future discussions and initiatives in the rapidly evolving AI landscape.

---

### 2: Dataset Overview
The CIFAR-10 dataset contains 50,000 training and 10,000 test RGB images belonging to 10 object classes. Images are of size 32 × 32 × 3.
- In this project, you will only use 500 training images and 100 test images per class. Therefore, your first task is to load the dataset and use the first 500 training images and the first 100 test images of each class.
- The Naive Bayes, decision trees, and MLPs are not well-suited for direct application to high-dimensional RGB image data. Therefore, you will need to convert them into low-dimensional vectors through feature extraction. Pre-trained CNNs can serve as good feature extractors for image classification tasks. You will use a pre-trained ResNet-18 CNN to extract 512 × 1 feature vectors for the RGB images. For this, you will first need to resize the images to 224 × 224 × 3 and normalize them, because ResNet-18 pre-trained on ImageNet expects the images in a certain format. You will also need to remove the last layer of ResNet-18. Once these steps are finished, you can pass pre-processed RGB images through the pre-trained ResNet-18 to extract feature vectors. More details about using pre-trained CNNs in Pytorch can be found here: [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- Next, use PCA in scikit-learn to further reduce the size of feature vectors from 512×1 to 50×1.

---

### 3: Naive Bayes
1. Implement the Gaussian Naive Bayes algorithm in Python. You are only allowed to use the basic Python and Numpy libraries. Fit the Naive Bayes on the training feature vectors of all 10 classes.
2. Next, repeat the above step but using Scikit-learn’s Gaussian Naive Bayes classifier.
3. Evaluate both of these models on the test set of CIFAR-10 (feature vectors of the images in the test set). Evaluation details are stated below in Section 7.

---

### 4: Decision Tree
You are only allowed to use the basic Python and Numpy libraries for all parts unless specified otherwise.
1. Develop a decision tree classifier with the Gini coefficient and maximum depth of 50 and train it on the training set of CIFAR-10 (feature vectors).
2. Experiment by varying the depth of the tree. Observe and document how the depth of the tree influences the model’s ability to learn and generalize from the data.
3. Next, repeat step 1 using Scikit-learn’s implementation of a Decision Tree.
4. Evaluate all of the above models on the test set of CIFAR-10 (feature vectors of the images in the test set).

---

### 5: Multi-Layer Perceptron (MLP)
You can use the basic Python, Numpy, and Pytorch libraries for this part. Use the feature level test set of CIFAR-10 for all evaluations.
1. Implement a three-layer MLP (details below) and train it on the feature vectors of the CIFAR-10’s training set. The details of the MLP architecture are:
   - Linear(50, 512) - ReLU
   - Linear(512, 512) - BatchNorm(512) - ReLU
   - Linear(512, 10)

   You should use the cross-entropy loss `torch.nn.CrossEntropyLoss` for training. Also, use the SGD optimizer with momentum=0.9.

2. Experiment by varying the depth of the network by adding or removing layers. Observe and document how the depth of the MLP influences the model’s ability to learn and generalize from the data.
3. Vary the sizes of the hidden layers. Experiment with larger and smaller sizes. Analyze the trade-offs in computational cost and performance of the model.

---

### 6: Convolutional Neural Network
You can use the basic Python, Numpy, and Pytorch libraries for this project. Perform your evaluations on the test images of CIFAR-10.
1. Implement and train a VGG11 net on the training set of CIFAR-10. Use the training images directly for this part. VGG11 was an earlier version of VGG16 and can be found as model A in Table 1 of this paper, whose Section 2.1 also gives you all the details about each layer.

   For your convenience, we list the details of the VGG11 architecture here. The convolutional layers are denoted as `Conv(number of input channels, number of output channels, kernel size, stride, padding)`; the batch normalization layers are denoted as `BatchNorm(number of channels)`; the max-pooling layers are denoted as `MaxPool(kernel size, stride)`; the fully-connected layers are denoted as `Linear(number of input features, number of output features)`; the dropout layers are denoted as `Dropout(dropout ratio)`:
   - Conv(001, 064, 3, 1, 1) - BatchNorm(064) - ReLU - MaxPool(2, 2)
   - Conv(064, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2)
   - Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU
   - Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2)
   - Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU
   - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
   - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU
   - Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
   - Linear(0512, 4096) - ReLU - Dropout(0.5)
   - Linear(4096, 4096) - ReLU - Dropout(0.5)
   - Linear(4096, 10)

   You should use the following in your training process unless specified otherwise: cross-entropy loss `torch.nn.CrossEntropyLoss`, and optimize using SGD optimizer with momentum=0.9.

2. Experiment by adding or removing convolutional layers in your architecture. Observe and document how the depth of the network influences the model’s ability to learn and generalize from the data.
3. Adjust the kernel sizes used in your convolutional layers. Experiment with larger kernels (e.g., 5 × 5 or 7 × 7) as well as smaller ones (e.g., 2 × 2 or 3 × 3). Analyze the trade-offs in terms of spatial granularity versus computational cost and how different kernel sizes influence the recognition of broader features versus finer details.

---

### 7: Evaluation
Evaluate the performance of your four models and their variants on the test set of CIFAR-10.
- For each model, generate a confusion matrix to visualize classification performance. Ensure that classes are clearly labeled, either directly on the matrix or using an accompanying legend.
- Summarize your findings in a table detailing the metrics: accuracy, precision, recall, and F1-measure. The table must have separate rows for the four models and their variants.
- Only use the libraries clearly stated for each model. If you are unsure about the use of a specific module, please ask using the Moodle Discussion forum.
- It is strongly recommended to run the evaluation using a saved model that you load and test in your program. This avoids having to re-train a model when making any changes to the evaluation part of your code.

---

### 8: Deliverables
1. Your project code (ideally using Jupyter Notebook or .py scripts). Make sure it is well-structured, with clear functions and classes where appropriate. Use comments to describe the purpose of sections and major steps. Ensure reproducibility by setting random seeds.
2. A report that summarizes your results, findings, and reflections on the project. The report should include the following sections:
   - Title Page
   - Introduction
   - Methodology
   - Results
   - Discussion
   - Conclusion
   - References

---

### 9: Grading
Your project will be graded out of 100 points, based on the following components:
- Implementation and quality of code (40 points)
- Clarity and detail of evaluation metrics (30 points)
- Quality of report (20 points)
- Proper use of libraries (10 points)

---

**Important Note:** Each student in the group is required to submit an individual copy of the project files on Moodle with the same group number. You may include a .zip file of your project folder that contains the code, datasets used, and report. 

Good luck, and may you enjoy the process of exploration and learning in this project!
