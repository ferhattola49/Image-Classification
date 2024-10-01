Project Overview
This project aims to predict whether an image belongs to a plane or not using TensorFlow Keras.

Dependencies:
TensorFlow
Keras
NumPy
PIL
scipy
Dataset
To train and evaluate the model, you need to prepare a dataset with the following structure:

./dataset
├── test_set
│ ├── plane
│ └── nonplane
└── train_set
├── plane
└── nonplane
test_set contains images for testing the model.

plane directory: Contains images of planes.
nonplane directory: Contains images that are not planes.
train_set contains images for training the model.

plane directory: Contains images of planes.
nonplane directory: Contains images that are not planes.
Instructions
Download the Script: Download the provided script that includes the implementation of the image classification model.

Create Dataset: Create the dataset folder as described above. Populate the plane and nonplane directories within both train_set and test_set with corresponding images.

Change Paths: Update the script to point to the correct paths for your train_set and test_set directories.

Train the Model: Run the script to train the model with your dataset.

Test the Model: After training, you can use the model to predict whether new images belong to a plane or not. Update the script with the paths of the images you want to test and observe the results.
