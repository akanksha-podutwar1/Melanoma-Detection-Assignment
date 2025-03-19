# Melanoma Detection Using Custom CNN

## Problem Statement

In this project, we are tasked with building a multiclass classification model using a custom Convolutional Neural Network (CNN) in TensorFlow to detect melanoma. Melanoma is a type of skin cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A model capable of evaluating images and alerting dermatologists about the presence of melanoma can significantly reduce the manual effort required for diagnosis.

### Dataset

The dataset consists of 2,357 images of malignant and benign oncological diseases, provided by the International Skin Imaging Collaboration (ISIC). These images were sorted based on classifications taken with ISIC, and the subsets were divided into nearly equal numbers, except for melanomas and moles, which have slightly more images.

The dataset includes the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

You can download the dataset [here](#).

### Assignment Notes

- The model building process must be based on a custom CNN. No pre-trained models or transfer learning should be used.
- Ensure that your model is trained from scratch, based on the provided base code.
- The training process may take time due to a large number of epochs, and it's recommended to use a GPU runtime in Google Colab.

## Project Pipeline

### 1. Data Reading / Data Understanding

- Define the paths for the training and testing images.
- Load and explore the dataset to understand its structure.

### 2. Dataset Creation

- Create a training and validation dataset from the training directory with a batch size of 32.
- Ensure that all images are resized to 180x180 pixels.

### 3. Dataset Visualization

- Implement a visualization code to display one sample image from each of the nine classes in the dataset for a better understanding of the data distribution.

### 4. Model Building & Training (Initial Model)

- Build a custom CNN model capable of detecting the nine classes in the dataset.
- Normalize the pixel values of the images by rescaling them to a range of (0, 1).
- Choose an appropriate optimizer and loss function for training.
- Train the model for approximately 20 epochs.

### 5. Evaluating Model Performance

- After training, evaluate the model to check for overfitting or underfitting.
- Write findings based on model performance and any evidence of overfitting or underfitting.

### 6. Data Augmentation

- If there is evidence of underfitting or overfitting, apply data augmentation techniques to enhance the model's performance.
- Retrain the model with the augmented data for approximately 20 epochs.
- Evaluate the performance again and document if the issue has been resolved.

### 7. Handling Class Imbalance

- Examine the current class distribution in the training dataset.
  - Identify which class has the least number of samples.
  - Identify which classes dominate the dataset in terms of the proportion of samples.
- Use the Augmentor library to rectify any class imbalances in the dataset.

### 8. Model Building & Training on Rectified Data

- After rectifying class imbalances, retrain the custom CNN model with the normalized images.
- Choose an appropriate optimizer and loss function.
- Train the model for approximately 30 epochs.
- Evaluate the model again and document the findings to see if the class imbalance issues were resolved.

## Requirements

- TensorFlow
- Keras
- Augmentor (for handling class imbalances)
- Matplotlib (for visualizations)
- Python 3.x

## Findings

After training the model and applying the necessary data augmentation and class imbalance rectification techniques, write your findings. Check if the model shows evidence of improvement, and if the earlier issues (e.g., overfitting, underfitting, or class imbalance) have been resolved.

## Conclusion

This project aims to develop a robust CNN model capable of accurately detecting melanoma and other skin diseases from images. With a well-built model and proper handling of data issues, it has the potential to assist dermatologists in the early detection of melanoma and other oncological diseases.


## Contact

Created by [@AkankshaPodutwar - feel free to contact me!
