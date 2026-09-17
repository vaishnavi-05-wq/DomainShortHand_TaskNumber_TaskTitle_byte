Cats vs Dogs Image Classification

Project Overview

This project implements a binary image classification system to distinguish between cats and dogs.

The project was completed as part of the Arithmatrix Virtual Internship Program (AVIP) 2026 under the AI/ML Engineering Basic domain.

A transfer learning approach using MobileNetV2 pretrained on ImageNet was used to build the classifier.

Dataset

The Cats vs Dogs dataset was obtained from the Microsoft Cats and Dogs dataset source.

Dataset Source

The dataset was obtained from the Microsoft Cats and Dogs dataset:

https://www.microsoft.com/en-us/download/details.aspx?id=54765

The original dataset contains cat and dog images organized into two classes:

Cat

Dog

Dataset Cleaning

The dataset was inspected and cleaned before model training.

Invalid or corrupted image files were identified and removed. Images were also converted to RGB format during the preprocessing pipeline to handle inconsistent image formats and channel representations.

After cleaning:

Total images: 24,994

Cat images: 12,499

Dog images: 12,495

Train, Validation and Test Split

The dataset was divided using a stratified split so that the class distribution remained balanced across the subsets.

Training images: 17,495

Validation images: 3,749

Test images: 3,750

The test set contains:

Cat: 1,875

Dog: 1,875

Image Preprocessing

Before being passed to the model, images were:

Converted to RGB

Resized to 224 × 224 pixels

Normalized to a pixel range of 0–1

The resulting input shape is:

224 × 224 × 3

Data Augmentation

Data augmentation was applied during training to improve model generalization.

The following augmentation techniques were used:

Random horizontal flipping

Random rotation

Random zoom

Model Architecture

MobileNetV2 was selected as the base model because it is a lightweight convolutional neural network suitable for image classification and transfer learning.

Model Structure

Input Image (224 × 224 × 3)
        ↓
Data Augmentation
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Dense Layer
        ↓
Sigmoid Output
        ↓
Cat / Dog

Transfer Learning

MobileNetV2 was initialized with ImageNet pretrained weights.

The convolutional base was frozen during the initial training stage, allowing the classifier to learn the Cats vs Dogs classification task without training the entire network from scratch.

This reduces training time and computational requirements.

Classification Layer

A single neuron with a sigmoid activation function was used for binary classification.

The output represents the probability of the image belonging to the Dog class.

Probability < 0.5 → Cat

Probability ≥ 0.5 → Dog

Training Configuration

The model was trained using:

Optimizer: Adam

Learning rate: 0.0001

Loss function: Binary Cross-Entropy

Maximum epochs: 5

Input size: 224 × 224 × 3

Training Callbacks

Two callbacks were used:

ModelCheckpoint — saves the best model based on validation accuracy

EarlyStopping — stops training when validation loss stops improving and restores the best weights

The best trained model was saved as:

models/cats_vs_dogs_best.keras

Model Evaluation

The saved model was evaluated on the held-out test set containing 3,750 images.

Test Results

Metric

Score

Accuracy

98.48%

Precision

98.20%

Recall

98.77%

F1-Score

98.48%

Confusion Matrix

The confusion matrix obtained on the test set was:

[[1841, 34],
 [23, 1852]]

The matrix represents:

                 Predicted
                 Cat    Dog

Actual Cat       1841    34
Actual Dog         23   1852

The model correctly classified most of the test images, with 34 cats incorrectly classified as dogs and 23 dogs incorrectly classified as cats.

The complete confusion matrix is available in:

results/confusion_matrix.txt

The evaluation metrics are available in:

results/evaluation_metrics.txt

Sample Inference

Ten sample images were passed through the trained model to demonstrate individual predictions.

Each sample contains:

Input image

Predicted label

Ground-truth label

Prediction confidence

All 10 generated sample predictions were correctly classified.

The sample prediction images are available in:

sample_predictions/

Inference Instructions

The trained model can be loaded and used for inference using the provided Python scripts.

Requirements

Install the required Python packages:

pip install tensorflow numpy pillow matplotlib scikit-learn

Running Sample Inference

To generate sample predictions, run:

python src/generate_samples.py

The generated sample prediction images are saved in:

sample_predictions/

Project Structure

The project is organized into separate folders for data, model files, results, sample predictions, notebooks, and source code.

Task1_Cats_vs_Dogs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── cats_vs_dogs_best.keras
│
├── notebooks/
│
├── results/
│   ├── evaluation_metrics.txt
│   └── confusion_matrix.txt
│
├── sample_predictions/
│   ├── sample_1_Dog_actual_Dog.png
│   ├── sample_2_Cat_actual_Cat.png
│   └── ...
│
├── src/
│   ├── analyze_dimensions.py
│   ├── check_dataset.py
│   ├── clean_dataset.py
│   ├── evaluate_model.py
│   ├── generate_samples.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   └── visualize_dataset.py
│
├── .gitignore
└── README.md

Limitations

Although the model achieved high accuracy on the test dataset, the project has some limitations.

The model was trained and evaluated only on the selected Cats vs Dogs dataset.

The images were resized to 224 × 224 pixels, which may cause some loss of image details.

The MobileNetV2 convolutional base was kept frozen during training, so the model was not fully fine-tuned for this specific dataset.

The model was trained using CPU-based computation, which increased the training time.

Real-world images with unusual lighting, backgrounds, poses, or image quality may produce different results.

The model is designed specifically for binary classification between cats and dogs and is not intended to classify other animal categories.

Future Improvements

The following improvements can be considered to make the Cats vs Dogs classification system more robust and suitable for real-world use:

Fine-tune the MobileNetV2 base model by unfreezing selected layers after initial training.

Use a larger and more diverse dataset containing images with different backgrounds, lighting conditions, and poses.

Apply more advanced data augmentation techniques to improve model generalization.

Experiment with other pretrained architectures such as EfficientNet or ResNet.

Optimize the trained model for faster inference and deployment on resource-constrained devices.

Develop a simple web or mobile interface where users can upload an image and receive a predicted label.

Monitor model performance on real-world images and retrain the model when necessary.

Conclusion

This project successfully implements a binary image classification system for distinguishing between cats and dogs.

A transfer learning approach using MobileNetV2 pretrained on ImageNet was used to build the classifier. The model was trained on a cleaned and stratified dataset and evaluated on a separate test set.

The model achieved an accuracy of 98.48% on the test dataset, demonstrating strong classification performance. The confusion matrix and sample inference results were also generated to evaluate the model's predictions.

The project demonstrates the practical application of deep learning, transfer learning, image preprocessing, data augmentation, model evaluation, and inference in an image classification problem.

Internship Task

This project was completed as Task 1 — Image Classification (Cats vs Dogs) for the Arithmatrix Virtual Internship Program (AVIP) 2026 under the AI/ML Engineering Basic domain.

The project fulfills the task requirements by including:

A documented Cats vs Dogs dataset and preprocessing pipeline

A trained image classification model

Test-set evaluation metrics

Confusion matrix

Ten sample inference images with predicted and ground-truth labels

A saved trained model

Inference instructions

Project documentation and source code