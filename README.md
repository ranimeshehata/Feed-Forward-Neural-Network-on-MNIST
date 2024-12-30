# Feed-Forward-Neural-Network-on-MNIST

This repository contains the implementation of a Feed Forward Neural Network (FFNN) to classify handwritten digits from the MNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction
The MNIST dataset is a large database of handwritten digits commonly used for training various image processing systems. This project implements a Feed Forward Neural Network to classify the digits with high accuracy.

## Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels.

The dataset is automatically downloaded and preprocessed using `torchvision.datasets`.

## Model Architecture
The Feed Forward Neural Network is defined in the `FeedForwardNeuralNetwork` class. The architecture consists of:
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer 1: 64 neurons
- Hidden layer 2: 32 neurons
- Output layer: 10 neurons (one for each digit class)

## Training

The training process involves the following steps:
1. **Initialize the model, loss function, and optimizer**:
    - Model: Softmax Regression
    - Loss Function: Cross-Entropy Loss
    - Optimizer: Stochastic Gradient Descent (SGD)

2. **Training Loop**:
    - For each epoch, iterate over the training dataset in batches.
    - Perform the forward pass to compute the model's predictions.
    - Compute the loss between the predictions and the ground truth labels.
    - Perform the backward pass to compute the gradients.
    - Update the model's parameters using the optimizer.

## Evaluation

The evaluation process involves:
1. **Validation**:
    - Evaluate the model on the validation dataset after each epoch.
    - Compute the validation loss and accuracy to monitor the model's performance.

2. **Testing**:
    - After training, evaluate the model on the test dataset.
    - Compute the test accuracy to assess the model's generalization performance.

## Results

The results of the training and evaluation process include:
- Training and validation loss over epochs.
- Training and validation accuracy over epochs.
- Test accuracy after training.
- Confusion matrix for the test dataset predictions.
- Plots for visualization.
- With and without L2 Regularization.

## Usage

To use this implementation, follow these steps:
1. Clone the repository:
    ```
    git clone https://github.com/ranimeshehata/Feed-Forward-Neural-Network-on-MNIST.git
    cd "Feed Forward Neural Network on MNIST"
    ```

2. Install the required dependencies:
    ```
    pip install torch torchvision scikit-learn matplotlib
    ```

3. Run the model

4. Observe the outputs and you can change batch size, learning rate or number of epochs for model tuning.

## Dependencies

- Python 3.9
- PyTorch
- torchvision
- scikit-learn
- matplotlib