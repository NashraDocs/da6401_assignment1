# da6401_assignment1
# Deep Learning Model: Training and Evaluation Guide

## Overview
This repository contains a deep learning model implemented in Python  Weights & Biases (W&B) for experiment tracking and hyperparameter tuning. The model is trained on the Fashion-MNIST dataset with various optimizers, batch sizes, and network architectures.

## Prerequisites
Ensure you have the following dependencies installed:
```bash

pip install wandb tensorflow numpy
```
Additionally, set up Weights & Biases by logging in:
```bash
wandb login
```

## Dataset Preparation
1. The model uses the Fashion-MNIST dataset for training.
2. Data is normalized by dividing by 255.0 to scale pixel values between 0 and 1.
3. Training and test datasets are automatically loaded using `keras.datasets.fashion_mnist.load_data()`.
4. The dataset is split into training, validation, and test sets using `train_test_split()`.

## Training the Model
The model is built using TensorFlow/Keras with multiple configurable hidden layers and neurons per layer. 
To train the model, run:
```bash
python train.py --wandb_entity <your_entity> --wandb_project <your_project>
```

### Function Details:
- `model_builded(hidden_layers, neurons_per_layer)`: 
  - Constructs a sequential neural network with specified layers.
  - Uses ReLU activation for hidden layers and Softmax for output.
  - Compiles with Adam optimizer and sparse categorical cross-entropy loss.

- `train(config=None)`: 
  - Runs a W&B Sweep experiment with different hyperparameter configurations.
  - Logs metrics in W&B, including accuracy, loss, and validation accuracy.
  - Configures the optimizer dynamically (Adam, SGD, RMSprop, or Nadam).
  - Uses L2 weight regularization and different kernel initializations (`random` or `glorot`).

- `main(wandb_entity, wandb_project, hidden_layers, neurons_per_layer, epochs, batch_size)`: 
  - Initializes W&B and runs the training process.
  - Logs final validation loss and accuracy.

## Hyperparameter Tuning with W&B Sweeps
To run hyperparameter tuning, execute:
```bash
python sweep.py
```
This script defines a sweep configuration for optimizing hyperparameters such as:
- Number of hidden layers
- Neurons per layer
- Dropout rate (`0.2`, `0.3`, `0.5`)
- Activation function (`relu` or `tanh`)
- Weight initialization (`random` or `glorot`)
- Learning rate (`0.01`, `0.001`, `0.0001`)
- Optimizers (`adam`, `sgd`, `rmsprop`, `nadam`)
- Number of epochs (`5` or `10`)
- Batch size (`32` or `64`)

The W&B agent runs multiple trials to find the best configuration.

## Evaluating the Model
To evaluate the trained model, run the evaluation script:
```bash
python evaluate.py
```
### Function Details:
- `model.evaluate(X_test, y_test)`: 
  - Computes test accuracy and loss after training.
  - Logs test performance in W&B.

## Experiment Tracking
- The model is trained with different hyperparameter settings using W&B Sweeps.
- Training loss, validation accuracy, and test accuracy are logged in W&B.
- Model performance for different configurations can be analyzed through W&B logs.

## Notes
- Modify `batch_sizes` and `optimizers` in the training loop to test different configurations.
- Ensure W&B is properly initialized before running training.
- Check experiment logs in the W&B dashboard for detailed analysis.


---



