Digit Prediction CNN Model
This project implements a Convolutional Neural Network (CNN) to classify digits from the Digits Dataset using PyTorch. The model is trained with techniques such as data augmentation, early stopping, and dropout regularization to improve accuracy and robustness.

Project Structure
Data Preparation: The dataset is preprocessed, scaled, and resized for input to the CNN.
Model Definition: The CNN model is defined with convolutional, pooling, dropout, and fully connected layers.
Training: Includes reduced data augmentation, early stopping, and class weights handling for balanced training.
Evaluation: Evaluates test set performance with a confusion matrix and loss/accuracy metrics.
Prerequisites
To run this project, you will need:

Python 3.8+
PyTorch
Additional libraries: scikit-learn, scipy, matplotlib, torchvision
Install required packages enter this on bash terminal:

pip install torch torchvision scikit-learn scipy matplotlib

Model Overview
Data Preparation
Load Dataset: The dataset is loaded from sklearn.datasets.
Scaling and Resizing: Features are scaled to [0, 1], reshaped, and resized to 28x28.
Dataset Splitting: The data is split into training, validation, and test sets (70%, 15%, and 15%).
Data Augmentation: Simple augmentation is applied to the training data to reduce overfitting.
CNN Architecture
The model is a basic CNN with:

Convolutional Layers: Two convolutional layers with ReLU activations.
Pooling Layer: A 2x2 max-pooling layer.
Dropout Layer: Dropout is applied to reduce overfitting.
Fully Connected Layers: A fully connected layer with 128 neurons and a final layer for 10 classes (digits 0-9).
Training and Early Stopping
The model is trained with a patience-based early stopping criterion to avoid overfitting. The best model (lowest validation loss) is saved and loaded automatically.

Evaluation and Metrics
The model is evaluated using:

Accuracy: Overall test set accuracy.
Confusion Matrix: Visualization of classification performance per class.
Loss Curves: Training and validation loss over epochs with early stopping point highlighted.
Code Usage
Running the Code
Set Up the Environment: Ensure you have all dependencies installed.
Train the Model: Execute the script to train the model. Early stopping will save the best model.
Evaluate the Model: The best model is evaluated on the test set, with metrics displayed in the console.
Plot Results: Plots are generated for the confusion matrix and loss curves.


Important Hyperparameters
batch_size: Set to 64 for training and evaluation.
epochs: Default is 50, with early stopping based on validation loss.
learning_rate: 0.0005 for the Adam optimizer.
patience: Number of epochs without improvement before stopping.
Model Evaluation
On completion, the model’s performance on the test dataset is summarized with:

Accuracy: Percentage of correct predictions.
Loss: Average loss on the test dataset.
Confusion Matrix: A graphical representation of model accuracy per digit.

Results
Initial training on the Digits dataset, without data augmentation, showed that the model reached 98% accuracy.
However, this high accuracy risked overfitting, as the dataset's simplicity allowed the model to memorize patterns 
without generalizing well to complex, unseen data. To enhance robustness, data augmentation was applied to introduce 
variation in training examples, and early stopping was used to prevent overfitting by monitoring validation loss. 
These techniques aim to ensure the model performs well in diverse real-world scenarios. 
This theoretical analysis requires further evaluation, planned for a future stage of the project.

