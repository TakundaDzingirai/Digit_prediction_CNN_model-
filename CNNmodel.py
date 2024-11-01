import torch
from torchvision import transforms, datasets as torch_datasets
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.optim as optim
import random
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set the seed for reproducibility

# Load the Digits Dataset from sklearn
digits = datasets.load_digits()
X_digits, y_digits = digits.data, digits.target
scaler = MinMaxScaler()
X_digits = scaler.fit_transform(X_digits)

# Reshape the digits dataset from (1797, 64) to (1797, 8, 8)
X_digits = X_digits.reshape(-1, 8, 8)

# Resize each 8x8 image to 28x28 using scipy's zoom function
zoom_factor = 28 / 8  # Calculate zoom factor
X_digits_resized = np.array([zoom(img, zoom_factor) for img in X_digits])  # Shape: (1797, 28, 28)

# Convert to PyTorch tensors and add the channel dimension
X_digits_resized = torch.tensor(X_digits_resized).unsqueeze(1).float()  # Shape: (1797, 1, 28, 28)
y_digits = torch.tensor(y_digits).long()

# Create training and validation sets for the digits dataset
train_size = int(0.8 * len(X_digits_resized))
valid_size = len(X_digits_resized) - train_size
digits_train_dataset, digits_valid_dataset = random_split(
    TensorDataset(X_digits_resized, y_digits), [train_size, valid_size])

# Create data loaders for digits dataset
train_loader_digits = DataLoader(digits_train_dataset, batch_size=32, shuffle=True)
valid_loader_digits = DataLoader(digits_valid_dataset, batch_size=32, shuffle=False)

print(f"Digits dataset resized and prepared. Training samples: {train_size}, Validation samples: {valid_size}")

