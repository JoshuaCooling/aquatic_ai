"""
dnet.py

This script trains a DenseNet-121 convolutional neural network using PyTorch to classify underwater aquatic plants,
including invasive species, based on a labeled image dataset. It handles loading and transforming image data, training
a modified DenseNet-121 model, evaluating accuracy, and generating a confusion matrix and classification report.

Main Features:
- Loads a dataset from ImageFolder format (organized by class directories)
- Splits data into training (80%) and testing (20%) sets
- Applies standard image transforms (resize, normalize)
- Fine-tunes a pre-trained DenseNet-121 for multi-class classification
- Evaluates the model and generates visual and text-based reports
- Saves the final trained model as `invasive_species_model.pth`

Dependencies:
- torch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn

Author: Joshua Cooling
Date: April 2025
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import DenseNet121_Weights

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=r'C:\Users\coolj\OneDrive\Documents\Winter25\Capstone\total', transform=transform)

# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create Train and Test Datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained DenseNet model
weights = DenseNet121_Weights.IMAGENET1K_V1
model = models.densenet121(weights=weights)

# Modify classifier for custom dataset
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

# Move model to GPU if available
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

print("Training complete!")

# Save trained model
model_path = "invasive_species_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Function to evaluate the model on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Run evaluation
test_accuracy = evaluate_model(model, test_loader)

# Confusion Matrix Function
def plot_confusion_matrix(model, test_loader, class_names, save_path='confusion_matrix.png'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()

# Compute and display confusion matrix
plot_confusion_matrix(model, test_loader, dataset.classes)

# Save and display classification report
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_report = classification_report(all_labels, all_preds, target_names=dataset.classes)
with open("classification_report.txt", "w") as f:
    f.write(class_report)

print("\nClassification Report:\n")
print(class_report)
