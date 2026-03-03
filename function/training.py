import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import copy

# PyTorch
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

#-----------------------------------#
# This file contains the training function for the two phases of the praining pipeline

# TRAINING LOOP
def training_loop(model, train_loader, val_loader, epochs, threshold, criterion, optimizer, device):
    # TRAINING BENCHMARKS
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []

    # VALIDATION BENCHMARKS
    val_loss = []
    val_acc = []
    val_precision = []
    val_recall = []


    for epoch in range(epochs):
        model.train()

        correct, total = 0, 0       # Track correct predictions and total samples
        running_loss = 0.0          # Track the running loss for the training step
        all_train_preds = []        # List to store all predictions for precision and recall during training
        all_train_labels = []       # List to store all true labels for precision and recall during training

        # Progress bar using tqdm
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            labels = labels.float().unsqueeze(1)

            outputs = model(images) # Gets model predictions

            loss = criterion(outputs, labels)
            optimizer.zero_grad()   # Reset gradients from the previous step
            loss.backward()         # Compute gradients
            optimizer.step()        # Update model parameters

            # Applying sigmoid to outputs and setting threshold as indicated
            preds = (torch.sigmoid(outputs) > threshold).int().squeeze(1)

            # Update the total number of samples and the number of correct predictions
            total += labels.size(0)
            correct += (preds == labels.squeeze(1).int()).sum().item()

            # Add the loss for this batch to the running loss
            running_loss += loss.item()

            # Store predictions and true labels for precision and recall calculation
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.squeeze(1).int().cpu().numpy())


        # Training loss, accuracy, precision, and recall for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_acc.append(correct / total)
        train_loss.append(avg_train_loss)

        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=1)
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=1)

        # VALIDATION
        model.eval()                # Set the model to evaluation mode (no gradient calculation)

        correct, total = 0, 0       # Reset correct and total for validation
        running_val_loss = 0.0      # Track the validation loss
        all_val_preds = []          # List to store all predictions for precision and recall during validation
        all_val_labels = []         # List to store all true labels for precision and recall during validation


        # Disabling the gradient
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

    
                labels = labels.float().unsqueeze(1)
                outputs = model(images)

                # Calculate the loss for the validation
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Apply sigmoid to outputs and then threshold as indicated
                preds = (torch.sigmoid(outputs) > threshold).int().squeeze(1)

                total += labels.size(0)
                correct += (preds == labels.squeeze(1).int()).sum().item()

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.squeeze(1).int().cpu().numpy())

        # Validation loss, accuracy, precision, and recall
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc.append(correct / total)
        val_loss.append(avg_val_loss)

        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=1)
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=1)


        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc[-1]:.2f}, Train Precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc[-1]:.2f}, Val Precision: {val_precision:.2f}, Val Recall: {val_recall:.2f}")
        print(f"\n")
