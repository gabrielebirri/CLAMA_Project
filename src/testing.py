import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy
import torch

# PyTorch
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

#-----------------------------------#
# This function is used to test the model on the test dataset
def testing_model(test_model, test_loader, threshold, criterion, device):
    test_model.eval()                   # Set the model to evaluation mode

    test_loss = 0
    correct = 0
    total = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = test_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            true_positives += ((preds == 1) & (labels == 1)).sum().item()
            true_negatives += ((preds == 0) & (labels == 0)).sum().item()
            false_positives += ((preds == 1) & (labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Model tested")
    return true_positives, true_negatives, false_positives, false_negatives, all_preds, all_labels, test_loss, correct, total


#-----------------------------------#
# This function is used to inference the model for a given image
def inference(model, image, sensitivity, return_prob=False):
    model.eval()                        # Set the model to evaluation mode
    with torch.inference_mode():        # Set the context for inference
        pred = model(image)
        prob = torch.sigmoid(pred)
        if prob > sensitivity:
            prediction = 1
        else:
            prediction = 0
        
        # Converting prediction to label
        if prediction == 1:
            label = "Malignant"
        else:
            label = "Benign"
    print(f"Model prediction: {label}")
    if return_prob:
        return prediction, prob.item()
    return prediction