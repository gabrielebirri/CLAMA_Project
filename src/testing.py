import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy

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
    test_model.eval()

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
# This function is used to inference the model given an image
def inference(model, image, sensitivity):
    pred = model(image)
    prob = torch.sigmoid(pred)
    if prob > sensitivity:
        print("Model prediction: Malignant")
    else:
        print("Model prediction: Benign")