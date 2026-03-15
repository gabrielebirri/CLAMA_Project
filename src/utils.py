import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy
import shutil


# PyTorch
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

#-----------------------------------#
# MODELS DITCIONARY
model_types = {
    "effnet_3.pth": "EfficientNet",
    "effnet_2.pth": "EfficientNet",
    "densenet_best_5.pth": "DenseNet121",
}
models_names = list(model_types.keys())
default_model = "effnet_3.pth"

# List of valid models in this project
valid_models = {"DenseNet121", "EfficientNet", "ResNet50"}

#-----------------------------------#
# Mean and standard deviation for imagenet

imagenet_mean = [0.485, 0.456, 0.406]   # mean used in model training
imagenet_std = [0.229, 0.224, 0.225]    # standard deviation used in model training

#-----------------------------------#
# This function helps to plot a normalized image

def im_show(tensor):
    img = tensor.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))
    img = np.clip(img * imagenet_std + imagenet_mean, 0, 1)
    plt.imshow(img)
    plt.axis('off')

#-----------------------------------#
# This function freezes the backbone of the model

def freeze_backbone(model):
    # First all the parameters of the model are being freezed
    for param in model.parameters():
        param.requires_grad = False

    # Only the last layer is unfreezed (classifier or fc)
    # The pipeline is studied to be compatible with the most known models for medical imaging
    # NOTE: if the model is not supported you need to freeze and unfreeze the layers manually
    # This operation depends on the architecture of the model

    if hasattr(model, 'fc'):
        # Case: ResNet, Inception
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Backbone freezed. Layer 'fc' active.")

    elif hasattr(model, 'classifier'):
        # Case: VGG, DenseNet, EfficientNet
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("Backbone freezed. Layer 'classifier' active.")

    else:
        print("Unable to automatically recognize model layers")

#-----------------------------------#
# This function unfreezes the backbone of the model

def unfreeze_backbone(model):
    # All the parameters are unfreezed
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfreezed")

#-----------------------------------#
# This functions build the various models

def build_densenet():
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, 1)   # Initializing the classifier to be binary
    return model

def build_resnet():
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 1)   # Initializing the fc layer to be binary
    return model

def build_efficientnet():
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    # Initializing the classifier to be binary
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


#-----------------------------------#
# This function download the dataset, discriminating between local environment and cloud
def download_dataset(IN_COLAB):
    if IN_COLAB:
        print("Detected environment: Colab")
        # Downloading the dataset from Kaggle
        root = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
        train_path = '/kaggle/input/melanoma-skin-cancer-dataset-of-10000-images/melanoma_cancer_dataset/train'
        test_path = '/kaggle/input/melanoma-skin-cancer-dataset-of-10000-images/melanoma_cancer_dataset/test'

    else:
        print("Detected environment: Local")
        root = kagglehub.dataset_download("hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")
        pre = os.path.join(root, "melanoma_cancer_dataset")
        train_path = os.path.join(pre, "train")
        test_path = os.path.join(pre, "test")
        
    return train_path, test_path

#-----------------------------------#
# This function helps choosing the model and building it
# This function is designed to be modular so if a new model is added, it can be quickly implemented
def build_chosen_model(model_name):
    # Downloading and preparing the model
    if model_name == "DenseNet121":
        model = build_densenet()
        print(f"Model: {model_name} \nStatus: Loaded correctly")

    elif model_name == "ResNet50":
        model = build_resnet()
        print(f"Model: {model_name} \nStatus: Loaded correctly")

    elif model_name == "EfficientNet":
        model = build_efficientnet()
        print(f"Model: {model_name}_v2_s \nStatus: Loaded correctly")
    return model

