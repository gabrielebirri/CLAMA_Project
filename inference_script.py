"""
Project name:   ADAS
Descrtption:    This module provides a fast and easy way to call and inference a model for a given image
                The currently supported models are only the ones on which this projects focuses on.
                Further implementations may bring compatibility to other model types.

Autore:         Gabriele Birri
Creation date:  5th March 2026
Version:        1.0.0
"""

# Importing dependancies
import torch
from torchvision import transforms, models
from src.utils import imagenet_mean, imagenet_std
from PIL import Image
from pathlib import Path

from src.testing import inference
from src.utils import build_chosen_model
from src.gradcam import grad_cam_setup, show_grad_cam
from src.utils import valid_models


# Setting device agnostic code
if torch.backends.mps.is_available():
    device = 'mps' 
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Pre processing of the image
image_path = './test_image.jpg'
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

img_pil = Image.open(image_path).convert('RGB')

image = transform_pipeline(img_pil)
image = image.unsqueeze(0)
image = image.to(device)


# Loading the model and the weights

print("Please choose one of the following model types:")
print("DenseNet121, EfficientNet, ResNet50")

"""
while True:
    name = input()

    if name in valid_models:
        break
    else:
        print("Please enter a valid name: ")
"""

name = "DenseNet121"
model = build_chosen_model(name)

# Defining paths

models_path = Path('./models')
models_path.mkdir(parents=True, exist_ok=True)


# model_name = input("Please enter the name of the model:")

model_name = "densenet_best_5.pth"
load_path = models_path / model_name



# Loading weights

model.load_state_dict(torch.load(load_path, map_location=device))
model.to(device)
print(f"current model: {load_path}")

# inference

sensitivity = 0.4
prediction = inference(model, image, sensitivity)

# Grad-CAM
# This section implements the grad cam for the choosen image
from src.gradcam import grad_cam_setup, show_grad_cam

cam = grad_cam_setup(load_path, device)

# Wraps the single image in a dummy dataset-like list to match show_grad_cam's expected input format
dummy_dataset = [(image.squeeze(0), 0)]
image_index = 0

show_grad_cam(image_index, dummy_dataset, cam, device, prediction=prediction)