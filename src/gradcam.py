from src.utils import build_densenet, build_efficientnet, build_resnet
from src.utils import im_show, imagenet_mean, imagenet_std
import torch
import numpy as np
import matplotlib.pyplot as plt


# Grad-CAM modules
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#-----------------------------------#
# Set up Grad-CAM for a specified model.
# This function loads a pre-trained EfficientNet model from the given path,
# assigns it to the specified device (CPU/GPU), and attaches a Grad-CAM hook 
# to its final convolutional layer to generate class activation maps.

def grad_cam_setup(load_path, device):
    # Setup of the grad-CAM

    load_path_str = str(load_path).lower()
    
    # Infer the correct model and target layer based on the filename
    if "densenet" in load_path_str:
        cam_model = build_densenet()
        target_layers = [cam_model.features[-1]]
    elif "resnet" in load_path_str:
        cam_model = build_resnet()
        target_layers = [cam_model.layer4[-1]]
    else: 
        cam_model = build_efficientnet()
        target_layers = [cam_model.features[-1]]

    # Load weights
    cam_model.load_state_dict(torch.load(load_path, map_location=device))
    cam_model.to(device)
    cam_model.eval()
    print(f"CAM model: {load_path}")        # Here the same path as the test load is used

    cam = GradCAM(model=cam_model, target_layers=target_layers)
    print("CAM set up correctly")
    return cam


#-----------------------------------#
# Overlay the Grad-CAM heatmap onto a selected image and display it.
# This function processes a specific image from the dataset, computes its 
# Grad-CAM heatmap using the provided `cam` object (targeting the 'Benign' class), 
# and displays a side-by-side matplotlib comparison between the original image 
# and the Grad-CAM visualization. It can optionally display the model's prediction.

def show_grad_cam(image_index, test_dataset, cam, device, prediction=None):

    cam_tensor, cam_label = test_dataset[image_index]
    cam_tensor = cam_tensor.to(device)

    if cam_label == 0:
        print("Actual label: Benign")
    else:
        print("Actual label: Malignant")

    cam_tensor = cam_tensor.unsqueeze(0)

    # HERE WAS EVALUATION

    targets = [ClassifierOutputTarget(0)]   # class identification (this problem there is only 1, because the classfier is binary)

    grayscale_cam = cam(input_tensor=cam_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Image processing
    img_for_visualization = cam_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_for_visualization = (img_for_visualization * imagenet_std) + imagenet_mean  # This denormalizes the image
    img_for_visualization = np.clip(img_for_visualization, 0, 1)

    # Visualization of the results
    visualization = show_cam_on_image(img_for_visualization, grayscale_cam, use_rgb=True, image_weight=0.7)

    plt.subplot(1,2,1)
    im_show(cam_tensor)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1,2,2)
    plt.imshow(visualization)
    plt.axis('off')
    if prediction:
        plt.title(f'Grad-CAM\nPrediction: {prediction}')
    else:
        plt.title('Grad-CAM')
    plt.show()