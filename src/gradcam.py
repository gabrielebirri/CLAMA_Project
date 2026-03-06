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
# This function sets up the grad cam to be used by a choosen model

def grad_cam_setup(load_path, device):
    # Setup of the grad-CAM

    # Defining model to be used in grad-CAM
    cam_model = build_efficientnet()
    cam_model.load_state_dict(torch.load(load_path, map_location=device))
    cam_model.to(device)
    print(f"CAM model: {load_path}")        # Here the same path as the test load is used

    target_layers = [cam_model.features[-1]]

    cam = GradCAM(model=cam_model, target_layers=target_layers)
    print("CAM set up correctly")
    return cam


#-----------------------------------#
# This function shows the grad-cam onto a selected image

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