import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.utils import build_chosen_model, imagenet_mean, imagenet_std
from src.testing import inference
from src.gradcam import grad_cam_setup, show_grad_cam

def analyze_image(image, selected_weights):
    if image is None:
        return "<strong style='color:red;'>Please upload an image first.</strong>", None
        
    model_options = {
        "effnet_2.pth": "EfficientNet",
        "densenet_best_5.pth": "DenseNet121"
    }
    
    model_name = model_options[selected_weights]
    weights_path = f"models/{selected_weights}"
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = build_chosen_model(model_name)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    # Pre-processing and Prediction
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    image_tensor = transform_pipeline(image).unsqueeze(0).to(device)

    sensitivity = 0.5
    
    prediction, prob = inference(model, image_tensor, sensitivity, return_prob=True)
    label = "MALIGNANT" if prediction == 1 else "BENIGN"
    
    color = "red" if label == "MALIGNANT" else "green"
    prob_percentage = f"{prob * 100:.2f}%"
    diagnosis_html = f"Suggested diagnosis: <strong style='color:{color}; font-size: 1.2em;'>{label}</strong><br>Probability: <strong>{prob_percentage}</strong>"

    # Grad-CAM Visualization
    cam = grad_cam_setup(weights_path, device)
    dummy_dataset = [(image_tensor.squeeze(0), 0)]
    
    fig = plt.figure(figsize=(10, 5))
    original_show = plt.show
    plt.show = lambda: None  # Prevent plt.show() from blocking or clearing the figure in Gradio
    
    try:
        show_grad_cam(0, dummy_dataset, cam, device, prediction=label.capitalize())
    finally:
        plt.show = original_show
        
    return diagnosis_html, fig

with gr.Blocks(title="ADAS") as demo:
    gr.Markdown("# ADAS")
    gr.Markdown("## Advanced Dermatological Assistance System")
    gr.Markdown("Deep Learning for melanoma detection")
    
    gr.HTML("""
    <div style="background-color: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 8px; color: #b71c1c; margin-bottom: 20px;">
        ⚠️ <strong style='color: red;'>Disclaimer:</strong> This application is a university project developed for educational purposes. It is <strong style='color: red;'>NOT</strong> a medical diagnostic device and should not be used for clinical decision-making or diagnosis.
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model configuration")
            model_dropdown = gr.Dropdown(
                choices=["effnet_2.pth", "densenet_best_5.pth"],
                value="effnet_2.pth",
                label="Select the Model"
            )
            image_input = gr.Image(type="pil", label="Upload a dermatoscopic image (JPG/PNG)")
            analyze_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            diagnosis_output = gr.HTML()
            gr.Markdown("### Grad-CAM Visualization")
            gradcam_output = gr.Plot()
            
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, model_dropdown],
        outputs=[diagnosis_output, gradcam_output]
    )

if __name__ == "__main__":
    demo.launch()