import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.utils import build_chosen_model, imagenet_mean, imagenet_std
from src.testing import inference
from src.gradcam import grad_cam_setup, show_grad_cam
from huggingface_hub import hf_hub_download

# Hugging Face Repository Configuration
# Repo ID
HF_REPO_ID = "bibri04/ADAS_models"

def analyze_image(image, selected_weights):
    if image is None:
        return "<strong style='color:red;'>Please upload an image first.</strong>", None
        
    model_options = {
        "effnet_2.pth": "EfficientNet",
        "densenet_best_5.pth": "DenseNet121"
    }
    
    model_name = model_options[selected_weights]
    
    try:
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=selected_weights, local_dir="models")
    except Exception as e:
        return f"<strong style='color:red;'>Failed to download model weights from Hugging Face: {e}</strong>", None
    
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
    gr.Markdown("# Advanced Dermatological Assistance System")
    gr.Markdown("### Developed by Gabriele Birri")
    
    gr.HTML("""
    <div style="background-color: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 8px; color: #b71c1c; margin-bottom: 20px;">
        ⚠️ <strong style="color: inherit;">Disclaimer:</strong> This tool is a university project for educational and research purposes only. It is <strong style="color: inherit;">NOT</strong> a medical diagnostic tool. The AI predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a dermatologist or other qualified health provider with any questions regarding a medical condition.
    </div>
    
    <div style="margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.1em;">Project repository:</span>
        <a href="https://github.com/gabrielebirri/ADAS" target="_blank" style="text-decoration: none;">
            <button style="background-color: #24292e; color: white; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 1em; font-weight: bold; display: flex; align-items: center; gap: 8px;">
                <svg style="width: 16px; height: 16px; fill: white;" viewBox="0 0 16 16" version="1.1" aria-hidden="true">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
                ADAS
            </button>
        </a>
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
            gr.Markdown("### Grad-CAM")
            gr.Markdown("<i>Grad-CAM is a technique that helps to visualize the regions of the image that the model is focusing on to make its prediction.</i>")
            gradcam_output = gr.Plot()
            
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, model_dropdown],
        outputs=[diagnosis_output, gradcam_output]
    )

if __name__ == "__main__":
    demo.launch()