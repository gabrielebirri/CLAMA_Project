import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.utils import build_chosen_model, im_show, imagenet_mean, imagenet_std
from src.testing import inference
from src.gradcam import grad_cam_setup, show_grad_cam

st.set_page_config(page_title="ADAS", layout="wide")

st.title("Advanced Dermatological Assistance System")
st.write("Deep Learning for skin cancer detection")

st.error("⚠️ **Disclaimer:** This application is a university project developed for educational purposes. It is **not** a medical diagnostic device and should not be used for clinical decision-making or diagnosis.")


# --- SIDEBAR: Configurazione ---
st.sidebar.header("Model configuration")

model_options = {
    "effnet_2.pth": "EfficientNet",
    "densenet_best_5.pth": "DenseNet121"
}

selected_weights = st.sidebar.selectbox(
    "Select the Model",
    list(model_options.keys())
)

model_name = model_options[selected_weights]
weights_path = f"models/{selected_weights}"

# Loading image
uploaded_file = st.file_uploader("Upload a dermatoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    if st.button("Analyze"):
        with st.spinner('Analyzing...'):
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                
            model = build_chosen_model(model_name)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)

            # 2. Pre-processing e Predizione
            transform_pipeline = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
            image_tensor = transform_pipeline(image).unsqueeze(0).to(device)

            sensitivity = 0.2
            
            prediction = inference(model, image_tensor, sensitivity)
            label = "MALIGNANT" if prediction == "Malignant" else "BENIGN"

            st.subheader("Results")
            color = "red" if label == "MALIGNANT" else "green"
            st.markdown(f"Suggested diagnosis: **:{color}[{label}]**")

            # 3. Visualizzazione Grad-CAM
            st.subheader("Grad-CAM Visualization")
            
            with st.spinner("Generating Grad-CAM..."):
                cam = grad_cam_setup(weights_path, device)
                dummy_dataset = [(image_tensor.squeeze(0), 0)]
                
                fig = plt.figure(figsize=(10, 5))
                original_show = plt.show
                plt.show = lambda: None  # Evita che plt.show() blocchi o pulisca la figura in Streamlit
                
                try:
                    show_grad_cam(0, dummy_dataset, cam, device, prediction=prediction)
                    st.pyplot(fig)
                finally:
                    plt.show = original_show
                    plt.close(fig)