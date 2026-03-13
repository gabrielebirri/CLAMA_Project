---
title: ADAS
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.9.0
python_version: "3.10" # Force python version to 3.10
app_file: app.py
pinned: false
---
# Advanced Dermatological Assistance System

### Disclaimer: this is a university project and it's NOT intended to be a diagnostic device.

## Update: ADAS is now available as a demo on Hugging Face Spaces!

To run a demo of ADAS on Hugging Face Spaces without the need to install any dependancy locally, click on the badge below:

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/bibri04/ADAShf)

## Abstract
The objective of this experimental research is to conduct a comparative analysis of diverse Convolutional Neural Network (CNN) architectures to assess their performance in the automated classification of cutaneous lesions as benign or malignant (melanoma). Given the inherent constraints of medical imaging datasets, a transfer learning methodology is adopted. Specifically, three prominent architectures, pre-trained on the ImageNet database, are employed as feature extractors to evaluate their diagnostic efficacy. To ensure experimental integrity and mitigate methodological biases, a standardized training and evaluation pipeline is maintained across all models.

The architectures are adapted for binary classification by substituting the original global classifier or fully connected layer with a task-specific head. The optimization follows a two-phase fine-tuning strategy: an initial stage where the convolutional backbone remains frozen to stabilize the new classifier's weights, followed by a selective unfreezing of deeper layers to facilitate domain-specific feature adaptation. This dual-stage approach is designed to enhance task specialization while preserving the generalization capabilities of the pre-trained weights. Furthermore, a grad-CAM technique is implemented to gain a sensible evidence of the spatial attention of the model.

The study utilizes a dataset of approximately 10,000 dermoscopic images, sourced from a publicly available repository and balanced between benign and malignant classes. The performance of the respective architectures is rigorously quantified using standard clinical evaluation metrics, providing a comprehensive perspective on the suitability of different CNN backbones for computer-aided diagnosis in dermatology.  

**Dataset used:** https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

## Model repository
All the trained models are available freely on Hugging Face Model Hub. To view and download the models click on the link below:

https://huggingface.co/spaces/bibri04/ADAS_models

A detailed explanation model performance and architecture (hyper-parameters, training strategy, etc.) can be found in the [model repository README](https://huggingface.co/spaces/bibri04/ADAS_models/blob/main/README.md) (It will be updated soon).

## How to run training and testing pipeline
The philosophy behind this project is that cancer fight is indeed a struggle that we, as humanity, need to endure together, and therefore this project is designed to be open source, in order for everybody to be able to contribute with knowledge and potentally game changing ideas. Moreover, given the fact that it is a medical environment there is the need for the pipeline to be accessible to be inspected in order to find potentially critical flaws. In this section an explanation of how to run the code will be given.

The pipeline is designed to be hybrid, in order to be able to both run locally and to leverage cloud computing power via Google Colab.

### HOW TO RUN LOCALLY

### 1. Check Python version
This project has been designed to be able to run on Python 3.10.19. If you have another version installed some libraries may not work. To check for the Python version run the following code in terminal:

<pre>
python --version
</pre>

### 2. Create and activate a virtual environment
It's best practice to create a virtual environment to avoid conflicts between libraries. The best way to do so is by downloading **Anaconda** and create an environment via conda. When you have created the environment (example: my_env), remember to activate it:

<pre>
conda activare my_env
</pre>

### 3. Install required dependancies
To install required libraries, run the following line in the terminal, it will automatically download and install all you need in the vortual environment:

<pre>
pip install -r requirements.txt
</pre>

### 4. Run the pipeline
This project is built with simplicity at its core. The entire workflow is managed through a Jupyter Notebook for an intuitive experience.To execute the training and testing pipeline, simply open **Pipeline_Driver.ipynb** and follow the step-by-step instructions provided within the cells.

### HOW TO RUN ON COLAB

The workflow to run the pipeline on Google Colab is much simpler and automated.

### 1. Open Pipeline_Driver.ipynb

### 2. Click on the Colab badge above the title

### 3. Run the notebook as locally

NOTE: sometimes Colab gives an error after downloading the dataset. If this happens wait for a minute and run the downloading cell again. This happens because the cache has still to update, therefore the following cells of the notebook don't find the dataset material yet

## Run the application locally

For privacy concerns, it is possible to run the gradio application locally. To do so, first clone the repository and install the dependancies as shown in the previous section, then run the following command in the terminal:

<pre>
gradio app.py
</pre>

This will open a local gradio application in the browser where you can test the model. Note that the first time you run the application it will take a few minutes to download the model weights from Hugging Face.
