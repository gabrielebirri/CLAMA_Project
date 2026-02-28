# CLAMA Project
Computer-Leveraged Analysis for Melanoma Assessment

<p align="left">
  <img src="logo.png" width="300" title="CLAMA Logo">
</p>

**Dataset used:** https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

## Abstract
The objective of this experimental research is to conduct a comparative analysis of diverse Convolutional Neural Network (CNN) architectures to assess their performance in the automated classification of cutaneous lesions as benign or malignant (melanoma). Given the inherent constraints of medical imaging datasets, a transfer learning methodology is adopted. Specifically, three prominent architectures, pre-trained on the ImageNet database, are employed as feature extractors to evaluate their diagnostic efficacy. To ensure experimental integrity and mitigate methodological biases, a standardized training and evaluation pipeline is maintained across all models.

The architectures are adapted for binary classification by substituting the original global classifier or fully connected layer with a task-specific head. The optimization follows a two-phase fine-tuning strategy: an initial stage where the convolutional backbone remains frozen to stabilize the new classifier's weights, followed by a selective unfreezing of deeper layers to facilitate domain-specific feature adaptation. This dual-stage approach is designed to enhance task specialization while preserving the generalization capabilities of the pre-trained weights. Furthermore, a grad-CAM technique is implemented to gain a sensible evidence of the spatial attention of the model.

The study utilizes a dataset of approximately 10,000 dermoscopic images, sourced from a publicly available repository and balanced between benign and malignant classes. The performance of the respective architectures is rigorously quantified using standard clinical evaluation metrics, providing a comprehensive perspective on the suitability of different CNN backbones for computer-aided diagnosis in dermatology.  
