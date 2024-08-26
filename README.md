# Deep Learning Model for Lung Cancer Detection and Segmentation

## Project3-Group 5: Cancer_Champions

## Group Members
* Tabibzadeh, Oliver
* Hessou, S. Prudence
* Dai, Xiwu
* Shen, Zhongzhe
* Qayyum, Wassam

## Project Overview
This project focuses on leveraging deep learning for the segmentation of lung cancer images using the U-Net architecture, a widely adopted neural network for biomedical image segmentation. U-Net addresses the challenges posed by limited annotated data in the medical field, efficiently segmenting complex medical images with fewer samples while maintaining high performance in both speed and accuracy. The project encompasses the entire deep learning workflow, from data preprocessing and augmentation to model training, evaluation, and fine-tuning.

## Goal
The overarching goal is to develop a robust image processing system for lung cancer detection that adheres to data privacy regulations, such as HIPAA and GDPR, while ensuring ethical handling of medical data. We progressively experimented with the dataset size, number of epochs, and batch sizes to optimize the model's performance. Evaluation metrics such as Dice coefficient, Intersection over Union (IoU), and accuracy were used to assess the model's reliability and accuracy in segmenting cancerous regions in medical images. This work contributes to advancing lung cancer detection methodologies, which could potentially assist medical professionals in their diagnosis efforts.

## Background

### Motivation
Lung cancer remains one of the deadliest cancers worldwide, with survival rates significantly improving when diagnosed at early stages. However, accurate and timely detection is often hindered by the complexity of medical images, variability in tumor appearance, and the time-consuming nature of manual diagnosis by radiologists. The motivation behind our project is to leverage the power of deep learning, particularly the U-Net architecture, to create a tool that aids in the precise segmentation of lung cancer from medical images.

By automating the segmentation process, our goal is to improve the efficiency and accuracy of lung cancer detection, assisting healthcare professionals in making faster, more informed decisions. This project seeks to bridge the gap between limited annotated medical data and the growing need for robust diagnostic tools that can operate with fewer resources, while also ensuring that patient data is handled with the utmost care and compliance with privacy regulations. Ultimately, we hope our work will contribute to the broader fight against lung cancer, enabling earlier interventions and better patient outcomes.

### What is U-Net?
The U-Net neural network, introduced by Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation," is a deep learning architecture designed specifically for semantic segmentation in biomedical imaging. Unlike traditional models that require large annotated datasets, U-Net is optimized to perform well with limited training data, a common issue in the medical field. Its architecture consists of a contracting path that compresses the input image to capture context, followed by an expansive path that restores the image’s resolution for precise segmentation.

A key feature of U-Net is its use of skip connections between the contracting and expansive paths, which allows the network to preserve spatial information critical for accurate segmentation. These connections enable the network to combine both global context and detailed localization in a highly efficient manner, leading to improved performance in segmenting complex medical images. This makes U-Net particularly well-suited for applications in medical image processing, such as detecting and segmenting cancerous tissues from scans.

## Method
* Activation Function:
- **Chosen**: Sigmoid
- **Reason**: Achieved better accuracy for binary segmentation tasks, despite Softmax yielding smaller loss values.

* Optimizer:
- **Chosen**: Adam with Adaptive Learning Rates
- **Reason**: Adaptively adjusts the learning rate based on gradients to improve model stability and performance.

* Hyperparameter Tuning:
- **Tuning Tool**: Keras Tuner
- **Key Hyperparameters**:
- 1. Learning rate
  2. Number of filters
  3. Kernel size
  4. Dropout rate
  5. Batch size
  6. Number of epochs
- **Strategy**: Used learning rate decay and warm-up techniques to stabilize training.
  
* Metrics:
- **Dice Coefficient**: 0.0018
- **IoU**: 0.0009
- **Sensitivity**: 0.0018

* Model:
- **Architecture**: U-Net
- **Reason**: Effective for segmentation tasks with limited annotated data, using skip connections to capture both context and precise localization.
