# Leaf Disease Detection and Classification

## Project Overview

This project focuses on automating the detection and classification of diseases in plant leaves, specifically targeting corn crops. The traditional manual method of disease detection is time-consuming, costly, and unreliable. Therefore, this project employs Convolutional Neural Networks (CNNs) for automatic detection using image processing techniques.

## Authors

- **Samhitha Suda**
  - *Department of Electronics and Communication*
  - *Sreenidhi Institute of Science and Technology, Hyderabad, India*
  - *Email: sudasamhitha@gmail.com*

- **Dr. C N Sujatha**
  - *Department of Electronics and Communication*
  - *Sreenidhi Institute of Science and Technology, Hyderabad, India*
  - *Email: cnsujatha@sreenidhi.edu.in*

## Abstract

Pests and diseases significantly impact agricultural production. This project addresses the need for efficient plant disease detection using deep learning techniques. It employs Convolutional Neural Networks, specifically the VGGNet architecture, and utilizes Tensorflow and Keras for model training. The system achieves approximately 98% accuracy for corn crops, contributing to the automation of disease detection in the agricultural sector.

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Survey](#literature-survey)
3. [Required Technologies & Software](#required-technologies--software)
4. [Proposed System](#proposed-system)
5. [Design and Implementation](#design-and-implementation)
6. [Results and Discussion](#results-and-discussion)
7. [Conclusion](#conclusion)
8. [Running Commands](#running-commands)
9. [File Descriptions](#file-descriptions)

## Introduction

India is the second-largest agricultural producer globally, and agriculture plays a crucial role in its economy. However, plant diseases pose a threat to crop yield. The project aims to automate the detection of diseases in corn crops using deep learning, specifically CNNs. The proposed system can be a valuable tool for early disease identification, aiding in effective disease prevention and management.

## Literature Survey

The project is grounded in a comprehensive review of existing research on plant disease detection. Various models, including CNNs, have been explored for their effectiveness in identifying and diagnosing plant diseases. The literature survey provides insights into the advancements and challenges in the field.

## Required Technologies & Software

- **Deep Learning:** Utilizing multiple nonlinear transformations for top-level abstractions in data through the use of model architectures.
- **Convolutional Neural Networks (CNN):** Specifically, the VGGNet architecture is employed for object recognition.
- **Object Detection:** A wide field of deep learning used for real-time object recognition.
- **Tensorflow:** An open-source software library for numerical computation with high accuracy.
- **Keras:** A high-level neural network API running on Tensorflow, Theano, and CNTK.
- **Anaconda:** An open-source distribution of R and Python programming languages.
- **Python:** A multi-paradigm, general-purpose, high-level programming language.
- **VGGNet:** A deep convolutional network used for object recognition.

## Proposed System

The system involves several stages, including image acquisition, pre-processing, segmentation, feature extraction, and training. Deep learning models, specifically CNNs, are trained to map inputs to outputs based on a set of training data. The proposed system achieves accurate disease detection, contributing to the agricultural sector's efficiency.

## Design and Implementation

The design and implementation include steps such as image acquisition, pre-processing, segmentation, feature extraction, and training. The accuracy of the model is measured, and the results indicate the successful detection of corn leaf diseases with high precision.

## Results and Discussion

The system demonstrates successful disease detection with an accuracy of approximately 98%. Images before and after segmentation showcase the effectiveness of the image processing techniques employed. The discussion highlights the potential applications and improvements for future development.

## Conclusion

The proposed system provides an automated solution for the early detection and classification of plant diseases, focusing on corn crops. With a high accuracy rate, the system can contribute to the agricultural sector by assisting farmers in timely disease management. Further enhancements, including real-time implementation and deployment on drones, are suggested for future work.

## Running Commands

Assuming that required dependencies and libraries are installed:

1. **Generate Marker:**
    ```bash
    python generate_marker.py
    ```

2. **Main Program:**
    ```bash
    python main.py
    ```

3. **Segmentation:**
    ```bash
    python segment.py
    ```

## File Descriptions

1. **generate_marker.py:**
   - This script is responsible for generating markers used in the image processing pipeline.

2. **main.py:**
   - The main program that executes the disease detection and classification using CNNs.

3. **segment.py:**
   - Contains code for image segmentation, a crucial step in the preprocessing pipeline.

4. **generate_marker.txt, main.txt, segment.txt:**
   - Text files containing additional documentation or instructions for the corresponding scripts.

