# Smiling or Not: Face Classification using Deep Learning

## Overview
This project demonstrates the application of deep learning techniques, specifically Convolutional Neural Networks (CNNs), to classify facial images into two categories: Smiling or Not Smiling. The assignment aims to showcase skills in data science, data preprocessing, regression, machine learning, Python programming, and deep learning.


## Key Features
- Utilizes the CelebFaces Attributes (CelebA) dataset containing 200,000 images with 40 attributes.
- Employs PySpark, SKLearn, TensorFlow, and Keras for data handling, exploration, and modeling.
- Implements a custom pipeline to streamline data processing, model training, and evaluation.
- Experiments with multiple models, including transfer learning, for comparison.
- Includes a user-input feature to classify a new image.


## Project Structure
The workflow is divided into seven major parts:

1. **Setup**:
   - Imports required libraries (PySpark, TensorFlow, Keras, SKLearn, Matplotlib, etc.).
   - Configures environment variables to resolve PySpark and Python compatibility issues.
   - Initializes a PySpark session for data handling.

2. **Initial Data Loading and Preprocessing**:
   - Retrieves file paths for images and labels from the dataset’s CSV file.
   - Merges image paths and labels into a single DataFrame.
   - Performs basic preprocessing, including data formatting, cleaning, and integration.

3. **Exploratory Data Analysis (EDA)**:
   - Ensures equal distribution of output class labels.
   - Visualizes data with bar charts and verifies labels by displaying random images.
   - Highlights dataset insights and validates the integrity of loaded data.

4. **Pipeline Creation**:
   - Develops a Python-based pipeline for end-to-end automation of the analysis.
   - Handles tasks such as data formatting, model creation, training, evaluation, and result generation.

5. **Using the Pipeline**:
   - Simplifies model analysis with a single function call.
   - Tests multiple models by passing them through the pipeline, enabling efficient evaluation and comparison.

6. **Experimenting with Alternative Models**:
   - Includes transfer learning for comparison.
   - Evaluates its computational cost and performance trade-offs against standard models.

7. **User Input**:
   - Allows users to upload an image, which is classified as either Smiling or Not Smiling.
   - Provides an intuitive interface for end-user interaction.


## Dataset
**Title**: CelebFaces Attributes (CelebA) Dataset  
**Author**: Jessica Li  
**Date**: 2018  
**Link**: [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

- Contains 200,000 images of celebrities with annotations for 40 attributes.
- Used 100,000 images for training and evaluation due to computational constraints.
- Recommended to reduce data size to 10,000 or 1,000 images for testing on standard hardware.


## Technical Details
- **Development Environment**:
  - Visual Studio Code with Jupyter Notebook extension.
  - Anaconda3 (Python 3.11.7).

- **Primary Dependencies**:
  - TensorFlow
  - PySpark
  - Keras
  - SKLearn
  - Matplotlib/Seaborn (for visualization)


## Key Insights
- **Pipeline Efficiency**: Automating the analysis reduces manual intervention and ensures consistency.
- **Model Versatility**: Enables easy experimentation with different architectures and hyperparameters.
- **Transfer Learning**: Demonstrates potential for improvement with pretrained models, though computational cost is higher.
- **Data Visualization**: Graphical evaluation aids in understanding model performance and tuning.
