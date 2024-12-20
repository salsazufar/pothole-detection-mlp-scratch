# Pothole Detection Project

This project implements a pipeline for detecting potholes using a dataset from Kaggle. The notebook includes data preprocessing, training a machine learning model (MLP) from scratch, and evaluating the model's performance. Below are the steps and features of the notebook.

## Features
- Downloading and extracting datasets from Kaggle.
- Data preprocessing and visualization.
- Implementing a Multilayer Perceptron (MLP) network for pothole detection.
- Functions to save and load trained models.
- Training the model and evaluating its accuracy.
- Performing inference on test data.

## Requirements
- Python 3.x
- Kaggle API key for dataset download.
- Necessary libraries listed in `requirements.txt`.

## Setup and Usage

1. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Kaggle API:**
   - Place your `kaggle.json` API key file in the appropriate directory (`~/.kaggle/`).
   - Set the necessary file permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Run the notebook:**
   Open and execute the notebook using Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook notebook.ipynb
   ```

4. **Dataset download and extraction:**
   - The notebook will automatically download and extract the required dataset using Kaggle API.

5. **Train the model:**
   Follow the steps in the notebook to preprocess data, train the MLP model, and save it for future use.

6. **Evaluate and test:**
   Evaluate the trained model and perform inference on new data.

## Dataset
The dataset used in this project is the "Pothole Detection Dataset" available on Kaggle.

## Notes
Ensure you have a Kaggle API key set up correctly to download the dataset without issues.


