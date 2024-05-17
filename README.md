# K-NN Hyperparameter Tuning and Analysis

This repository contains a Jupyter notebook that demonstrates hyperparameter tuning for the k-Nearest Neighbors (K-NN) algorithm. The notebook includes data preprocessing, model training, hyperparameter tuning, and visualization of results.

## Files

- `lab.ipynb`: Jupyter notebook with the complete code for K-NN hyperparameter tuning and analysis.

## Overview

The notebook performs the following tasks:

1. **Data Loading and Preprocessing**: 
   - Loads the dataset.
   - Performs necessary preprocessing steps such as handling missing values, encoding categorical variables, and scaling features.

2. **Model Training**:
   - Splits the data into training and testing sets.
   - Trains the K-NN model using different hyperparameters.

3. **Hyperparameter Tuning**:
   - Utilizes techniques like Grid Search and Random Search to find the optimal hyperparameters for the K-NN model.

4. **Evaluation**:
   - Evaluates the model using metrics like accuracy.
   - Plots training and testing accuracy to visualize the effect of varying the number of neighbors.

5. **Visualization**:
   - Generates plots to show the performance of the K-NN model with different hyperparameters.

## Dependencies

The notebook requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
