# GasTurbine-PM
## Gas Turbine Frigate Simulator Predictive Maintenance
This repository contains the code and documentation for a predictive maintenance analysis of gas turbine systems on a frigate simulator, utilizing machine learning models to predict the health of the compressor (kMc) and turbine (kMt). The project leverages a dataset from the UCI Machine Learning Repository and delivers models with high accuracy (98.53% for kMc) and R² (99.05% for kMt), ready for deployment in real-world maintenance systems.

## Project Overview
The goal was to develop a robust predictive maintenance system using sensor data from gas turbines. The analysis includes data preprocessing, feature selection, model training with GradientBoostingClassifier and RandomForestRegressor, and evaluation metrics such as accuracy, confusion matrix, and R². The final models are saved for deployment, with a focus on actionable health status predictions.

#### kMc Model: 
Classifies compressor health with 98.53% accuracy.
#### kMt Model: 
Predicts turbine health with an R² of 0.9905.
#### Dataset: 
UCI CBM Dataset (Condition-Based Maintenance).
Features
Data loading and validation from a whitespace-delimited text file.
Feature scaling and polynomial feature engineering for enhanced modeling.
Hyperparameter-tuned machine learning models.
Cross-validation for model robustness.
Health status generation for maintenance decisions.
Saved model artifacts for deployment.
Installation
To run the project locally, follow these steps:

Clone the Repository:
bash

git clone https://github.com/Ekom-obong/gas-turbine-frigate-simulator.git
cd gas-turbine-frigate-simulator
Install Dependencies: Ensure you have Python 3.11 or later installed. Then, install the required packages:
bash

#### pip install -r requirements.txt
#### The requirements.txt file should include:
#### text

pandas==2.0.3
numpy==1.26.4
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.2
joblib==1.3.2
Download the Dataset:
Obtain the dataset from the UCI Machine Learning Repository.
#### Run the Notebook:
Open Gas Turbine Frigate Simulator.ipynb in Jupyter Notebook or JupyterLab and execute the cells sequentially.
## Usage
#### Exploratory Analysis: 
Review the initial cells for data validation and statistics.
#### Model Training:
Execute the feature selection and model training sections to reproduce the results.
#### Evaluation: 
Check the evaluation metrics (accuracy, R², confusion matrix) and health status predictions.
#### Deployment: 
Use the saved models (clf_kmc_model.pkl, rf_kmt_model.pkl, etc.) for integration into a production environment.
## Results
### kMc Model:
#### Accuracy: 
0.9853
#### Confusion Matrix:
[[1859, 13], [22, 493]]
#### Cross-Validation Accuracy:
0.9802 (±0.0061)
### kMt Model:
#### Best Parameters:
{'n_estimators': 250, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 30}
#### Feature Importances: 
{'GTT': 0.1204, 'GTn': 0.0854, 'GGn': 0.1557, 'P2': 0.4831, 'Pexh': 0.0859, 'TIC': 0.0695}
#### Test Set: 
##### MSE: 0.000001, R²: 0.9905
#### Cross-Validation R²: 0.9460 (±0.1309)
### Health Status (Sample): 
e.g., Sample 1: kMc=0 (Healthy), kMt=0.987 (Healthy)
## Credits
The dataset used in this project is sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu). The original owners, Markelle Kelly, Rachel Longjohn, and Kolby Nottingham, should be referenced in any publication or derivative work using this dataset, as per their request.
