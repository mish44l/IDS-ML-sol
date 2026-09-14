# IDS-ML-sol
Machine Learning Based Network Intrusion Detection System for SecureNet Corp.
# Machine Learning Based Network Intrusion Detection System

## Project Overview

This project develops a proof-of-concept Machine Learning-based Network Intrusion Detection System (NIDS) for the fictional organization **SecureNet Corp.**

The system uses network traffic data from the **UNSW-NB15 dataset** and applies a Random Forest classifier to distinguish between normal and malicious network traffic.

## Objective

The main objective is to demonstrate how Machine Learning can be used to automatically identify potentially malicious network activity and support cybersecurity monitoring.

## Dataset

The project uses the `UNSW_NB15_training-set.csv` file from the UNSW-NB15 dataset.

The dataset contains network traffic features representing normal and malicious activity.

For this project, the available training dataset was divided into training and testing subsets. The separate official UNSW-NB15 testing file was not used.

## Methodology

The project follows these main steps:

1. Load the network traffic dataset using Python and pandas.
2. Inspect and preprocess the data.
3. Perform Exploratory Data Analysis (EDA).
4. Separate input features from the target label.
5. Split the data into training and testing subsets.
6. Train a Random Forest classification model.
7. Generate predictions on unseen testing records.
8. Evaluate the model using accuracy, precision, recall, and F1 score.
9. Analyze the confusion matrix and security-related errors.
10. Examine feature importance.

## Model

**Algorithm:** Random Forest Classifier

The Random Forest model was selected because it is suitable for classification tasks involving multiple network traffic features and can learn complex patterns in the data.

## Results

The trained model achieved the following results:

| Metric              | Result |
| ------------------- | -----: |
| Accuracy            | 95.98% |
| Precision           | 96.37% |
| Recall              | 97.78% |
| F1 Score            | 97.07% |
| False Positive Rate |  7.86% |
| False Negative Rate |  2.22% |

The confusion matrix showed:

* True Negatives: 10,320
* False Positives: 880
* False Negatives: 531
* True Positives: 23,338

The high recall indicates that the model detected most malicious network traffic in the testing data.

## Repository Contents

* `info security assign 1.ipynb` — Jupyter Notebook containing the complete Machine Learning workflow.
* `results/` — Model evaluation results and generated visualizations.
* `README.md` — Project documentation.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Disclaimer

This project is a proof-of-concept developed for academic purposes. The model is not intended to replace professional cybersecurity infrastructure or human security analysis.
