# Network Intrusion Detection using Machine Learning

## Overview

This repository contains a Python project implemented in a Jupyter Notebook (`.ipynb`) that aims to detect network intrusions using machine learning techniques. The code walks through the process of loading network traffic data, preprocessing it, training various classification models, evaluating their performance, and visualizing the results. The goal is to classify network connections as either 'normal' or 'anomaly'.

## Dataset

The dataset used in this project is the **Network Intrusion Detection** dataset available on Kaggle:

-   **Source:** [Kaggle - Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)
-   **Files Used:** Primarily `Train_data.csv` for training and evaluation (split into training and testing sets within the notebook). `Test_data.csv` is also available in the dataset.

The dataset contains various features extracted from network connections, including duration, protocol type, service, flags, byte counts, and connection rates, along with a label indicating whether the connection is normal or represents an attack (anomaly).

## Workflow

The Jupyter Notebook (`your_notebook_name.ipynb`) follows these key steps:

1.  **Data Loading:** Loads the `Train_data.csv` file using Pandas.
2.  **Exploratory Data Analysis (EDA):** Basic exploration including checking data shapes, types, missing values, and the distribution of the target variable ('class').
3.  **Preprocessing:**
    *   **Target Encoding:** Encodes the binary target variable ('normal', 'anomaly') into numerical format (0 and 1) using `LabelEncoder`.
    *   **Feature Type Identification:** Separates features into numerical and categorical types.
    *   **Numerical Feature Scaling:** Applies `StandardScaler` to scale numerical features.
    *   **Categorical Feature Encoding:** Applies `OneHotEncoder` to convert categorical features (`protocol_type`, `service`, `flag`) into a numerical format.
    *   **Pipeline Creation:** Uses `ColumnTransformer` and `Pipeline` to streamline preprocessing steps.
4.  **Train/Test Split:** Splits the data into training (70%) and testing (30%) sets using `train_test_split`, ensuring stratification to maintain class balance.
5.  **(Optional) Feature Selection:** Includes an example using `SelectKBest` (commented out by default in the provided code template).
6.  **Model Training:** Trains several machine learning classifiers on the preprocessed training data:
    *   Decision Tree (`DecisionTreeClassifier`)
    *   Random Forest (`RandomForestClassifier`)
    *   Gaussian Naive Bayes (`GaussianNB`)
    *   *(Other models like SVM or MLP might be included as commented-out options)*
7.  **Model Evaluation:** Evaluates the trained models on the preprocessed test set using standard metrics:
    *   Accuracy
    *   Precision (Weighted)
    *   Recall (Weighted)
    *   F1-Score (Weighted)
    *   Classification Report (per-class metrics)
    *   Confusion Matrix
8.  **Visualization:** Generates visualizations to interpret results:
    *   Confusion matrices for each model.
    *   Feature importance plots (for tree-based models like Random Forest).
    *   Bar charts comparing model performance metrics and execution times.

## Requirements

To run the notebook, you need Python 3 and the following libraries:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `jupyter` (to run the notebook environment)

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter