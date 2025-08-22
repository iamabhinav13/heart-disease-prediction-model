# Heart Disease Prediction using Random Forest 🩺

This project contains a complete machine learning workflow for predicting the likelihood of heart disease based on patient data. It includes data preprocessing, model training, evaluation, and scripts for deployment as an online endpoint using Azure Machine Learning.

---

## 🚀 Project Overview

The goal of this project is to build a reliable classifier to identify patients at high risk of heart disease. The model is a `RandomForestClassifier` trained on a combination of demographic, clinical, and lifestyle data.

### Key Features
* **Data Preprocessing**: Merges and cleans data from multiple sources.
* **Model Training**: Uses a Random Forest model with hyperparameter tuning via `RandomizedSearchCV`.
* **Evaluation**: Assesses model performance using Accuracy, ROC AUC, Classification Report, and a Confusion Matrix.
* **Deployment Ready**: Includes scripts to register the model and deploy it as a managed online endpoint on Azure ML.

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.8+
* Conda
* An Azure account and Azure ML Workspace (for deployment)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/heart-disease-prediction.git](https://github.com/your-username/heart-disease-prediction.git)
    cd heart-disease-prediction
    ```

2.  **Create the Conda environment:**
    The `conda.yml` file contains all necessary dependencies.
    ```bash
    conda env create -f deployment/conda.yml
    conda activate basic-env-cpu
    ```

3.  **Place Data:**
    Download the dataset and place the `.csv` files (`demographic_table.csv`, `patient_condition_table.csv`, etc.) into the `data/` directory.

---

## 📖 Usage

### Model Training

The entire training process is documented in the Jupyter Notebook. To run it:

1.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Open and run the cells in `notebooks/Heart_Disease_Prediction_Model_(Random_Forest).ipynb`.

The notebook will perform the following steps:
1.  Load and merge all data sources.
2.  Preprocess the data (handling missing values).
3.  Split the data into training and testing sets.
4.  Train a Random Forest model using `RandomizedSearchCV` for hyperparameter optimization.
5.  Evaluate the best model's performance.
6.  Save the trained model as `heart_disease_model.pkl` in the root directory (this file is git-ignored).

### Model Deployment on Azure

The scripts in the `deployment/` folder are used to deploy the trained model.

1.  **Authenticate with Azure:**
    ```bash
    az login
    ```

2.  **Update Deployment Script:**
    Open `deployment/deploy_model.py` and ensure the `subscription_id`, `resource_group`, and `workspace_name` variables match your Azure ML workspace details.

3.  **Run the Deployment Script:**
    ```bash
    python deployment/deploy_model.py
    ```
    This script will:
    * Register the `heart_disease_model.pkl` in your Azure ML workspace.
    * Create a managed online endpoint.
    * Deploy the model to the endpoint with the specified compute resources and environment.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
