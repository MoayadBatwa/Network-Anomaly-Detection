# Network Anomaly Detection using Machine Learning

This project detects anomalies in network traffic using a variety of machine learning and deep learning models. It is based on the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and compares the performance of five different classification models.

This code is an implementation of the project described in the "Anomaly Detection in Sequential Data" document.

## 🚀 Project Overview

The goal is to build and evaluate models that can distinguish between "Normal" network activity and "Attack" traffic. The project walks through the entire data science pipeline:
1.  **Data Loading & Preprocessing:** Cleaning the data, encoding categorical features, and normalizing all features.
2.  **Model Training:** Building five different models on the training data.
3.  **Model Evaluation:** Comparing the models using accuracy, classification reports, confusion matrices, and ROC/AUC curves.

## 📊 Dataset

This project uses the **UNSW-NB15 dataset**. This is a comprehensive dataset for network intrusion detection, which includes a mix of normal network activity and nine different types of modern attacks (e.g., Fuzzers, DoS, Worms, etc.).

Due to its size, a 10% sample of the dataset was used for training and testing.

## 🤖 Models Implemented

Five models were trained and compared:

**Machine Learning Models:**
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

**Deep Learning Models:**
* Simple Neural Network (NN)
* Long Short-Term Memory (LSTM)

## 📈 Results

Based on the evaluation, the **Random Forest** model was the top performer across all metrics:
* **Highest Accuracy:** ~96%
* **Best AUC Score:** 0.99 (near-perfect)
* **Strong F1-Score:** Showed a great balance between precision (few false alarms) and recall (few missed attacks).

The LSTM, while powerful for sequential data, had the lowest performance in this specific configuration, suggesting the classic ML models were more effective for this task.

## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Install dependencies:**
    The project requires the following Python libraries. You can install them using `pip`:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

3.  **Get the dataset:**
    * Download the `UNSW_NB15_dataset.csv` file (e.g., from [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)).
    * Place the `.csv` file in the project's root directory (or any directory you prefer).

4.  **Update the dataset path:**
    Open the Python script and change this line to point to your file's location:
    ```python
    dataset_path = '/content/UNSW_NB15_dataset.csv' # <-- Change this path!
    ```

5.  **Run the script:**
    ```bash
    python Network_Anomaly_Detection.py
    ```
