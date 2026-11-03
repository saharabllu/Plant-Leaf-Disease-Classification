# Plant Leaf Disease Classification

Deep learning project developed for the **Informatics Festival (I-Fest) 6.0** Machine Learning Competition. This repository contains two distinct experiments using custom Convolutional Neural Networks (CNNs) built from scratch to classify plant leaves.

## Key Results

Both models were trained on an 80/10/10 train-validation-test split. To ensure a fair evaluation of feature extraction capabilities, no pre-trained models (transfer learning) were used; all architectures were built from the ground up.

| Experiment | Classes | Test Accuracy | Macro F1-Score |
|---|---|---|---|
| **4-Class Plant Classification** | 4 (Plant Types) | **94.63%** | 0.95 |
| **27-Class Disease Classification** | 27 (Plant + Disease) | **83.66%** | 0.79 |

### Analysis
*   **4-Class Model:** Achieved near-perfect classification with a simple 4-layer CNN. The features of Potato, Lemon, Tea, and Tomato leaves are distinct enough for the model to learn effectively.
*   **27-Class Model:** Required a deeper 5-layer CNN and heavier augmentation. While performance remained high (Accuracy 83.66%), error analysis via the Confusion Matrix revealed expected challenges within the Solanaceae family (Tomato and Potato). Visually similar diseases (e.g., Early Blight vs. Target Spot) often overlap in visual symptoms, causing some misclassifications.

## Dataset

The dataset consists of over 4,000 leaf images across 4 plant types and 27 disease/condition classes. It exhibits significant class imbalance, presenting a realistic challenge for model training. 

Due to size limits, the raw dataset is not hosted here. Please refer to **[dataset/README.md](dataset/README.md)** for download instructions and detailed EDA.

## Methodology

1. **Data Preprocessing:** Dataset is split into 80% Train, 10% Validation, and 10% Test per plant category to maintain class proportions.
2. **Augmentation:** Applied only to the training set to prevent data leakage. Includes rotation, width/height shift, shear, zoom, and horizontal flip. The 27-class model additionally uses brightness and channel shift.
3. **Architecture:** Custom CNNs with Batch Normalization and Dropout (0.5) to prevent overfitting.
4. **Training Strategy:** Adam optimizer (lr=0.001). Utilized `EarlyStopping` and `ReduceLROnPlateau` to stop training when validation loss stagnates and reduce learning rate dynamically. `ModelCheckpoint` saves the best `.h5` weights.
5. **Evaluation:** Confusion Matrix, Classification Report (Precision, Recall, F1-Score), and Training History plots.

## Repository Structure

    plant-leaf-disease-classification/
    ├── dataset/                  # Dataset instructions (raw data excluded)
    ├── notebooks/                # Full step-by-step experiments
    │   ├── 01_plant_classification_4_classes.ipynb
    │   └── 02_disease_classification_27_classes.ipynb
    ├── src/                      # Refactored Python modules for reproducibility
    │   ├── four_classes/         # Code for the 4-class experiment
    │   └── twenty_seven_classes/ # Code for the 27-class experiment
    ├── outputs/                  # Confusion matrices and training plots
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## How to Run

1. Clone this repository and install dependencies:
   
       pip install -r requirements.txt

2. Download the raw dataset from the Google Drive link provided in `dataset/README.md` and place the `.zip` file inside the `dataset/` folder.
3. Open the notebooks using Jupyter or Google Colab. The scripts will automatically extract, clean, and split the data.

## Authors

This project was developed as a collaborative effort for the I-Fest 6.0 competition.

*   **Elvionita Sahara** — [@saharabllu](https://github.com/saharabllu)
*   **M. Ma'ruf Qomaruddin Kafi** — [@mmarufqk](https://github.com/mmarufqk)
