# Car Damage Severity Detector

Classify car damage images into **minor / moderate / severe** using transfer learning — built for automating insurance claim triage.

---

## Overview

This project trains a CNN classifier on car damage images to predict damage severity. It's designed to support insurance claim workflows by giving adjusters an automated first-pass severity score with visual explanations.

The original work lives in `notebooks/Car_Damage_Severity_Detection.ipynb`. This repo refactors it into a structured Python project and adds Grad-CAM visualizations and a Flask web app on top.

---

## Model Architecture

- **Backbones:** VGG16, ResNet50, MobileNetV2 (ImageNet weights, frozen or partially fine-tuned)
- **Head:** GlobalAveragePooling → Dense(512, ReLU) → BN → Dropout → Dense(128, ReLU) → BN → Dropout → Dense(3, Softmax)
- **Regularization:** L2 (0.01), Dropout (0.2)
- **Optimizer:** Adam (lr=0.01), CategoricalCrossentropy loss
- **Tuning:** KerasTuner GridSearch across 6 configurations (frozen vs. fine-tuned variants of each backbone)
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint (best val_accuracy)

---


## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your raw images under `data/training/{minor,moderate,severe}/` and test images under `data/test/{minor,moderate,severe}/`.

### 3. Train

```bash
python -m src.train
```

This runs: split → augment → GridSearch tuning → saves best model to `models/best_model.keras`.

### 4. Run data prep only

```bash
python -m src.data_prep
```

### 5. Launch Flask app

```bash
cd flask_app
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Results

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| *(run training to populate)* | — | — | — | — |

---

## Tech Stack

- TensorFlow / Keras
- KerasTuner (GridSearch)
- Flask
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn