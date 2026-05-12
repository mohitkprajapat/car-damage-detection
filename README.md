# Car Damage Severity Detector

A deep learning web app that classifies car damage images into three severity levels — **minor**, **moderate**, and **severe** — to help automate the initial triage stage of insurance claim reviews.

---

## What it does

You upload a photo of a damaged car, and the model tells you how bad it is. That's the short version.

Under the hood, it runs an ensemble of fine-tuned transfer learning models (VGG16, ResNet50, MobileNetV2) and combines their predictions through a majority vote to produce a final classification with a confidence score. The ensemble was selected by exhaustively evaluating all possible model combinations and picking the one with the best trade-off between accuracy, confidence, and size.

The three damage classes:

| Class | What it means |
|---|---|
| **Minor** | Surface scratches, small dents, scuffs — no structural impact |
| **Moderate** | Panel damage, broken lights, bumper deformation — may need parts replaced |
| **Severe** | Frame/structural damage, major crumpling — likely a significant repair job |

---

## How the training pipeline works

**1. Data preparation**

`src/data_prep.py` splits the raw training images (in `data/training/`) into train and validation sets (90/10 by default), then augments the training split with random rotations, shifts, zooms, and horizontal flips — generating 3 augmented copies per original image.

**2. Hyperparameter search**

`src/train.py` runs a `GridSearch` over six model variants:

- `resnet50` — frozen backbone
- `resnet50-165` — fine-tuned from layer 165
- `mobnetv2` — frozen backbone
- `mobnetv2-143` — fine-tuned from layer 143
- `vgg16` — frozen backbone
- `vgg16-15` — fine-tuned from layer 15

Each variant shares the same classification head: `GlobalAveragePooling → Dropout → Dense(512) → BN → Dropout → Dense(128) → BN → Dropout → Softmax(3)`.

**3. Ensemble selection**

`src/ensemble.py` loads the top-N tuner trials, collects predictions on the test set, and evaluates every possible combination of models. Combinations are ranked using a harmonic mean of their accuracy rank, mean confidence rank, and size rank — favouring smaller ensembles that don't sacrifice much accuracy. The best combo is saved to `models/best_models/` along with a `best_combo.json` manifest.

**4. Serving**

`src/predictor.py` loads the saved ensemble at startup and exposes a `predict(img_path)` method. The Flask app calls this on every upload and renders the result with per-class probability bars.

---

## Getting started

**Prerequisites:** Python 3.11, `uv` (or `pip`)

```bash
# Clone and install
git clone <repo-url>
cd car-damage-severity
pip install -e .

# Set the root dir if needed (defaults to cwd)
echo "ROOT_DIR=$(pwd)" > .env
```

**Running the app** (requires trained models in `models/best_models/`):

```bash
python app.py
```

Then open `http://localhost:5000`.

**Training from scratch:**

```bash
# 1. Put your images in data/training/{minor,moderate,severe}/
# 2. Run the full training pipeline
python -m src.train

# 3. Run ensemble selection
python -m src.ensemble
```

---

## Configuration

Everything lives in `src/config.py`. Key settings:

```python
img_shape     = (224, 224)   # Input image size
batch_size    = 32
epochs        = 20
patience      = 5            # Early stopping
split_ratio   = 0.1          # Validation split
aug_count     = 3            # Augmented copies per image
class_labels  = ["minor", "moderate", "severe"]
```

---

## Tech stack

- **TensorFlow / Keras** — model training and inference
- **Keras Tuner** — `GridSearch` over backbone variants
- **Flask** — lightweight web server
- **scikit-learn** — confusion matrices and evaluation metrics
- **NumPy / Pandas** — ensemble scoring and results handling

---

## Notes

- Uploaded images are automatically deleted after 7 days (`src/utils.py`).
- If no ensemble is found at startup, the app falls back to a single `best_model.keras` if one exists.
- The tuner writes trial artifacts to `tuner_results/` — these are needed to re-run ensemble selection without retraining.