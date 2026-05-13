import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import config
from src.ensemble import MODELS_PATH, majority_vote


def damage_score(minor: float, moderate: float, severe: float) -> float:
    CLASS_MIDPOINTS = {
        "minor":    2.5,
        "moderate": 5.5,
        "severe":   9.0,
    }
    score = (
        minor    * CLASS_MIDPOINTS["minor"] +
        moderate * CLASS_MIDPOINTS["moderate"] +
        severe   * CLASS_MIDPOINTS["severe"]
    )

    return round(score, 1)

class Predictor:
    def __init__(self):
        self.labels = config.class_labels
        self._vote_fn = majority_vote

        if os.path.exists(MODELS_PATH):
            self._load_combo(MODELS_PATH)
        elif os.path.exists(config.model_path):
            print("No combo manifest found — loading single best model.")
            self._models = [tf.keras.models.load_model(config.model_path)]
        else:
            raise FileNotFoundError(
                "No model found. Run ensemble_eval.main() or train a model first."
            )

    def _load_combo(self,models_path: str) -> list:
        model_dirs = model_dirs = sorted(
            [d for d in os.listdir(MODELS_PATH) if d.startswith("model_")]
            )
        
        self._models = []
        for d in model_dirs:
            self._models.append(tf.keras.models.load_model(os.path.join(models_path, d)))

    def predict(self, img_path: str) -> dict:
        img = load_img(img_path, target_size=config.img_shape)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        if len(self._models) == 1:
            preds = self._models[0].predict(arr, verbose=0)[0]
            idx = int(preds.argmax())
            pred_class = self.labels[idx]
            confidence = float(preds[idx])
            probs = {self.labels[i]: float(preds[i]) for i in range(len(self.labels))}
        else:
            probs_list = [
                model.predict(arr, verbose=0)[0] for model in self._models
            ]
            pred_class, confidence = self._vote_fn(probs_list)
            mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
            probs = {self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))}

        score = damage_score(
            probs.get("minor", 0),
            probs.get("moderate", 0),
            probs.get("severe", 0),
            )
        
        return {
            "pred_class": pred_class,
            "confidence": confidence,
            "probs": probs,
            "score": score
        }