import json
import os

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import config
from src.ensemble import COMBO_MANIFEST_PATH, majority_vote
from src.model import transfer_model


class Predictor:
    def __init__(self):
        self.labels = config.class_labels
        self._vote_fn = majority_vote

        if os.path.exists(COMBO_MANIFEST_PATH):
            self._load_combo(COMBO_MANIFEST_PATH)
        elif os.path.exists(config.model_path):
            print("No combo manifest found — loading single best model.")
            self._models = [tf.keras.models.load_model(config.model_path)]
        else:
            raise FileNotFoundError(
                "No model found. Run ensemble_eval.main() or train a model first."
            )

    def _load_combo(self, manifest_path: str) -> None:
        with open(manifest_path) as f:
            manifest = json.load(f)

        indices: list[int] = manifest["model_indices"]
        names: list[str] = manifest["model_names"]
        n_to_load = max(indices) + 1 

        tuner = kt.GridSearch(
            transfer_model,
            objective=kt.Objective("val_accuracy", direction="max"),
            directory=config.tuner_dir,
            project_name=config.tuner_project,
            overwrite=False,
        )

        all_models = tuner.get_best_models(num_models=n_to_load)
        self._models = [all_models[i] for i in indices]

        print(
            f"Loaded {'ensemble' if len(indices) > 1 else 'single'} combo "
            f"({' + '.join(names)})  |  test accuracy: {manifest['accuracy']:.4f}"
        )

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

        return {
            "pred_class": pred_class,
            "confidence": confidence,
            "probs": probs,
        }