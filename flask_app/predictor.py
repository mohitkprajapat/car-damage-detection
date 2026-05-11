import os
import uuid

import tensorflow as tf

from src import config, gradcam


# new
class Predictor:
    def __init__(self):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError("Model not found. Train the model first.")
        self.model = tf.keras.models.load_model(config.model_path)
        self.labels = config.class_labels

    def predict(self, img_path):
        import numpy as np
        from tensorflow.keras.preprocessing.image import img_to_array, load_img

        img = load_img(img_path, target_size=config.img_shape)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        preds = self.model.predict(arr, verbose=0)[0]
        idx = preds.argmax()
        return {
            "pred_class": self.labels[idx],
            "confidence": float(preds[idx]),
            "probs": {self.labels[i]: float(preds[i]) for i in range(len(self.labels))},
        }

    def predict_with_gradcam(self, img_path, save_dir):
        result = self.predict(img_path)
        # Just reuse the uploaded image as the display image (no Grad-CAM)
        fname = os.path.basename(img_path)
        result["gradcam_path"] = fname
        return result
