import os
import uuid
import tensorflow as tf

from src import config
from src import gradcam


# new
class Predictor:
    def __init__(self):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model not found at {config.model_path}. Train the model first.")
        self.model = tf.keras.models.load_model(config.model_path)
        self.labels = config.class_labels

    def predict(self, img_path):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        import numpy as np
        img = load_img(img_path, target_size=config.img_shape)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        preds = self.model.predict(arr, verbose=0)[0]
        idx = preds.argmax()
        return {
            "pred_class": self.labels[idx],
            "confidence": float(preds[idx]),
            "probs": {self.labels[i]: float(preds[i]) for i in range(len(self.labels))}
        }

    def predict_with_gradcam(self, img_path, save_dir):
        result = gradcam.analyze(self.model, img_path, self.labels)
        fname = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(save_dir, fname)
        result["overlay"].save(save_path)
        result["gradcam_path"] = fname
        return result
