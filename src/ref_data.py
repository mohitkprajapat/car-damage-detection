import csv
import os
from datetime import datetime, timezone

from src import config
from src.predictor import Predictor

REFERENCE_PATH = os.path.join(config.root_dir, "monitoring", "reference_data.csv")

FIELDNAMES = [
    "timestamp",
    "image_path",
    "true_label",
    "pred_class",
    "prob_minor",
    "prob_moderate",
    "prob_severe",
    "confidence",
    "damage_score",
]


def _iter_test_images():
    """Yield (image_path, true_label) for every image under data/test/."""
    for class_name in config.class_labels:
        class_dir = os.path.join(config.test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(class_dir, fname), class_name


def build_reference_data(output_path: str = REFERENCE_PATH) -> str:
    predictor = Predictor()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows_written = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for img_path, true_label in _iter_test_images():
            result = predictor.predict(img_path)
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": img_path,
                "true_label": true_label,
                "pred_class": result["pred_class"],
                "prob_minor": result["probs"].get("minor", 0.0),
                "prob_moderate": result["probs"].get("moderate", 0.0),
                "prob_severe": result["probs"].get("severe", 0.0),
                "confidence": result["confidence"],
                "damage_score": result["score"],
            })
            rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    build_reference_data()