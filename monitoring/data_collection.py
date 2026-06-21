import csv
import os
import threading
from datetime import datetime, timezone

from src import config

PRODUCTION_LOG_PATH = os.path.join(config.root_dir, "monitoring", "production_data.csv")

FIELDNAMES = [
    "timestamp",
    "image_path",
    "pred_class",
    "prob_minor",
    "prob_moderate",
    "prob_severe",
    "confidence",
    "damage_score",
    "true_label",
]

_lock = threading.Lock()


def _ensure_file(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def log_prediction(result: dict, image_filename: str, log_path: str = PRODUCTION_LOG_PATH) -> None:
    """Append one prediction's output to the production log.

    `result` is the dict returned by Predictor.predict().
    `image_filename` is the filename saved under static/uploads/ (e.g. fname
    from app.py), used later to find the file again for review.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": image_filename,
        "pred_class": result["pred_class"],
        "prob_minor": result["probs"].get("minor", 0.0),
        "prob_moderate": result["probs"].get("moderate", 0.0),
        "prob_severe": result["probs"].get("severe", 0.0),
        "confidence": result["confidence"],
        "damage_score": result["score"],
        "true_label": "",
    }
    with _lock:
        _ensure_file(log_path)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def get_unlabeled_rows(log_path: str = PRODUCTION_LOG_PATH) -> list[dict]:
    """Return all logged predictions that don't have a true_label yet."""
    if not os.path.exists(log_path):
        return []
    with _lock:
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
    return [r for r in rows if not r.get("true_label")]


def set_true_label(image_path: str, true_label: str, log_path: str = PRODUCTION_LOG_PATH) -> bool:
    """Fill in true_label for the first unlabeled row matching image_path."""
    if true_label not in config.class_labels:
        return False
    if not os.path.exists(log_path):
        return False

    with _lock:
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))

        updated = False
        for row in rows:
            if row.get("image_path") == image_path and not row.get("true_label"):
                row["true_label"] = true_label
                updated = True
                break

        if updated:
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

    return updated