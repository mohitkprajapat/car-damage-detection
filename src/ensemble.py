from __future__ import annotations

import itertools
from typing import Callable
import os
import numpy as np
import pandas as pd

import keras_tuner as kt

from src import config
from src.dataset import get_datasets
from src.model import transfer_model

N_MODELS = 6
DEFAULT_PRED_CACHE = os.path.join(config.models, "test_predictions.npz")

def majority_vote(
    probs_list: list[np.ndarray],
) -> tuple[str, float]:
    mean_probs: np.ndarray = np.mean(np.stack(probs_list, axis=0), axis=0)
    idx: int = int(mean_probs.argmax())
    return config.class_labels[idx], float(mean_probs[idx])


def collect_predictions(
    n_models: int = N_MODELS,
    cache_path: str = DEFAULT_PRED_CACHE,
    force_rerun: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    
    if not force_rerun and os.path.exists(cache_path):
        print(f"  Loading cached predictions from {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        y_true = data["y_true"]
        model_names: list[str] = data["model_names"].tolist()
        all_probs: list[np.ndarray] = [
            data[f"probs_{i}"] for i in range(len(model_names))
        ]
        print(f"  Loaded {len(model_names)} models, {len(y_true)} samples.")
        return y_true, all_probs, model_names


    _, _, test_data = get_datasets()

    tuner = kt.GridSearch(
        transfer_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        directory=config.tuner_dir,
        project_name=config.tuner_project,
        overwrite=False,
    )

    models = tuner.get_best_models(num_models=n_models)
    best_hps = tuner.get_best_hyperparameters(num_trials=n_models)
    model_names: list[str] = [
        hp.values.get("base model", f"model_{i}") for i, hp in enumerate(best_hps)
    ]

    y_true: np.ndarray = test_data.classes

    all_probs: list[np.ndarray] = []
    for i, model in enumerate(models):
        print(f"  Predicting with model {i + 1}/{n_models}: {model_names[i]} …")
        test_data.reset()
        probs = model.predict(test_data, verbose=0)  # shape (n_samples, n_classes)
        all_probs.append(probs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        y_true=y_true,
        model_names=np.array(model_names),
        **{f"probs_{i}": probs for i, probs in enumerate(all_probs)},
    )
    print(f"  Saved predictions to {cache_path}")

    return y_true, all_probs, model_names

 
def rerank_combinations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rank_accuracy"]    = df["accuracy"].rank(ascending=False, method="min")
    df["rank_confidence"]  = df["mean_confidence"].rank(ascending=False, method="min")
    df["rank_size"]        = df["combo_size"].rank(ascending=True,  method="dense")

    offset = 60
    df["harmonic_mean_rank"] = 3 / (
        1 / (df["rank_accuracy"]+ offset) +
        1 / (df["rank_confidence"]+ offset) +
        1 / (df["rank_size"]+ offset)
    ) - offset

    df["final_rank"] = df["harmonic_mean_rank"].rank(ascending=True, method="min").astype(int)

    df = df.sort_values("final_rank")

    return df

def evaluate_ensembles(
    y_true: np.ndarray,
    all_probs: list[np.ndarray],
    model_names: list[str],
    vote_fn: Callable[[list[np.ndarray]], tuple[str, float]] = majority_vote,
    top_k: int = 10,
) -> pd.DataFrame:
    n = len(all_probs)
    n_samples = len(y_true)
    label_to_idx = {label: i for i, label in enumerate(config.class_labels)}

    rows: list[dict] = []

    total_combos = sum(
        len(list(itertools.combinations(range(n), r))) for r in range(1, n + 1)
    )
    print(f"\nEvaluating {total_combos} ensemble combinations …\n")

    combo_num = 0
    for r in range(1, n + 1):
        for indices in itertools.combinations(range(n), r):
            combo_num += 1

            # Run vote_fn sample-by-sample
            y_pred = np.empty(n_samples, dtype=int)
            confidences = np.empty(n_samples, dtype=float)

            for s in range(n_samples):
                probs_list = [all_probs[i][s] for i in indices]
                pred_label, conf = vote_fn(probs_list)
                y_pred[s] = label_to_idx[pred_label]
                confidences[s] = conf

            accuracy = float((y_pred == y_true).mean())
            mean_conf = float(confidences.mean())

            rows.append(
                {
                    "combo_id": combo_num,
                    "combo_size": r,
                    "model_indices": list(indices),
                    "models": " + ".join(model_names[i] for i in indices),
                    "accuracy": round(accuracy, 4),
                    "mean_confidence": round(mean_conf, 4),
                }
            )

    df = pd.DataFrame(rows)
    df = rerank_combinations(df)
    df = df.reset_index(drop=True)

    print(f"{'Rank':<5} {'Accuracy':>9} {'Mean Conf':>10} {'Size':>5}  Models")
    print("─" * 80)
    for rank, row in df.head(top_k).iterrows():
        print(
            f"{rank + 1:<5} {row['accuracy']:>9.4f} {row['mean_confidence']:>10.4f}"
            f" {row['combo_size']:>5}  {row['models']}"
        )

    return df

def report_individual_baselines(
    y_true: np.ndarray,
    all_probs: list[np.ndarray],
    model_names: list[str],
) -> None:
    print("\nIndividual model baselines:")
    print(f"{'Model':<25} {'Accuracy':>9}")
    print("─" * 36)
    label_to_idx = {label: i for i, label in enumerate(config.class_labels)}
    for name, probs in zip(model_names, all_probs):
        y_pred = probs.argmax(axis=1)
        acc = float((y_pred == y_true).mean())
        print(f"{name:<25} {acc:>9.4f}")

COMBO_MANIFEST_PATH = os.path.join(config.models, "best_combo.json")
 
 
def save_best_combo(
    results_df: pd.DataFrame,
    model_names: list[str],
    manifest_path: str = COMBO_MANIFEST_PATH,
) -> dict:
    best_row = results_df.iloc[0]
 
    manifest = {
        "model_indices": best_row["model_indices"],
        "model_names":   [model_names[i] for i in best_row["model_indices"]],
        "accuracy":      float(best_row["accuracy"]),
        "combo_size":    int(best_row["combo_size"]),
    }
 
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
 
    print(f"\nBest combo saved to {manifest_path}")
    print(f"  Models : {' + '.join(manifest['model_names'])}")
    print(f"  Accuracy: {manifest['accuracy']:.4f}  |  Size: {manifest['combo_size']}")
 
    return manifest



def main(force_rerun) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1 — Loading tuner and collecting predictions …")
    print("=" * 60)
    y_true, all_probs, model_names = collect_predictions(force_rerun=force_rerun)

    report_individual_baselines(y_true, all_probs, model_names)

    print("\n" + "=" * 60)
    print("Step 2 — Evaluating all ensemble combinations …")
    print("=" * 60)
    results_df = evaluate_ensembles(
        y_true,
        all_probs,
        model_names,
        vote_fn=majority_vote,
        top_k=10,
    )
    
    print("\n" + "=" * 60)
    print("Step 3 — Saving best combo …")
    print("=" * 60)
    save_best_combo(results_df, model_names)

    return results_df


if __name__ == "__main__":
    df = main(True)