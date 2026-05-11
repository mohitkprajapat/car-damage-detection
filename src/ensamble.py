import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


# load all 6 models from tuner
def load_models(tuner):
    return tuner.get_best_models(num_models=6), tuner.get_best_hyperparameters(num_trials=6)


# get val data as a flow (same pipeline as notebook)
def get_val_data():
    val_data = ImageDataGenerator().flow_from_directory(
        directory=config.val_dir,
        target_size=config.img_shape,
        classes=config.class_labels,
        batch_size=256,
        shuffle=False,
    )
    return val_data


# run all 6 models on val set, collect probs — shape (n_models, n_samples, n_classes)
def collect_probs(models, val_data):
    val_data.reset()
    y_true = val_data.classes

    all_probs = []
    for i, m in enumerate(models):
        val_data.reset()
        p = m.predict(val_data, verbose=0)
        all_probs.append(p)
        print(f"  model {i + 1}/6 done — solo acc: {(p.argmax(1) == y_true).mean():.4f}")

    return np.array(all_probs), y_true  # (6, N, 3), (N,)


# average probs for a subset of models, return predicted classes
def ensemble_pred(probs_subset):
    avg = probs_subset.mean(axis=0)  # (N, 3)
    return avg.argmax(axis=1)


# try every non-empty subset of the 6 models (~63 combos)
def search_combos(all_probs, y_true, model_names):
    n = len(all_probs)
    rows = []

    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            subset = all_probs[list(combo)]  # (r, N, 3)
            y_hat = ensemble_pred(subset)

            acc = (y_hat == y_true).mean()
            f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
            prec = precision_score(y_true, y_hat, average="macro", zero_division=0)
            rec = recall_score(y_true, y_hat, average="macro", zero_division=0)

            names = " + ".join(model_names[i] for i in combo)
            rows.append(
                {
                    "combo": names,
                    "n_models": r,
                    "accuracy": round(acc, 4),
                    "f1": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                }
            )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df


# print top-k results and plot best combo confusion matrix
def show_results(df, all_probs, y_true, model_names, top_k=10):
    print(f"\nTop {top_k} combos by val accuracy:")
    print(df.head(top_k).to_string(index=False))

    # best combo
    best = df.iloc[0]
    print(f"\nBest combo: {best['combo']}")
    print(
        f"  accuracy={best['accuracy']}  f1={best['f1']}  precision={best['precision']}  recall={best['recall']}"
    )

    # confusion matrix for best combo
    best_idxs = [list(model_names).index(n) for n in best["combo"].split(" + ")]
    subset = all_probs[best_idxs]
    y_hat = ensemble_pred(subset)

    cm = confusion_matrix(y_true, y_hat)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.class_labels,
        yticklabels=config.class_labels,
    )
    plt.title(f"Best Ensemble — {best['combo']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # accuracy by combo size
    plt.figure(figsize=(7, 4))
    for k, grp in df.groupby("n_models"):
        plt.scatter([k] * len(grp), grp["accuracy"], alpha=0.5, label=f"{k} model(s)")
    plt.xlabel("Number of models in combo")
    plt.ylabel("Val accuracy")
    plt.title("Ensemble accuracy by combo size")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return df


# save results to csv
def save_results(df, path="ensemble_results.csv"):
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")


def run(tuner=None, models=None, model_names=None):
    # accept either a tuner or pre-loaded models list
    if tuner is not None:
        print("Loading models from tuner...")
        models, hps = load_models(tuner)
        model_names = np.array([hp.values["base model"] for hp in hps])
    elif models is not None:
        model_names = (
            np.array(model_names)
            if model_names
            else np.array([f"model_{i}" for i in range(len(models))])
        )
    else:
        raise ValueError("Pass either tuner= or models= + model_names=")

    print("Running all models on val set...")
    val_data = get_val_data()
    all_probs, y_true = collect_probs(models, val_data)

    print(f"\nSearching {2 ** len(models) - 1} combos...")
    df = search_combos(all_probs, y_true, model_names)

    show_results(df, all_probs, y_true, model_names)
    save_results(df)
    return df


# --- usage ---
# from tuner:
#   from src.ensemble import run
#   run(tuner=tuner_b)
#
# from manually loaded models:
#   models = [tf.keras.models.load_model(p) for p in model_paths]
#   names  = ["resnet50", "mobnetv2", "vgg16", "resnet50-165", "mobnetv2-143", "vgg16-15"]
#   run(models=models, model_names=names)

if __name__ == "__main__":
    import sys

    # expect model paths as args, e.g.:
    # python -m src.ensemble models/m1.keras models/m2.keras ...
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m src.ensemble <model1.keras> <model2.keras> ...")
        print("       (pass up to 6 model paths)")
        sys.exit(1)

    models = []
    for p in paths:
        print(f"Loading {p}...")
        models.append(tf.keras.models.load_model(p))

    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    run(models=models, model_names=names)
