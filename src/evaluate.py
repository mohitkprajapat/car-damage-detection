import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

from src import config


def eval_top_models(tuner, test_data):
    models_cdp = tuner.get_best_models(num_models=6)
    best_hps = tuner.get_best_hyperparameters(num_trials=6)

    y_true_test = test_data.classes
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()
    print(len(models_cdp))
    rows = []

    for i in range(len(models_cdp)):
        test_data.reset()
        print(i,1)
        model_cdp = models_cdp[i]
        print(i,2)
        results = model_cdp.evaluate(test_data, verbose=1)
        print(i,3)
        metric_names = model_cdp.metrics_names
        print(i,4)
        y_pred_test = model_cdp.predict(test_data, verbose=1)
        y_hat_test = y_pred_test.argmax(axis=1)
        cm = confusion_matrix(y_true_test, y_hat_test)
        print(i,5)
        ax = axes[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="0.2f",
            cmap="Blues",
            xticklabels=config.class_labels,
            yticklabels=config.class_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        title = "Confusion Matrix - " + best_hps[i].values["base model"]
        ax.set_title(title)
        print(i,6)
        r = {"model": best_hps[i].values["base model"]}
        for name, val in zip(metric_names, results):
            r[name] = round(val, 4)
        rows.append(r)
        print(i)

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    return df


def show_misclassified(model, test_data):
    y_pred_test = model.predict(test_data, verbose=0)
    y_hat_test = y_pred_test.argmax(axis=1)
    y_true_test = test_data.classes

    misclassified_idx = np.where(y_true_test != y_hat_test)[0]
    show_img = 15
    row = math.ceil(show_img / 3)
    plt.figure(figsize=(20, row * 6))
    j = 1
    for i in misclassified_idx[:show_img]:
        img_path = test_data.filepaths[i]
        img = Image.open(img_path)
        plt.subplot(row, 3, j)
        plt.imshow(img)
        true_val = config.class_labels[y_true_test[i]]
        pred_val = config.class_labels[y_hat_test[i]]
        plt.title(f"True: {true_val}, Pred: {pred_val}")
        plt.axis("off")
        j += 1

    plt.show()


if __name__ == "__main__":
    import keras_tuner as kt
    from src.model import transfer_model
    from src.dataset import get_datasets

    train_ds, val_ds, test_data = get_datasets()

    tuner_b = kt.GridSearch(
        transfer_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        directory=config.tuner_dir,
        project_name=config.tuner_project,
        overwrite=False,
    )

    results_df = eval_top_models(tuner_b, test_data)