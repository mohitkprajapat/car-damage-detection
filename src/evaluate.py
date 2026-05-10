import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

from src import config


# original notebook code + new comparison table
def eval_top_models(tuner, test_data):
    models_cdp = tuner.get_best_models(num_models=6)
    best_hps = tuner.get_best_hyperparameters(num_trials=6)

    y_true_test = test_data.classes
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()

    # new — comparison table data
    rows = []

    for i in range(len(models_cdp)):
        model_cdp = models_cdp[i]
        results = model_cdp.evaluate(test_data, verbose=0)
        metric_names = model_cdp.metrics_names

        y_pred_test = model_cdp.predict(test_data, verbose=0)
        y_hat_test = y_pred_test.argmax(axis=1)
        cm = confusion_matrix(y_true_test, y_hat_test)

        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="0.2f", cmap="Blues",
                    xticklabels=config.class_labels, yticklabels=config.class_labels, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        title = "Confusion Matrix - " + best_hps[i].values['base model']
        ax.set_title(title)

        # new — collect metrics for comparison table
        r = {'model': best_hps[i].values['base model']}
        for name, val in zip(metric_names, results):
            r[name] = round(val, 4)
        rows.append(r)

    plt.tight_layout()
    plt.show()

    # new — print comparison table
    df = pd.DataFrame(rows)
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    return df


# original notebook code
def show_misclassified(model, test_data):
    y_pred_test = model.predict(test_data, verbose=0)
    y_hat_test = y_pred_test.argmax(axis=1)
    y_true_test = test_data.classes

    misclassified_idx = np.where(y_true_test != y_hat_test)[0]
    show_img = 15
    row = math.ceil(show_img / 3)
    plt.figure(figsize=(20, row*6))
    j = 1
    for i in misclassified_idx[:show_img]:
        img_path = test_data.filepaths[i]
        img = Image.open(img_path)
        plt.subplot(row, 3, j)
        plt.imshow(img)
        plt.title(f"True: {config.class_labels[y_true_test[i]]}, Pred: {config.class_labels[y_hat_test[i]]}")
        plt.axis("off")
        j += 1

    plt.show()
