import os

import keras_tuner as kt
import tensorflow as tf

from src import config
from src.data_prep import augment_data, split_data
from src.dataset import get_datasets
from src.model import transfer_model


# original notebook code
def run_tuner(train_ds, val_ds):
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=config.patience, restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        config.ckpt_path, monitor="val_accuracy", verbose=0, save_best_only=True
    )

    tuner_b = kt.GridSearch(
        transfer_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        directory=config.tuner_dir,
        project_name=config.tuner_project,
        overwrite=True,
    )

    tuner_b.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[earlystop_callback, checkpoint_callback],
    )

    # reload with overwrite=False to retain results
    tuner_b = kt.GridSearch(
        transfer_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        directory=config.tuner_dir,
        project_name=config.tuner_project,
        overwrite=False,
    )

    tuner_b.results_summary()

    best_model = tuner_b.get_best_models(num_models=1)[0]
    best_model.save(config.model_path)
    print(f"Best model saved to {config.model_path}")

    return tuner_b


if __name__ == "__main__":
    split_data()
    augment_data()
    train_ds, val_ds, test_data = get_datasets()
    run_tuner(train_ds, val_ds)
