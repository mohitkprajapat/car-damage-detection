import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config

def get_datasets():
    image_shape = config.img_shape

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.train_aug_dir,
        label_mode="categorical",
        image_size=image_shape,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.val_dir,
        label_mode="categorical",
        image_size=image_shape,
        batch_size=config.batch_size,
        shuffle=True,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_data = ImageDataGenerator().flow_from_directory(
        directory=config.test_dir,
        target_size=image_shape,
        classes=config.class_labels,
        batch_size=config.test_batch_size,
        shuffle=False,
    )

    return train_ds, val_ds, test_data