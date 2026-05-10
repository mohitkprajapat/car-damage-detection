import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


# original notebook code
def get_datasets():
    image_shape = config.img_shape

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.train_aug_dir, 
        label_mode="categorical", 
        image_size=image_shape, 
        batch_size=config.batch_size, 
        shuffle=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.val_dir, 
        label_mode="categorical", 
        image_size=image_shape, 
        batch_size=config.batch_size, 
        shuffle=True)

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_data = ImageDataGenerator().flow_from_directory(
        directory=config.test_dir,
        target_size=image_shape,
        classes=config.class_labels,
        batch_size=256,
        shuffle=False
    )

    return train_ds, val_ds, test_data


# original notebook code
def plotImages(image_arr, labels, img_count):
    row = math.ceil(math.sqrt(img_count))
    plt.figure(figsize=(row*2, row*2))
    np_rand = np.random.randint(0, 256, img_count)
    for i in range(img_count):
        k = np_rand[i]
        plt.subplot(row, row, i + 1)
        plt.imshow(image_arr[k].astype('uint8'))
        plt.title(config.class_labels[labels[k].argmax()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
