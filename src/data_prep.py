import os
import random
import shutil

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)
from tqdm import tqdm

from src import config


def split_data():
    source_dir = config.training_dir
    target_base = config.split_dir
    split_ratio = config.split_ratio

    class_labels = config.class_labels
    random.seed(config.seed)

    for class_name in class_labels:
        src_class_dir = os.path.join(source_dir, class_name)
        file_ext = (".jpg", ".jpeg", ".png")
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(file_ext)]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        val_images = images[:split_index]
        train_images = images[split_index:]

        val_class_dir = os.path.join(target_base, "val", class_name)
        train_class_dir = os.path.join(target_base, "train", class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(train_class_dir, exist_ok=True)

        for img in tqdm(val_images, desc=f"Copying val/{class_name}"):
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(val_class_dir, img))

        for img in tqdm(train_images, desc=f"Copying train/{class_name}"):
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))


def augment_data():
    source_dir = config.train_dir
    target_dir = config.train_aug_dir
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    augmentor = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    class_labels = config.class_labels
    image_shape = config.img_shape

    for class_name in class_labels:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        for image_file in tqdm(os.listdir(source_class_dir), desc=f"Processing {class_name}"):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(source_class_dir, image_file)

            shutil.copy(img_path, os.path.join(target_class_dir, image_file))

            image = load_img(img_path, target_size=image_shape)
            image_array = img_to_array(image)
            image_array = image_array.reshape((1,) + image_array.shape)

            aug_iter = augmentor.flow(image_array, batch_size=1)
            for i in range(config.aug_count):
                aug_image = next(aug_iter)[0].astype("uint8")
                aug_pil = array_to_img(aug_image)

                save_name = f"{os.path.splitext(image_file)[0]}_aug{i + 1}.jpg"
                save_path = os.path.join(target_class_dir, save_name)
                aug_pil.save(save_path)


if __name__ == "__main__":
    split_data()
    augment_data()
