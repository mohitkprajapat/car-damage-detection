# paths
data_dir = "data"
training_dir = "data/training"
split_dir = "data/split_val_train"
train_dir = "data/split_val_train/train"
val_dir = "data/split_val_train/val"
train_aug_dir = "data/train_augmented"
test_dir = "data/test"
ckpt_dir = "checkpoints"
ckpt_path = "checkpoints/checkpoint.weights.h5"
tuner_dir = "tuner_results"
tuner_project = "car_damage_tuner"
model_path = "models/best_model.keras"

# training
img_shape = (224, 224)
batch_size = 32
split_ratio = 0.1
aug_count = 3
epochs = 1
patience = 5
class_labels = ["minor", "moderate", "severe"]

# model
dropout = 0.2
l2 = 0.01
lr = 0.01
seed = 42
