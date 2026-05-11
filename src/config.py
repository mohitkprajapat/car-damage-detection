import os
from dotenv import load_dotenv
load_dotenv()


# paths
root_dir = os.getenv("ROOT_DIR",os.getcwd())
data_dir = os.path.join(root_dir,"data")
training_dir = os.path.join(data_dir,"training")
split_dir = os.path.join(data_dir,"split_val_train")
train_dir = os.path.join(split_dir,"train")
val_dir = os.path.join(split_dir,"val")
train_aug_dir = os.path.join(data_dir,"train_augmented")
test_dir = os.path.join(data_dir,"test")
ckpt_dir = os.path.join(root_dir,"checkpoints")
ckpt_path = os.path.join(ckpt_dir,"checkpoint.weights.h5")
tuner_dir = os.path.join(root_dir,"tuner_results")
tuner_project = "car_damage_tuner"
models = os.path.join(root_dir,"models")
model_path = os.path.join(models,"best_model.keras")
upload_path = os.path.join(root_dir,"static","uploads")

# training
img_shape = (224, 224)
batch_size = 32
test_batch_size = 32
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
