import os
import random
import shutil

# Define paths
dataset_path = "dataset"
train_path = "dataset/train"
test_path = "dataset/test"

# Create directories
for category in ["wanted", "unwanted"]:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

# Split data
split_ratio = 0.8

for category in ["wanted", "unwanted"]:
    images = os.listdir(os.path.join(dataset_path, category))
    random.shuffle(images)

    split = int(len(images) * split_ratio)
    train_images, test_images = images[:split], images[split:]

    for img in train_images:
        shutil.move(os.path.join(dataset_path, category, img), os.path.join(train_path, category, img))
    
    for img in test_images:
        shutil.move(os.path.join(dataset_path, category, img), os.path.join(test_path, category, img))

print("Dataset successfully split into training and testing sets.")
