import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image preprocessing (rescale & augment training data)
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Augment: rotate images
    width_shift_range=0.1,   # Augment: shift images horizontally
    height_shift_range=0.1,  # Augment: shift images vertically
    horizontal_flip=True,    # Augment: flip images
    fill_mode="nearest"      # Handle empty pixels after transformations
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize test images

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize all images
    batch_size=32,
    class_mode="binary"  # Binary classification (wanted/unwanted)
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)
