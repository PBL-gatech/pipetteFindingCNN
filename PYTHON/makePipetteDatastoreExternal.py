import os
import glob
import random
import numpy as np
from PIL import Image
import custom_process  # Import the custom preprocessing function from custom_process.py
import transform  # Import the flipping functions from transform.py

# Directory setup (adjust these paths as needed)
RAW_ROOT_PATH = "/path/to/raw/images/"  # Directory containing raw images
NEW_ROOT = "/path/to/processed/images/"  # Directory for storing processed images
IMG_SIZE = (224, 224)  # Target image size for resizing
PERC_VAL = 0.1  # Percentage of images to be used for validation

# Create the output directory if it doesn't exist
if not os.path.exists(NEW_ROOT):
    os.makedirs(NEW_ROOT)

# Fetch all image paths
image_paths = glob.glob(os.path.join(RAW_ROOT_PATH, "*.jpg"))  # Adjust extension as needed for your image format

# Split dataset into training and validation
n_images = len(image_paths)
n_valid = int(n_images * PERC_VAL)
n_train = n_images - n_valid

# Shuffle images for random distribution between train and validation
random.shuffle(image_paths)
train_paths = image_paths[:n_train]
valid_paths = image_paths[n_train:]

# Function to process and save images
def process_and_save_images(image_paths, output_dir, augment=True):
    for img_path in image_paths:
        # Load the image
        img = Image.open(img_path)

        # Apply custom preprocessing (resize and contrast adjustment)
        processed_img = custom_process.custom_preprocess(np.array(img), IMG_SIZE)

        # Save the preprocessed image
        base_name = os.path.basename(img_path)
        processed_path = os.path.join(output_dir, base_name)
        Image.fromarray(processed_img).save(processed_path)

        # Data augmentation: left-right and up-down flipping
        if augment:
            # Horizontal flip
            flipped_lr = transform.transform_lr(processed_img)
            lr_path = os.path.join(output_dir, f"LR_{base_name}")
            Image.fromarray(flipped_lr).save(lr_path)

            # Vertical flip
            flipped_ud = transform.transform_ud(processed_img)
            ud_path = os.path.join(output_dir, f"UD_{base_name}")
            Image.fromarray(flipped_ud).save(ud_path)

            # Both flips (left-right and up-down)
            flipped_lr_ud = transform.transform_ud(flipped_lr)
            lr_ud_path = os.path.join(output_dir, f"LR_UD_{base_name}")
            Image.fromarray(flipped_lr_ud).save(lr_ud_path)

# Create subdirectories for training and validation datasets
train_output_dir = os.path.join(NEW_ROOT, "train")
valid_output_dir = os.path.join(NEW_ROOT, "validation")
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(valid_output_dir, exist_ok=True)

# Process training images with augmentation
print("Processing training images...")
process_and_save_images(train_paths, train_output_dir, augment=True)

# Process validation images without augmentation
print("Processing validation images...")
process_and_save_images(valid_paths, valid_output_dir, augment=False)

print("Data preprocessing complete. Training and validation datasets have been saved.")
