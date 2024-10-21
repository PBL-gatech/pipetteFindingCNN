import cv2
import numpy as np
from PIL import Image

def custom_preprocess(image, image_size):
    # Convert the image to float32 (similar to im2single)
    image = image.astype(np.float32) / 255.0

    # Get image dimensions
    ysize, xsize = image.shape[:2]
    min_dimension = min(xsize, ysize)

    # Calculate cropping coordinates
    xmin = (xsize - min_dimension) // 2
    ymin = (ysize - min_dimension) // 2

    # Crop the image to a centered square
    cropped_image = image[ymin:ymin + min_dimension, xmin:xmin + min_dimension]

    # Calculate mean and standard deviation
    avg = np.mean(cropped_image)
    sigma = np.std(cropped_image)

    # Adjust pixel values to be within ±2 standard deviations
    n = 2
    min_val = max(0, avg - n * sigma)
    max_val = min(1, avg + n * sigma)
    adjusted_image = np.clip((cropped_image - min_val) / (max_val - min_val), 0, 1)

    # Resize the image to the target size using bilinear interpolation
    resized_image = cv2.resize(adjusted_image, image_size, interpolation=cv2.INTER_LINEAR)

    # Convert back to uint8 for further processing if needed
    final_image = (resized_image * 255).astype(np.uint8)

    return final_image
