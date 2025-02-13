"""
utils.py
----------
This module contains classes for image preprocessing, data augmentation,
and visualization.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageProcessor:
    @staticmethod
    def custom_preprocess(image, image_size=(224, 224)):
        """
        Preprocess the image by:
          - Converting to float and scaling to [0,1]
          - Center-cropping to a square
          - Adjusting contrast (clipping to ±2 standard deviations)
          - Resizing to the target image_size
          - Converting back to uint8
        """
        # If the image is a PIL Image, convert to numpy array.
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0

        ysize, xsize = image.shape[:2]
        min_dimension = min(xsize, ysize)
        xmin = (xsize - min_dimension) // 2
        ymin = (ysize - min_dimension) // 2
        cropped_image = image[ymin:ymin + min_dimension, xmin:xmin + min_dimension]

        # Adjust contrast based on mean and std deviation.
        avg = np.mean(cropped_image)
        sigma = np.std(cropped_image)
        n = 2
        min_val = max(0, avg - n * sigma)
        max_val = min(1, avg + n * sigma)
        adjusted_image = np.clip((cropped_image - min_val) / (max_val - min_val), 0, 1)

        # Resize the image.
        resized_image = cv2.resize(adjusted_image, image_size, interpolation=cv2.INTER_LINEAR)
        final_image = (resized_image * 255).astype(np.uint8)
        return final_image

    @staticmethod
    def transform_lr(image):
        """
        Flip the image left-to-right.
        """
        return np.fliplr(image)

    @staticmethod
    def transform_ud(image):
        """
        Flip the image up-and-down.
        """
        return np.flipud(image)


class Visualizer:
    @staticmethod
    def visualize_prediction(image, real_coords, pred_coords, marker_size=9):
        """
        Visualize the real and predicted coordinates on the image.
        Draws a circle (cyan) for real_coords and a circle (red) for pred_coords.

        Parameters:
          image: a PIL Image
          real_coords: (x, y) tuple for ground truth (if available)
          pred_coords: (x, y) tuple for prediction
        """
        plt.figure()
        plt.imshow(image)
        ax = plt.gca()

        # Draw a circle for the real coordinate.
        real_circle = plt.Circle(real_coords, marker_size, color='cyan',
                                 fill=False, linewidth=2)
        ax.add_patch(real_circle)

        # Draw a circle for the predicted coordinate.
        pred_circle = plt.Circle(pred_coords, marker_size, color='red',
                                 fill=False, linewidth=2)
        ax.add_patch(pred_circle)

        plt.title("Real (cyan) vs Predicted (red)")
        plt.axis('off')
        plt.show()
