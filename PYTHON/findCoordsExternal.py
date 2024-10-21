import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import custom_process  # Assuming custom_preprocess function is in custom_process.py

def find_coords_external(image, model=None):
    """
    Predict the coordinates of the pipette tip in the given image.

    Parameters:
    - image (PIL.Image or numpy.ndarray): Input image.
    - model: Pre-trained model for predicting pipette tip coordinates.

    Returns:
    - guess (list): Predicted coordinates (x, y, z).
    """
    # Set the new dimension and preprocess the image
    new_dimension = 224
    img_size = (new_dimension, new_dimension)
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Preprocess image using custom function
    pipette_img = custom_process.custom_preprocess(image, img_size)

    # Save the temporary image to a folder
    temp_folder = "./temp_images/"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    temp_path = os.path.join(temp_folder, f"{time.time()}.png")
    Image.fromarray(pipette_img).save(temp_path)

    # Dummy prediction (replace with actual model inference later)
    if model:
        # Assuming the model takes an image tensor and returns x, y, z coordinates
        # For example: guess = model.predict(image_tensor)
        guess = model.predict(temp_path)  # Placeholder for actual model inference
    else:
        # Placeholder coordinates as dummy prediction
        guess = [112, 112, 0]  # Center point in the 224x224 image

    # Transform the point to match the original image
    ysize, xsize = image.shape[:2]
    min_dimension = min(xsize, ysize)

    transformed_point = [
        new_dimension * ((guess[0] + (xsize / 2)) - (xsize - min_dimension) / 2) / min_dimension,
        new_dimension * ((guess[1] + (ysize / 2)) - (ysize - min_dimension) / 2) / min_dimension,
    ]

    # Convert the guess to microns
    guess_um = [coord * 0.1 / 1.093 for coord in guess]

    # Plot the predicted point on the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.scatter([transformed_point[0]], [transformed_point[1]], c='red', marker='x', s=100)
    plt.title("Predicted Pipette Tip Location")
    plt.show()

    return guess_um

# Example usage
if __name__ == "__main__":
    # Load an example image (replace with the actual path)
    image_path = "/path/to/sample_image.jpg"
    image = Image.open(image_path)

    # Call the function (without a model for now)
    find_coords_external(image)
