import numpy as np

def transform_lr(image):
    """
    Flip the image left-to-right (horizontal flip).
    
    Parameters:
    image (numpy.ndarray): Input image.
    
    Returns:
    numpy.ndarray: Horizontally flipped image.
    """
    print("Flipping left-to-right...")
    return np.fliplr(image)

def transform_ud(image):
    """
    Flip the image up-and-down (vertical flip).
    
    Parameters:
    image (numpy.ndarray): Input image.
    
    Returns:
    numpy.ndarray: Vertically flipped image.
    """
    print("Flipping up-and-down...")
    return np.flipud(image)
