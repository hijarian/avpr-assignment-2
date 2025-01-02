from PIL import Image, ImageDraw
import numpy as np

def create_bitmask(image_size, bbox):
    """
    Create a bitmask for a given bounding box.
    :param image_size: (width, height) of the image
    :param bbox: (x_min, y_min, x_max, y_max) bounding box coordinates
    :return: Bitmask as a NumPy array
    """
    mask = Image.new('L', image_size, 0)  # Create a blank mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=1)  # Fill the bounding box
    return np.array(mask, dtype=np.float32)

# Example usage:
image_size = (400, 400)  # Replace with actual image size
bbox = (50, 50, 350, 350)  # Example bounding box
bitmask = create_bitmask(image_size, bbox)
