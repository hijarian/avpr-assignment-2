import torch
import numpy as np
from PIL import Image, ImageDraw

def create_bitmask(image_size, bbox):
    """
    Create a bitmask for a given bounding box.
    :param image_size: (width, height) of the image
    :param bbox: (x_min, y_min, x_max, y_max) bounding box coordinates
    :return: Bitmask as a tensor
    """
    mask = Image.new('L', image_size, 0)  # Create a blank mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=1)  # Fill the bounding box
    return torch.tensor(np.array(mask), dtype=torch.float32)

# Example usage:
image_size = (224, 224)
bbox = (50, 50, 150, 150)  # Example bounding box
bitmask = create_bitmask(image_size, bbox).unsqueeze(0)  # Add channel dimension


