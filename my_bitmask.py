import numpy as np
from PIL import Image, ImageDraw

def generate_bitmask(image_size, bbox):
        """
        Generate a bitmask from a bounding box annotation file.
        :param image_size: Size of the image (width, height).
        :param bbox: Tuple (xmin, ymin, xmax, ymax) representing the bounding box.
        :return: Bitmask as a NumPy array.
        """
        width, height = image_size
        mask = Image.new('L', (width, height), 0)  # Blank mask

        x_min, y_min, x_max, y_max = bbox
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x_min, y_min, x_max, y_max], fill=1)

        return np.array(mask, dtype=np.float32)