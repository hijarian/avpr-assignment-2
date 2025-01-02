import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ask for an image from user input (show the OS image picker dialog)
# from the image filename, go to `../XMLAnnotations/` folder and find the `.xml` file with the same filename
# print the contents of this XML file.
import xml.etree.ElementTree as ET
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
    draw.rectangle(bbox, fill=255)  # Fill the bounding box
    return mask

# Hide the root window
Tk().withdraw()

# Ask for an image file
image_path = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if not image_path:
    raise ValueError("No image file selected.")

# Get the filename without extension
image_filename = os.path.splitext(os.path.basename(image_path))[0]

# Construct the path to the XML file
xml_path = os.path.join(os.path.dirname(image_path), '../XMLAnnotations', image_filename + '.xml')

if not os.path.exists(xml_path):
    raise ValueError(f"XML file for {image_filename} not found.")

# Parse and print the contents of the XML file
tree = ET.parse(xml_path)
root = tree.getroot()
ET.dump(root)
bbox = root.find('object/bndbox')

if bbox is None:
    raise ValueError(f"No bounding box found in {xml_path}")

box = [
    float(bbox.find('xmin').text),
    float(bbox.find('ymin').text),
    float(bbox.find('xmax').text),
    float(bbox.find('ymax').text)
]
# Read the image
image = Image.open(image_path)

mask = create_bitmask(image.size, box)

bitmask_folder = os.path.join(os.path.dirname(image_path), '../Bitmasks')
os.makedirs(bitmask_folder, exist_ok=True)

# write the bitmask to a file
mask.save(os.path.join(bitmask_folder, image_filename + '.png'))

# show the image with the bounding box drawn on it
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(image)

# Create a Rectangle patch
rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

# Show the plot
plt.show()

