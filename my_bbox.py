import os
import xml.etree.ElementTree as ET
import my_paths

def get_bounding_box(image_id):
    xml_path = os.path.join(my_paths.annotations_path, image_id + '.xml')

    if not os.path.exists(xml_path):
        raise ValueError(f"XML file for {image_id} not found at {xml_path}")

    # Parse and print the contents of the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bbox = root.find('object/bndbox')

    if bbox is None:
        raise ValueError(f"No bounding box found in {xml_path}")

    box = [
        float(bbox.find('xmin').text),
        float(bbox.find('ymin').text),
        float(bbox.find('xmax').text),
        float(bbox.find('ymax').text)
    ]
    size = root.find('size')
    if size is None:
        raise ValueError(f"No size found in {xml_path}")

    depth = size.find('depth')
    if depth is None or depth.text != '3':
        print(f"Image {image_id} is not a 3-channel image: {depth if depth is None else depth.text}")

    size = [
        float(size.find('width').text),
        float(size.find('height').text)
    ]

    return size, box