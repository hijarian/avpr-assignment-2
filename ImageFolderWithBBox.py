import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from my_bbox import get_bounding_box
from my_bitmask import generate_bitmask

class ImageFolderWithBBox(ImageFolder):
    def __init__(self, root, augment=False):
        super(ImageFolderWithBBox, self).__init__(root)
        self.augment = augment

    def __getitem__(self, index):
        # Get the image and label from the parent class
        path, label = self.samples[index]
        image = self.loader(path)  # Load image using default loader

        image_id = os.path.basename(path).split('.')[0]
        # Get the corresponding bounding box annotation
        _, bbox = get_bounding_box(image_id)

        original_width, original_height = image.size

        resize = transforms.Resize((224, 224))

        # apply resize to the image
        image = resize(image)

        # Resize the bounding box to the same proportions
        resized_width, resized_height = image.size

        width_scale = resized_width / original_width
        height_scale = resized_height / original_height

        x_min, y_min, x_max, y_max = bbox

        x_min = int(x_min * width_scale)
        y_min = int(y_min * height_scale)
        x_max = int(x_max * width_scale)
        y_max = int(y_max * height_scale)

        bbox = (x_min, y_min, x_max, y_max)

        # must apply color jitter before converting to tensor and adding 4th channel
        if self.augment:
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            image = jitter(image)

        bitmask = generate_bitmask(image.size, bbox)

        # Convert bitmask to a tensor and add it as the 4th channel
        bitmask_tensor = torch.tensor(bitmask, dtype=torch.float32).unsqueeze(0)
        if not isinstance(image, torch.Tensor):  # Convert image to tensor if it's not already
            image = transforms.ToTensor()(image)
        image = torch.cat((image, bitmask_tensor), dim=0)

        # ToTensor normalizes pixels to [0, 1], but further augments don't care about that so it's safe to apply it here

        # coefficients tailored to the dataset
        # mean 0 and std 1 for the bitmask - don't change it
        mean = np.array([0.46824582, 0.44051382, 0.40057998, 0])
        std = np.array([0.24759944, 0.23944832, 0.24277655, 1])
        normalize = transforms.Normalize(mean=mean, std=std)

        if self.augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                normalize
            ])
        else:
            transform = normalize

        image = transform(image)

        # Apply transformations to the label (if any)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

