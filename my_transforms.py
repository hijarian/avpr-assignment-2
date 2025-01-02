from torchvision import transforms
import numpy as np

# coefficients tailored to the dataset
# mean 0 and std 1 for the bitmask - don't change it
mean4ch = np.array([0.46824582, 0.44051382, 0.40057998, 0])
std4ch = np.array([0.24759944, 0.23944832, 0.24277655, 1])
normalize4ch = transforms.Normalize(mean=mean4ch, std=std4ch)

mean = np.array([0.46824582, 0.44051382, 0.40057998])
std = np.array([0.24759944, 0.23944832, 0.24277655])
normalize = transforms.Normalize(mean=mean, std=std)

def get_transforms():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize images
        transforms.ToTensor(),         # Convert to PyTorch tensor
        normalize
    ])
    return transform

def get_resnet_transforms():
    # Define transformations
    transform = transforms.Compose([
        # resize to 224x224 for resnet
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),         # Convert to PyTorch tensor
        # normalize using ImageNet mean and std - not sure if this is the best choice
        normalize
    ])
    return transform

def get_augmented_transforms():
    # Define transformations with augmentation
    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize images
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(10),  # Randomly rotate images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
        transforms.ToTensor(),         # Convert to PyTorch tensor
        normalize
    ])
    return transform

def get_resnet_augmented_transforms():
    # Define transformations with augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(10),  # Randomly rotate images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
        transforms.ToTensor(),         # Convert to PyTorch tensor
        normalize
    ])
    return transform