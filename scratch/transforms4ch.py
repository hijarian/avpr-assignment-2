from torchvision import transforms

def get_transforms():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),         # Convert to PyTorch tensor
        # mean for bitmask should be 0, and std should be 1 so that we spread from 0 to 1
        transforms.Normalize([0.46824582, 0.44051382, 0.40057998, 0], [0.24759944, 0.23944832, 0.24277655, 1])  # Normalize RGB + bitmask channels
    ])
    return transform
