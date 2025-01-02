from torchvision import transforms

def get_transforms():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize images
        transforms.ToTensor(),         # Convert to PyTorch tensor
        transforms.Normalize([0.46824582, 0.44051382, 0.40057998], [0.24759944, 0.23944832, 0.24277655])  # Normalize RGB channels
    ])
    return transform

def get_resnet_transforms():
    # Define transformations
    transform = transforms.Compose([
        # resize to 224x224 for resnet
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),         # Convert to PyTorch tensor
        # normalize using ImageNet mean and std - not sure if this is the best choice
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize RGB channels
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
        transforms.Normalize([0.46824582, 0.44051382, 0.40057998], [0.24759944, 0.23944832, 0.24277655])  # Normalize RGB channels
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize RGB channels
    ])
    return transform