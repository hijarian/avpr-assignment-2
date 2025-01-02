import torch
import torch.nn as nn

import torchvision.models as models

def make_model(device):
    resnet = models.resnet18(pretrained=True)  # Example with ResNet-101

    # Get the original weights of the first layer
    original_conv1 = resnet.conv1
    new_conv1 = nn.Conv2d(
        in_channels=4,  # Change to 4 channels
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )

    # Copy pretrained weights for the first 3 channels (RGB)
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = original_conv1.weight
        new_conv1.weight[:, 3:, :, :] = 0  # Initialize weights for the new channel to 0

    # Replace the first layer in ResNet
    resnet.conv1 = new_conv1

    # Modify the final layer for our task (40 classes)
    resnet.fc = nn.Linear(resnet.fc.in_features, 40)
    
    return resnet.to(device)
