import torch.nn as nn
import torchvision.models as models

def make_model(device):
    resnet = models.resnet18(pretrained=True)  # Example with ResNet-18
    # Modify the final layer for our task (40 classes)
    resnet.fc = nn.Linear(resnet.fc.in_features, 40)
    return resnet.to(device)