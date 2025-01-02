import torch.nn as nn

CHANNELS_COUNT=4

class CustomCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(CHANNELS_COUNT, 32, kernel_size=3, stride=1, padding=1),  # 3x400x400 -> 32x400x400
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 32x400x400 -> 32x200x200

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 32x200x200 -> 64x200x200
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 64x200x200 -> 64x100x100

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# 64x100x100 -> 128x100x100
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 128x100x100 -> 128x50x50

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# 128x50x50 -> 256x50x50
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # 256x50x50 -> 256x25x25
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                         # 256x25x25 -> 256*25*25
            nn.Linear(256 * 25 * 25, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)                         # 1024 -> 40
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def make_model(device):
    """
    Make the custom CNN model compatible with resnet or other pretrained models.
    Configured for images with 3 input channels and 40 output channels.
    """
    return CustomCNN(num_classes=40).to(device)