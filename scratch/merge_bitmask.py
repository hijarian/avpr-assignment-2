import torch

# Assuming image_tensor is a PyTorch tensor of shape (3, H, W)
# and bitmask is a NumPy array of shape (H, W)
bitmask_tensor = torch.tensor(bitmask).unsqueeze(0)  # Add channel dimension
image_with_mask = torch.cat((image_tensor, bitmask_tensor), dim=0)  # Combine
