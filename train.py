import torch
import torch.nn as nn

def training_loop(num_epochs, model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train_on_dataset(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss.item()}")

    print("Training complete.")

def train_on_dataset(model, dataloader, optimizer, criterion, device):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        loss = train_on_sample(model, images, labels, optimizer, criterion)

        return loss

def train_on_sample(model, images, labels, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss