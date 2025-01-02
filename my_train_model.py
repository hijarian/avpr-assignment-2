import torch
import torch.nn as nn

def training_loop(num_epochs, model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} started.")
        train_loss = train_on_dataset(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} completed. Loss: {train_loss.item()}")

    print("Training complete.")

def train_on_dataset(model, dataloader, optimizer, criterion, device):
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        loss = train_on_batch(model, images, labels, optimizer, criterion)
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # watch for the proper tabulation!!!
    return loss

def train_on_batch(model, images, labels, optimizer, criterion):
    # see https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

    # Calculate prediction error
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss