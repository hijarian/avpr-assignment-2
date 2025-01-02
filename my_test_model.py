import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def test_model(model, test_loader, device):
    correct = 0
    total = len(test_loader.dataset)

    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            print(f'images: {images.shape} labels: {labels.shape}')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            matches = (predicted == labels).sum().item()
            print(f'outputs: {outputs.shape} matches: {matches}')
            correct += matches

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    # Confusion matrix

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(40)+1, yticklabels=np.arange(40)+1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return correct, total, accuracy, all_labels, all_predictions