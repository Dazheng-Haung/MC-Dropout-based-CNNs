import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import csv
import os


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data=None):
        self.x_data = [torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32) for x in x_data]
        if y_data is not None:
            self.y_data = [torch.tensor(y, dtype=torch.long) for y in y_data]
        else:
            self.y_data = None

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.y_data is not None:
            return self.x_data[idx], self.y_data[idx]
        else:
            return self.x_data[idx]


class CustomDenseNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CustomDenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.densenet.features.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.relu0(x)
        x = self.densenet.features.pool0(x)

        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        x = self.dropout(x)

        x = self.densenet.features.denseblock2(x)
        x = self.densenet.features.transition2(x)
        x = self.dropout(x)

        x = self.densenet.features.denseblock3(x)
        x = self.densenet.features.transition3(x)
        x = self.dropout(x)

        x = self.densenet.features.denseblock4(x)
        x = self.densenet.features.norm5(x)

        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.densenet.classifier(x)
        return x


def save_metrics(metrics, filename='metrics.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Val AUC'])
        for metric in metrics:
            writer.writerow(metric)


def train_model(x_data, y_data, num_epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    y_data = np.array([y for y in y_data])

    dataset = CustomDataset(x_data, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomDenseNetModel(num_classes=2, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8, verbose=True)

    metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        val_accuracy = correct / total
        val_auc = roc_auc_score(all_labels, all_probs)

        metrics.append([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, val_auc])

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    save_metrics(metrics)
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete. Final model saved as 'final_model.pth'.")


if __name__ == "__main__":
    x_data = np.load('X_AsAuSbHg100.npy')
    y_data = np.load('Y_AsAuSbHg100.npy')

    train_model(x_data, y_data)
