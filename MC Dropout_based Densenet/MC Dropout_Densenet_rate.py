import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
import os
import pandas as pd
import random


data_path = 'data/sample15+15.csv'
data = pd.read_csv(data_path)
X = data[['X', 'Y']].values
y = data['Label'].values

npy_path = 'SGS200-data.npy'
image_data = np.load(npy_path)


def extract_patches_centered(X_coords, y_coords, labels, images):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = []

    for img in images:
        for (x, y, label) in zip(X_coords, y_coords, labels):
            xi = int(x) - 8
            yi = int(y) - 8

            if 0 <= xi <= 284 and 0 <= yi <= 284:
                patch = img[yi:yi + 16, xi:xi + 16]
                if patch.shape[0] == 16 and patch.shape[1] == 16:
                    dataset.append((transform(patch), label))

    return dataset


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


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_loss = val_loss
        torch.save(model.state_dict(), self.path)


def save_metrics(metrics, filename='metrics.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Dropout Rate', 'Fold', 'Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Val AUC'])
        for metric in metrics:
            writer.writerow(metric)


def train_model(X, y, image_data, num_epochs=30, batch_size=32, lr=0.001, k_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    y_data = np.array([y for y in y])

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_dropout_rate = 0
    best_val_auc = 0
    dropout_auc = []
    metrics = []
    aucs_per_dropout_rate = {rate: [] for rate in dropout_rates}

    for dropout_rate in dropout_rates:
        fold_results = []
        images_subset = random.sample(list(image_data), 200)  # 随机抽取200张图像
        patches_subset = extract_patches_centered(X[:, 0], X[:, 1], y, images_subset)
        x_data = [x[0].numpy() for x in patches_subset]
        y_data = [x[1] for x in patches_subset]

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_data, y_data)):
            print(f'Fold {fold + 1}/{k_folds}')

            train_subset = Subset(CustomDataset(x_data, y_data), train_idx)
            val_subset = Subset(CustomDataset(x_data, y_data), val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CustomDenseNetModel(num_classes=2, dropout_rate=dropout_rate).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8,
                                                             verbose=True)
            early_stopping = EarlyStopping(patience=5, delta=0.01)

            fold_result = []
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

                metrics.append(
                    [dropout_rate, fold + 1, epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, val_auc])

                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")

                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                fold_result.append(val_auc)

            avg_val_auc = np.mean(fold_result)
            fold_results.append(avg_val_auc)
            aucs_per_dropout_rate[dropout_rate].append(avg_val_auc)

            print(f"Fold {fold + 1}/{k_folds} Average Validation AUC: {avg_val_auc:.4f}")

        avg_dropout_auc = np.mean(fold_results)
        print(f'Dropout Rate: {dropout_rate}, Average Validation AUC: {avg_dropout_auc:.4f}')

        if avg_dropout_auc > best_val_auc:
            best_val_auc = avg_dropout_auc
            best_dropout_rate = dropout_rate

        dropout_auc.append(avg_dropout_auc)

    save_metrics(metrics)
    print(f'Best Dropout Rate: {best_dropout_rate}, Best Validation AUC: {best_val_auc:.4f}')


    plot_auc_vs_dropout(aucs_per_dropout_rate)


def plot_auc_vs_dropout(aucs_per_dropout_rate):
    plt.figure(figsize=(10, 8))

    dropout_rates, all_aucs = zip(*aucs_per_dropout_rate.items())

    plt.gca().set_facecolor('#D3D3D3')

    for dropout_rate, aucs in zip(dropout_rates, all_aucs):
        plt.scatter([dropout_rate] * len(aucs), aucs, color='blue', alpha=0.8, marker='+', zorder=3)

    avg_aucs = [np.mean(aucs) for aucs in all_aucs]

    if len(dropout_rates) >= 4:
        xnew = np.linspace(min(dropout_rates), max(dropout_rates), 300)
        f = interp1d(dropout_rates, avg_aucs, kind='linear')
        ynew = f(xnew)

        plt.plot(xnew, ynew, color='orange', linewidth=2, zorder=2)
    else:
        plt.plot(dropout_rates, avg_aucs, color='orange', linewidth=2, zorder=2)

    plt.xlabel('Dropout Rate', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    plt.grid(True, linestyle='--', color='white', alpha=0.7, zorder=1)

    plt.xticks(dropout_rates, fontsize=14)
    plt.yticks(fontsize=14)

    plt.tick_params(axis='both', which='both', width=1)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)

    plt.savefig(f'dropout_rates.png', format='png', bbox_inches='tight', dpi=600)
    plt.savefig(f'dropout_rates.svg', format='svg', bbox_inches='tight', dpi=600)

    plt.show()


if __name__ == "__main__":
    train_model(X, y, image_data)
