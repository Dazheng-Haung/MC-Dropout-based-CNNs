import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import models, transforms
import matplotlib.pyplot as plt
import random

data_path = 'data/sample15+15.csv'
data = pd.read_csv(data_path)
X = data[['X', 'Y']].values
y = data['Label'].values

npy_path = 'SGS200-data.npy'
image_data = np.load(npy_path)


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
    def __init__(self, data):
        self.x_data = [x[0] for x in data]
        self.y_data = [x[1] for x in data]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx].float(), self.y_data[idx]


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


def train_and_evaluate_model(train_data, val_data, dropout_rate, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomDenseNetModel(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    best_auc = 0
    best_model_wts = None
    fold_results = []
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

        val_labels = []
        val_preds = []
        val_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        val_accuracy = correct / total
        thresholds = np.arange(0.0, 1.1, 0.1)
        accuracies = []
        for threshold in thresholds:
            val_pred_labels = (np.array(val_probs) >= threshold).astype(int)
            accuracy = accuracy_score(val_labels, val_pred_labels)
            accuracies.append(accuracy)

        best_threshold = thresholds[np.argmax(accuracies)]
        best_val_accuracy = np.max(accuracies)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}, "
              f"Best Threshold: {best_threshold:.2f}, Best Val Accuracy: {best_val_accuracy:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        fold_results.append(val_auc)
        avg_val_auc = np.mean(fold_results)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = model.state_dict()

    return best_auc, best_model_wts, avg_val_auc, fold_results


def cross_validate_and_select_model(X, y, image_data, num_folds1=5, num_folds2=2,
                                    num_images_list=[1, 5, 10, 20, 35, 50, 75, 100, 125, 150, 175, 200],
                                    dropout_rate=0.3, num_epochs1=30, batch_size=32, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    skf1 = StratifiedKFold(n_splits=num_folds1, shuffle=True, random_state=42)
    best_num_images = 0
    best_auc = 0
    best_model_wts = None

    aucs_per_num_images = []

    for num_images in num_images_list:
        fold_aucs = []
        images_subset = random.sample(list(image_data), num_images)
        patches_subset = extract_patches_centered(X[:, 0], X[:, 1], y, images_subset)

        for fold, (train_index, val_index) in enumerate(
                skf1.split(patches_subset, [label for _, label in patches_subset])):
            print(f'Fold {fold + 1}, Num Images: {num_images}')
            print(f'Train labels distribution: {np.bincount([patches_subset[i][1] for i in train_index])}')
            print(f'Validation labels distribution: {np.bincount([patches_subset[i][1] for i in val_index])}')

            train_data = [patches_subset[i] for i in train_index]
            val_data = [patches_subset[i] for i in val_index]

            _, _, avg_val_auc, fold_results = train_and_evaluate_model(train_data, val_data, dropout_rate, num_epochs1,
                                                                       batch_size, lr)
            fold_aucs.append(avg_val_auc)

        aucs_per_num_images.append((num_images, fold_aucs))
        avg_auc = np.mean(fold_aucs)
        print(f'Num Images: {num_images}, Average AUC: {avg_auc:.4f}')

        if avg_auc > best_auc:
            best_auc = avg_auc
            best_num_images = num_images
    print(f'Best Number of Realization: {best_num_images}, Best Validation AUC: {best_auc:.4f}')
    images_subset = random.sample(list(image_data), best_num_images)
    patches_subset = extract_patches_centered(X[:, 0], X[:, 1], y, images_subset)

    return aucs_per_num_images


if __name__ == "__main__":
    aucs_per_num_images = cross_validate_and_select_model(X, y, image_data)

    # plot
    plt.figure(figsize=(10, 8))

    num_images_list, all_aucs = zip(*aucs_per_num_images)

    plt.gca().set_facecolor('#D3D3D3')

    for num_images, aucs in zip(num_images_list, all_aucs):
        plt.scatter([num_images] * len(aucs), aucs, color='blue', alpha=0.8, marker='+', zorder=3)
        avg_aucs = [np.mean(aucs) for aucs in all_aucs]

    plt.plot(num_images_list, avg_aucs, marker='o', color='orange', linestyle='-', linewidth=2, markersize=2.5,
             zorder=2)

    plt.xlabel('Number of Realization', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    #plt.title('Cross-validation', fontsize=16)
    plt.grid(True, linestyle='--', color='white', alpha=0.7, zorder=1)

    plt.xticks(num_images_list, fontsize=14)
    plt.yticks(fontsize=14)

    plt.tick_params(axis='both', which='both', width=1)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)

    plt.show()


from scipy.interpolate import interp1d

# plot
plt.figure(figsize=(10, 8))

num_images_list, all_aucs = zip(*aucs_per_num_images)

plt.gca().set_facecolor('#D3D3D3')

for num_images, aucs in zip(num_images_list, all_aucs):
    plt.scatter([num_images] * len(aucs), aucs, color='blue', alpha=0.8, marker='+', zorder=3)

avg_aucs = [np.mean(aucs) for aucs in all_aucs]

if len(num_images_list) >= 4:
    xnew = np.linspace(min(num_images_list), max(num_images_list), 300)
    f = interp1d(num_images_list, avg_aucs, kind='linear')
    ynew = f(xnew)

    plt.plot(xnew, ynew, color='orange', linewidth=2,  zorder=2)
else:
    plt.plot(num_images_list, avg_aucs, color='orange', linewidth=2, zorder=2)

plt.xlabel('Number of Realization', fontsize=16)
plt.ylabel('AUC', fontsize=16)
#plt.title('Cross-validation', fontsize=16)
plt.grid(True, linestyle='--', color='white', alpha=0.7, zorder=1)


plt.xticks(num_images_list, fontsize=14)
plt.yticks(fontsize=14)

plt.tick_params(axis='both', which='both', width=1)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)


plt.savefig(f'Realization Number.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(f'Realization Number.svg', format='svg', dpi=600, bbox_inches='tight')
plt.show()
