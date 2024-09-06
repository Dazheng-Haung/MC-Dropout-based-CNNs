import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns


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


def train_and_evaluate_model(X_train, y_train, X_val, y_val, images, dropout_rate, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = extract_patches_centered(X_train[:, 0], X_train[:, 1], y_train, images)
    val_data = extract_patches_centered(X_val[:, 0], X_val[:, 1], y_val, images)
    print(f"train_data shape: {len(train_data)}, first element shape: {train_data[0][0].shape}")
    print(f"val_data shape: {len(val_data)}, first element shape: {val_data[0][0].shape}")
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomDenseNetModel(num_classes=2, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    #early_stopping = EarlyStopping(patience=5, delta=0.01)

    best_model_wts = None
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

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        val_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if epoch == num_epochs - 1:
            best_model_wts = model.state_dict()

    return best_model_wts


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def compute_mc_dropout_predictions(model, dataloader, num_samples=15):
    model.eval()  # Set the model to evaluation mode to disable batch normalization
    enable_dropout(model)  # Enable dropout during test-time
    device = next(model.parameters()).device
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for _ in range(num_samples):
            probs = []
            labels = []
            for inputs, label in dataloader:
                inputs = inputs.to(device)
                outputs = F.softmax(model(inputs), dim=1)[:, 1]
                probs.append(outputs.cpu().numpy())
                labels.append(label.numpy())
            all_probs.append(np.concatenate(probs))
            all_labels = np.concatenate(labels)

    mean_probs = np.mean(all_probs, axis=0)
    uncertainty = np.var(all_probs, axis=0)
    final_preds = (mean_probs > 0.5).astype(int)

    return mean_probs, uncertainty, all_labels, final_preds


def evaluate_mc_dropout(model_path, X_val, y_val, images, num_images=100, batch_size=32, num_samples=15, fold=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomDenseNetModel(num_classes=2, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    val_data = extract_patches_centered(X_val[:, 0], X_val[:, 1], y_val, images[:num_images])
    val_dataset = CustomDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    mean_probs, uncertainty, labels, preds = compute_mc_dropout_predictions(model, val_loader, num_samples=num_samples)

    results = pd.DataFrame({
        'probability': mean_probs,
        'uncertainty': uncertainty,
        'true_label': labels,
        'predicted_label': preds
    })

    results.to_csv(f'mc_dropout_results_fold_{fold + 1}.csv', index=False)

    return results


def cross_validate_and_train_final_model(X, y, image_data, num_folds=5,
                                         num_images=100, dropout_rate=0.3, num_epochs=20, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    images_subset = image_data[:num_images]

    best_model_wts_list = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f'Training on fold {fold + 1}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        best_model_wts = train_and_evaluate_model(X_train, y_train, X_val, y_val, images_subset, dropout_rate, num_epochs, batch_size, lr)
        best_model_wts_list.append(best_model_wts)

        model_path = f'best_densenet_model_fold_{fold + 1}_{num_images}_images.pth'
        torch.save(best_model_wts, model_path)
        print(f'Model saved for fold {fold + 1} with num images: {num_images}')

    return best_model_wts_list


def plot_uncertainty_density(results, fold):
    correct = results[results['true_label'] == results['predicted_label']]
    incorrect = results[results['true_label'] != results['predicted_label']]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    sns.kdeplot(correct['uncertainty'], fill=True, color='g', label='Correct', ax=ax1)
    sns.kdeplot(incorrect['uncertainty'], fill=True, color='r', label='Incorrect', ax=ax1)

    ax1.set_xlabel('Uncertainty (σ)', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title(f'Uncertainty of Predictions (Fold {fold + 1})', fontsize=16)
    ax1.grid(True)

    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax2 = ax1.twinx()
    thresholds = np.linspace(0, results['uncertainty'].max(), 100)
    youdens_j = []

    for threshold in thresholds:
        tpr = ((results['uncertainty'] <= threshold) & (results['true_label'] == results['predicted_label'])).sum() / (results['true_label'] == results['predicted_label']).sum()
        fpr = ((results['uncertainty'] <= threshold) & (results['true_label'] != results['predicted_label'])).sum() / (results['true_label'] != results['predicted_label']).sum()
        youdens_j.append(tpr - fpr)

    optimal_threshold = thresholds[np.argmax(youdens_j)]
    ax2.plot(thresholds, youdens_j, 'k--', label="Youden's J")
    ax2.set_yticklabels([])
    ax2.set_ylabel("")
    ax2.yaxis.set_ticks([])
    ax2.axvline(x=optimal_threshold, color='k', linestyle='--', linewidth=1, dashes=(2, 2))

    ax1.text(optimal_threshold + 0.001, ax1.get_ylim()[1] * 0.9, f'Optimal θ: {optimal_threshold:.4f}',
             horizontalalignment='left', color='black', fontsize=12)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.85, 0.85), fontsize=12)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    fig.savefig(f'uncertainty_density_plot_fold_{fold + 1}.png', dpi=600)

    fig.savefig(f'uncertainty_density_plot_fold_{fold + 1}.svg', dpi=600)
    fig.savefig(f'uncertainty_density_plot_fold_{fold + 1}.pdf', dpi=600)
    plt.show()

    return optimal_threshold


def plot_scatter(results, fold):
    optimal_threshold = plot_uncertainty_density(results, fold)
    high_confidence = results['uncertainty'] <= optimal_threshold
    low_confidence = results['uncertainty'] > optimal_threshold

    correct_high_confidence = results[(results['true_label'] == results['predicted_label']) & high_confidence]
    incorrect_high_confidence = results[(results['true_label'] != results['predicted_label']) & high_confidence]
    correct_low_confidence = results[(results['true_label'] == results['predicted_label']) & low_confidence]
    incorrect_low_confidence = results[(results['true_label'] != results['predicted_label']) & low_confidence]

    plt.figure(figsize=(8, 6))
    plt.scatter(incorrect_high_confidence['probability'], incorrect_high_confidence['uncertainty'], color='red',
                label='Incorrect, high-confidence', s=10, marker='x')
    plt.scatter(correct_high_confidence['probability'], correct_high_confidence['uncertainty'], color='blue',
                label='Correct, high-confidence', s=10)
    plt.scatter(correct_low_confidence['probability'], correct_low_confidence['uncertainty'], color='gray',
                label='Correct, low-confidence', s=10)
    plt.scatter(incorrect_low_confidence['probability'], incorrect_low_confidence['uncertainty'], color='pink',
                label='Incorrect, low-confidence', s=10, marker='x')
    plt.legend(fontsize=12)

    plt.axhline(y=optimal_threshold, color='red', linestyle='--', label='Uncertainty threshold')
    plt.xlabel('Probability', fontsize=16)
    plt.ylabel('Uncertainty', fontsize=16)
    plt.title(f'Probability vs Uncertainty (Fold {fold + 1})', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tick_params(axis='both', which='major', labelsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.savefig(f'scatter_plot_fold_{fold + 1}.png', dpi=600)
    plt.savefig(f'scatter_plot_fold_{fold + 1}.svg', dpi=600)
    plt.savefig(f'scatter_plot_fold_{fold + 1}.pdf', dpi=600)
    plt.show()


if __name__ == "__main__":
    data_path = 'data/sample15+15.csv'
    data = pd.read_csv(data_path)
    X = data[['X', 'Y']].values
    y = data['Label'].values

    npy_path = 'SGS200-data.npy'
    image_data = np.load(npy_path)

    num_images = 100

    best_model_wts_list = cross_validate_and_train_final_model(X, y, image_data)

    for fold in range(5):
        model_path = f'best_densenet_model_fold_{fold + 1}_{num_images}_images.pth'
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
            if fold_idx == fold:
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                results = evaluate_mc_dropout(model_path, X_val, y_val, image_data, num_images=num_images, fold=fold)
                plot_uncertainty_density(results, fold)
                plot_scatter(results, fold)
                print(f"Results and plot saved for fold {fold + 1}")
                break
