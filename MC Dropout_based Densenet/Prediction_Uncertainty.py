import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models, transforms
import scipy.ndimage


# DenseNet
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


def sliding_window(data, window_size, stride):
    c, h, w = data.shape
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            yield data[:, i:i + window_size, j:j + window_size], i, j


def predict(data_path, model_path='final_model.pth', window_size=16, stride=1, T=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 加载数据
    print("Loading data...")
    data = np.load(data_path)
    data = torch.from_numpy(data).to(torch.float32).to(device)
    data = torch.nn.functional.pad(data, (8, 7, 8, 7), mode='constant', value=0)

    print(f"Data shape (original): {data.shape}")

    # 加载模型
    print("Loading model...")
    model = CustomDenseNetModel(num_classes=2, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.dropout.train()  # Dropout
    print("Model loaded and dropout enabled.")

    predictions = []
    probabilities = []
    uncertainties = []

    #
    print("Starting sliding window prediction...")
    for patch, i, j in tqdm(sliding_window(data, window_size, stride)):

        patch = torch.as_tensor(patch, dtype=torch.float32).unsqueeze(0).to(device).clone().detach()

        outputs = [model(patch) for _ in range(T)]
        outputs = torch.stack(outputs)
        probs = F.softmax(outputs, dim=2)
        pred_probs11 = probs[:, :, 1]  #

        pred_probs_mean11 = torch.mean(pred_probs11).item()

        uncertainty1 = torch.var(pred_probs11, dim=0).mean().item()

        mean_probs = probs.mean(dim=0)
        pred_class = mean_probs.argmax(dim=1).item()

        predictions.append(pred_class)
        probabilities.append(pred_probs_mean11)
        uncertainties.append(uncertainty1)
    result = np.array(probabilities).reshape(300, 300)
    np.savetxt("./Result_probability.txt", result, fmt='%.12f')
    uncertainty = np.array(uncertainties).reshape(300, 300)
    np.savetxt("./Result_uncertainty.txt", uncertainty, fmt='%.30f')

    return predictions, probabilities, uncertainties


if __name__ == "__main__":
    data_path = 'data_prediction.npy'
    model_path = 'final_mode.pth'
    predictions, probabilities, uncertainties = predict(data_path, model_path)

    for pred, prob, uncertainty in zip(predictions, probabilities, uncertainties):
        print(f"Prediction: {pred}, Probability: {prob:.8f}, Uncertainty: {uncertainty:.8f}")