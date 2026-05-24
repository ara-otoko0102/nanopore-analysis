import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score

# ======================
# Dataset
# ======================
class WaveformDataset(Dataset):
    def __init__(self, data, labels, max_len=1000):
        filtered_data = []
        filtered_labels = []
        for w, y in zip(data, labels):
            if len(w) <= max_len:
                filtered_data.append(w)
                filtered_labels.append(y)

        self.data = filtered_data
        self.labels = filtered_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return waveform, label


# ======================
# Model
# ======================
class WaveformCNN(nn.Module):
    def __init__(self):
        super(WaveformCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


# ======================
# Data loader function
# ======================
def load_and_preprocess(folder, label):
    data, labels = [], []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            wave = df.iloc[:, 0].dropna().values
            wave = wave / 1e3
            wave = wave - wave[0]
            data.append(wave)
            labels.append(label)

    return data, labels


def create_dataset(folder_pairs):
    X, Y = [], []
    for folder, label in folder_pairs:
        x, y = load_and_preprocess(folder, label)
        X += x
        Y += y
    return WaveformDataset(X, Y)



# ======================
# Trainer class
# ======================
class Trainer:
    def __init__(self, model, device, lr=0.002):
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for waves, labels in loader:
            waves = waves.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(waves).squeeze(-1)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for waves, labels in loader:
                waves = waves.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(waves).squeeze(-1)

                all_probs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        return roc_auc_score(all_labels.numpy(), all_probs.numpy())
    def run(self, dataset, num_epochs=100, early_stop=0.98, save_path="cnn_model.pth"):
        # データセット作成

        # train / test 分割

        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for epoch in range(num_epochs):
            loss = self.train_one_epoch(train_loader)
            auc = self.evaluate(dataset)

            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, AUC: {auc:.4f}")

            if auc > early_stop:
                print("Early stopping")
                break

        torch.save(self.model.state_dict(), save_path)


