import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
# Transformer Model
# ======================
class WaveformTransformer(nn.Module):
    def __init__(self, patch_size=4, dim=64, depth=4, heads=4, mlp_dim=128, emb_dropout = 0.1, max_patches=512):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=(1, patch_size))
        self.patch_embed = nn.Linear(patch_size, dim)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        x = x.unsqueeze(2)  # (B,1,1,L)

        patches = self.unfold(x)           # (B, patch, N)
        patches = patches.transpose(1, 2) # (B, N, patch)

        x = self.patch_embed(patches)


        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]

        # Dropout
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.fc(x)
        return torch.sigmoid(x)


# ======================
# Data loader
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
# Trainer
# ======================
class Trainer:
    def __init__(self, model, device, lr=0.0001):
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

    def run(self, dataset, num_epochs=30, save_path="transformer_model.pth"):
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        for epoch in range(num_epochs):
            loss = self.train_one_epoch(loader)
            auc = self.evaluate(dataset)

            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, AUC: {auc:.4f}")

        torch.save(self.model.state_dict(), save_path)