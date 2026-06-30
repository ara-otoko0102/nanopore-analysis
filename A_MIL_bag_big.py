import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'DejaVu Sans']


# ═══════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════
MAX_LEN     = 1000
BAG_SIZE    = 50
LR          = 0.001
NUM_EPOCHS  = 10
TEMPERATURE = 1   # < 1: sharper attention, > 1: flatter (uniform → 1.0)
NUM_WORKERS = 8     # 並列CSVロード用スレッド数


# ═══════════════════════════════════════════════════════════════
# Dataset  (MIL_CNN_Attention の MILDataset と同じ)
# 同じラベルの波形を bag_size 個ずつまとめてバッグを構成する
# ═══════════════════════════════════════════════════════════════
class MILDataset(Dataset):
    def __init__(self, data, labels, bag_size):
        self.data    = data
        self.labels  = np.array(labels)
        self.bag_size = bag_size

        self.label_indices      = [np.where(self.labels == i)[0] for i in np.unique(self.labels)]
        self.num_bags_per_label = [len(idx) // bag_size for idx in self.label_indices]
        self.num_bags           = sum(self.num_bags_per_label)
        self.labels_in_dataset  = np.unique(self.labels)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        for label_idx, num_bags in enumerate(self.num_bags_per_label):
            if idx < num_bags:
                class_label   = self.labels_in_dataset[label_idx]
                class_indices = self.label_indices[label_idx]
                start         = idx * self.bag_size
                bag_indices   = class_indices[start:start + self.bag_size]
                bag_waveforms = [torch.tensor(self.data[i], dtype=torch.float32)
                                 for i in bag_indices]
                return bag_waveforms, torch.tensor(class_label, dtype=torch.float32)
            idx -= num_bags


def collate_bag(batch):
    bags   = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


# ═══════════════════════════════════════════════════════════════
# Feature Extractor  (1D-CNN)
# ═══════════════════════════════════════════════════════════════
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.feature_dim = 32

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1)
        return x.view(x.size(0), -1)   # (N, 32)


# ═══════════════════════════════════════════════════════════════
# Gated Attention  (Ilse et al. 2018)
#   a_i = softmax( w^T ( tanh(V h_i) ⊙ sigmoid(U h_i) ) )
# ═══════════════════════════════════════════════════════════════
class GatedAttention(nn.Module):
    def __init__(self, in_features, hidden_features=128, temperature=1.0):
        super().__init__()
        self.att_V = nn.Linear(in_features, hidden_features)
        self.att_U = nn.Linear(in_features, hidden_features)
        self.att_w = nn.Linear(hidden_features, 1)
        self.temperature = temperature

    def forward(self, x):
        A = self.att_w(torch.tanh(self.att_V(x)) * torch.sigmoid(self.att_U(x)))  # (N, 1)
        A = F.softmax(A.T / self.temperature, dim=1)   # (1, N)
        M = torch.mm(A, x)          # (1, in_features)
        return M, A


# ═══════════════════════════════════════════════════════════════
# MIL Model
# ═══════════════════════════════════════════════════════════════
class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.attention         = GatedAttention(self.feature_extractor.feature_dim, temperature=TEMPERATURE)
        self.classifier        = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, bag_of_waveforms):
        H = torch.cat([self.feature_extractor(inst.unsqueeze(0))
                       for inst in bag_of_waveforms], dim=0)   # (bag_size, 32)
        M, A   = self.attention(H)       # M: (1, 32), A: (1, bag_size)
        output = self.classifier(M)      # (1, 1)
        return output.squeeze(), A


# ═══════════════════════════════════════════════════════════════
# Data Loading  (NPYキャッシュ優先 → 並列CSVロード → キャッシュ保存)
# ═══════════════════════════════════════════════════════════════
def _load_one_csv(filepath):
    try:
        df   = pd.read_csv(filepath)
        wave = (df['波形'].dropna().values if '波形' in df.columns
                else df.iloc[:, 0].dropna().values)
        if len(wave) == 0 or len(wave) > MAX_LEN:
            return None
        return (wave - wave[0]).astype(np.float32)
    except Exception:
        return None


def load_and_preprocess_base(folder, label):
    cache_prefix = folder + "_cache"
    data_path    = cache_prefix + "_data.npy"
    labels_path  = cache_prefix + "_labels.npy"

    # NPYキャッシュがあれば即ロード
    if os.path.exists(data_path) and os.path.exists(labels_path):
        print(f"[cache] {os.path.basename(folder)}: {data_path}")
        data   = list(np.load(data_path,   allow_pickle=True))
        labels = list(np.load(labels_path))
        print(f"  -> {len(data):,} 件ロード完了")
        return data, labels

    # キャッシュなし → 並列CSVロード
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    total = len(files)
    print(f"[csv] {os.path.basename(folder)}: {total:,} ファイルを {NUM_WORKERS} スレッドで並列ロード中...")

    data      = []
    processed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for wave in executor.map(_load_one_csv, files):
            processed += 1
            if wave is not None:
                data.append(wave)
            if processed % 100000 == 0:
                print(f"  {processed:,}/{total:,} 処理済み ({len(data):,} 件ロード)")

    labels = [label] * len(data)

    # 次回のためにNPYキャッシュ保存
    np.save(data_path,   np.array(data, dtype=object), allow_pickle=True)
    np.save(labels_path, np.array(labels, dtype=np.int64))
    print(f"  -> {len(data):,} 件 キャッシュ保存: {cache_prefix}_*.npy")

    return data, labels


def shuffled(X, Y):
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X_s, Y_s = zip(*combined)
    return list(X_s), list(Y_s)


# ═══════════════════════════════════════════════════════════════
# Folder Configuration
#   label 0: 101, 102, 103, 301, 302, 303
#   label 1: 401, 402, 403, 601, 602, 603
# ═══════════════════════════════════════════════════════════════
folder1  = r"C:\bag\101\EWMAuto_events"
folder2  = r"C:\bag\102\EWMAuto_events"
folder3  = r"C:\bag\103\EWMAuto_events"
folder4  = r"C:\bag\401\EWMAuto_events"
folder5  = r"C:\bag\402\EWMAuto_events"
folder6  = r"C:\bag\403\EWMAuto_events"
folder7  = r"C:\bag\301\EWMAuto_events"
folder8  = r"C:\bag\302\EWMAuto_events"
folder9  = r"C:\bag\303\EWMAuto_events"
folder10 = r"C:\bag\601\EWMAuto_events"
folder11 = r"C:\bag\602\EWMAuto_events"
folder12 = r"C:\bag\603\EWMAuto_events"

X1,  y1  = load_and_preprocess_base(folder1,  0)
X2,  y2  = load_and_preprocess_base(folder2,  0)
X3,  y3  = load_and_preprocess_base(folder3,  0)
X4,  y4  = load_and_preprocess_base(folder4,  1)
X5,  y5  = load_and_preprocess_base(folder5,  1)
X6,  y6  = load_and_preprocess_base(folder6,  1)
X7,  y7  = load_and_preprocess_base(folder7,  0)
X8,  y8  = load_and_preprocess_base(folder8,  0)
X9,  y9  = load_and_preprocess_base(folder9,  0)
X10, y10 = load_and_preprocess_base(folder10, 1)
X11, y11 = load_and_preprocess_base(folder11, 1)
X12, y12 = load_and_preprocess_base(folder12, 1)

#   条件A: 101(label0) + 401(label1)
#   条件B: 102(label0) + 402(label1)
#   条件C: 103(label0) + 403(label1)
#   条件D: 301(label0) + 601(label1)
#   条件E: 302(label0) + 602(label1)
#   条件F: 303(label0) + 603(label1)
# XA, YA = shuffled(X1  + X6,  y1  + y6)
# XB, YB = shuffled(X2  + X4,  y2  + y4)
# XC, YC = shuffled(X3  + X5,  y3  + y5)
XD, YD = shuffled(X7  + X12,  y7  + y12)
XE, YE = shuffled(X8  + X10, y8  + y10)
XF, YF = shuffled(X9 + X11, y9 + y11)

print("2")
split_idx        = int(len(XD) * 0.8)
X_train, y_train = XD[:split_idx], YD[:split_idx]
X_test,  y_test  = XD[split_idx:], YD[split_idx:]

train_dataset = MILDataset(X_train, y_train, bag_size=BAG_SIZE)
test_dataset  = MILDataset(X_test,  y_test,  bag_size=BAG_SIZE)
# datasetA      = MILDataset(XA,      YA,      bag_size=BAG_SIZE)
# datasetB      = MILDataset(XB,      YB,      bag_size=BAG_SIZE)
# datasetC      = MILDataset(XC,      YC,      bag_size=BAG_SIZE)
datasetD      = MILDataset(XD,      YD,      bag_size=BAG_SIZE)
datasetE      = MILDataset(XE,      YE,      bag_size=BAG_SIZE)
datasetF      = MILDataset(XF,      YF,      bag_size=BAG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=collate_bag)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, collate_fn=collate_bag)
# loaderA      = DataLoader(datasetA,      batch_size=1, shuffle=False, collate_fn=collate_bag)
# loaderB      = DataLoader(datasetB,      batch_size=1, shuffle=False, collate_fn=collate_bag)
# loaderC      = DataLoader(datasetC,      batch_size=1, shuffle=False, collate_fn=collate_bag)
loaderD      = DataLoader(datasetD,      batch_size=1, shuffle=False, collate_fn=collate_bag)
loaderE      = DataLoader(datasetE,      batch_size=1, shuffle=False, collate_fn=collate_bag)
loaderF      = DataLoader(datasetF,      batch_size=1, shuffle=False, collate_fn=collate_bag)


# ═══════════════════════════════════════════════════════════════
# Model / Optimizer
# ═══════════════════════════════════════════════════════════════
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
model     = MILModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════
def evaluate(loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for bags, labels in loader:
            bag   = [inst.to(device) for inst in bags[0]]
            label = labels[0].to(device)
            output, _ = model(bag)
            all_probs.append(output.cpu())
            all_labels.append(label.cpu())
    if not all_labels:
        return 0.0
    return roc_auc_score(torch.stack(all_labels).numpy(),
                         torch.stack(all_probs).numpy())


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for bags, labels in train_loader:
        bag   = [inst.to(device) for inst in bags[0]]
        label = labels[0].to(device)

        optimizer.zero_grad()
        output, _ = model(bag)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss  = total_loss / len(train_loader) if train_loader else 0
    train_auc = evaluate(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}")
    if(train_auc > 0.98):
        break


# ═══════════════════════════════════════════════════════════════
# Test & Cross-condition Evaluation  (CNN_bag の loaderA/B/C 対応)
# ═══════════════════════════════════════════════════════════════
print(f"Test AUC:              {evaluate(test_loader):.4f}")
# print(f"Condition A (101+401): {evaluate(loaderA):.4f}")
# print(f"Condition B (102+402): {evaluate(loaderB):.4f}")
# print(f"Condition C (103+403): {evaluate(loaderC):.4f}")
print(f"Condition D (301+601): {evaluate(loaderD):.4f}")
print(f"Condition E (302+602): {evaluate(loaderE):.4f}")
print(f"Condition F (303+603): {evaluate(loaderF):.4f}")


# ═══════════════════════════════════════════════════════════════
# Attention Visualization
# ═══════════════════════════════════════════════════════════════
CLASS_NAMES = {0: "label-0", 1: "label-1"}
CMAP = cm.get_cmap("RdYlBu_r")


def visualize_attention_bags(loader, title, n_bags=4):
    model.eval()
    per_class = max(1, n_bags // 2)
    collected = {0: [], 1: []}

    with torch.no_grad():
        for bags, labels in loader:
            label = int(labels[0].item())
            if len(collected[label]) >= per_class:
                continue
            bag       = [inst.to(device) for inst in bags[0]]
            output, A = model(bag)
            prob      = output.item()
            weights   = A.squeeze().cpu().numpy()
            waves     = [inst.cpu().numpy() for inst in bag]
            collected[label].append((waves, label, prob, weights))
            if all(len(v) >= per_class for v in collected.values()):
                break

    samples  = collected[0] + collected[1]
    all_vals = np.concatenate([w for s in samples for w in s[0]])
    y_min, y_max = all_vals.min(), all_vals.max()
    y_pad = (y_max - y_min) * 0.05 or 0.01

    for waves, label, prob, weights in samples:
        n_inst = len(waves)
        w_norm = weights / (weights.max() + 1e-8)
        max_idx = int(weights.argmax())

        fig = plt.figure(figsize=(3 * n_inst, 6))
        gs  = fig.add_gridspec(2, n_inst, height_ratios=[1, 2.5],
                               hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"{title}  |  真ラベル: {CLASS_NAMES[label]}"
            f"  |  予測確率: {prob:.3f}", fontsize=11
        )

        ax_bar = fig.add_subplot(gs[0, :])
        ax_bar.bar(range(n_inst), weights,
                   color=[CMAP(w) for w in w_norm], edgecolor="gray")
        ax_bar.set_xticks(range(n_inst))
        ax_bar.set_xlabel("Instance index", fontsize=9)
        ax_bar.set_ylabel("Attention weight", fontsize=9)
        ax_bar.set_title("Attention weights per instance", fontsize=10)
        ax_bar.annotate("★ max", xy=(max_idx, weights[max_idx]),
                        xytext=(max_idx, weights[max_idx] * 1.05),
                        ha="center", fontsize=8, color="red")

        for i, (wave, w, wn) in enumerate(zip(waves, weights, w_norm)):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(wave, color=CMAP(wn), linewidth=0.8 + 2.0 * wn)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_title(f"a = {w:.3f}", fontsize=9,
                         color="red" if i == max_idx else "black",
                         fontweight="bold" if i == max_idx else "normal")
            ax.set_xlabel("Time step", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel("Amplitude", fontsize=8)
        plt.show()


# 条件Cの学習データから attention 可視化
# visualize_attention_bags(loaderC, title="Condition C (103+403)", n_bags=4)
