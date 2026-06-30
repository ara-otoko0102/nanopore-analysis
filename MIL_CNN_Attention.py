import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'DejaVu Sans']


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════
class MILDataset(Dataset):
    """
    可変長のインスタンス（波形）を「バッグ」として扱うデータセット。
    同じラベルの波形を bag_size 個ずつまとめてバッグを構成する。
    """
    def __init__(self, data, labels, bag_size):
        self.data = data
        self.labels = np.array(labels)
        self.bag_size = bag_size

        self.label_indices = [np.where(self.labels == i)[0] for i in np.unique(self.labels)]
        self.num_bags_per_label = [len(indices) // bag_size for indices in self.label_indices]
        self.num_bags = sum(self.num_bags_per_label)
        self.labels_in_dataset = np.unique(self.labels)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        for label_idx, num_bags in enumerate(self.num_bags_per_label):
            if idx < num_bags:
                class_label  = self.labels_in_dataset[label_idx]
                class_indices = self.label_indices[label_idx]
                start = idx * self.bag_size
                end   = start + self.bag_size
                bag_indices = class_indices[start:end]
                bag_waveforms = [torch.tensor(self.data[i], dtype=torch.float32)
                                 for i in bag_indices]
                return bag_waveforms, torch.tensor(class_label, dtype=torch.float32)
            idx -= num_bags


# ═══════════════════════════════════════════════════════════════
# Feature Extractor  (1D-CNN, 変更なし)
# ═══════════════════════════════════════════════════════════════
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
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
# Gated Attention  (Ilse et al. 2018, Pattern C)
#
# 旧 Attention:  a_i = softmax( w^T tanh(V h_i) )           ← tanh 1経路
# 新 GatedAttention: a_i = softmax( w^T (tanh(V h_i) ⊙ sigmoid(U h_i)) )
#                                                            ← tanh × sigmoid 2経路
# sigmoid ゲートが "どの次元を見るか" を動的に制御することで
# 複雑なパターンへの対応力が向上する。
# ═══════════════════════════════════════════════════════════════
class GatedAttention(nn.Module):
    def __init__(self, in_features, hidden_features=128):
        super(GatedAttention, self).__init__()
        self.att_V = nn.Linear(in_features, hidden_features)   # tanh 経路
        self.att_U = nn.Linear(in_features, hidden_features)   # sigmoid 経路 (ゲート)
        self.att_w = nn.Linear(hidden_features, 1)

    def forward(self, x):
        # x: (N, in_features)
        A = self.att_w(torch.tanh(self.att_V(x)) * torch.sigmoid(self.att_U(x)))  # (N, 1)
        A = A.T                   # (1, N)
        A = F.softmax(A, dim=1)  # (1, N)
        M = torch.mm(A, x)       # (1, in_features)
        return M, A


# ═══════════════════════════════════════════════════════════════
# MIL Model
# ═══════════════════════════════════════════════════════════════
class MILModel(nn.Module):
    def __init__(self):
        super(MILModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.attention = GatedAttention(self.feature_extractor.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, bag_of_waveforms):
        feature_list = [self.feature_extractor(inst.unsqueeze(0))
                        for inst in bag_of_waveforms]
        H = torch.cat(feature_list, dim=0)   # (bag_size, 32)
        M, A = self.attention(H)             # M: (1, 32), A: (1, bag_size)
        output = self.classifier(M)          # (1, 1)
        return output.squeeze(), A


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════
def load_and_preprocess_base(folder, label):
    data, labels = [], []
    max_len = 1000
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(folder, file))
        wave = (df['波形'].dropna().values if '波形' in df.columns
                else df.iloc[:, 0].dropna().values)
        if len(wave) > max_len:
            continue
        wave = wave - wave[0]
        data.append(wave)
        labels.append(label)
    return data, labels


folder1  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-50mV\supersuperlow\EWMAuto_events"
folder2  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-50mV\supersuperlow\EWMAuto_events"
folder3  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-100mV\supersuperlow\EWMAuto_events"
folder4  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-100mV\supersuperlow\EWMAuto_events"
folder5  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-150mV\supersuperlow\EWMAuto_events"
folder6  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-150mV\supersuperlow\EWMAuto_events"
folder7  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-200mV\supersuperlow\EWMAuto_events"
folder8  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-200mV\supersuperlow\EWMAuto_events"
folder9  = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-250mV\supersuperlow\EWMAuto_events"
folder10 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-250mV\supersuperlow\EWMAuto_events"
folder11 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-300mV\supersuperlow\EWMAuto_events"
folder12 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-300mV\supersuperlow\EWMAuto_events"

X1,  y1  = load_and_preprocess_base(folder1,  0)
X2,  y2  = load_and_preprocess_base(folder2,  1)
X3,  y3  = load_and_preprocess_base(folder3,  0)
X4,  y4  = load_and_preprocess_base(folder4,  1)
X5,  y5  = load_and_preprocess_base(folder5,  0)
X6,  y6  = load_and_preprocess_base(folder6,  1)
X7,  y7  = load_and_preprocess_base(folder7,  0)
X8,  y8  = load_and_preprocess_base(folder8,  1)
X9,  y9  = load_and_preprocess_base(folder9,  0)
X10, y10 = load_and_preprocess_base(folder10, 1)
X11, y11 = load_and_preprocess_base(folder11, 0)
X12, y12 = load_and_preprocess_base(folder12, 1)

def shuffled(X, Y):
    combined = list(zip(X, Y))
    random.shuffle(combined)
    return zip(*combined)

X50,  Y50  = shuffled(X1  + X2,  y1  + y2)
X100, Y100 = shuffled(X3  + X4,  y3  + y4)
X150, Y150 = shuffled(X5  + X6,  y5  + y6)
X200, Y200 = shuffled(X7  + X8,  y7  + y8)
X250, Y250 = shuffled(X9  + X10, y9  + y10)
X300, Y300 = shuffled(X11 + X12, y11 + y12)

T_X, T_Y = list(X300), list(Y300)
split_idx  = int(len(T_X) * 0.8)
X_train, y_train = T_X[:split_idx], T_Y[:split_idx]
X_test,  y_test  = T_X[split_idx:], T_Y[split_idx:]

def collate_bag(batch):
    """batch = [(list_of_tensors, label), ...] をそのまま返す。
    デフォルト collate はリストを転置してしまうため、明示的に指定する。"""
    bags   = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels

bag_size = 5
train_dataset = MILDataset(X_train, y_train, bag_size=bag_size)
test_dataset  = MILDataset(X_test,  y_test,  bag_size
                           =bag_size)

dataset50  = MILDataset(list(X50),  list(Y50),  bag_size=bag_size)
dataset100 = MILDataset(list(X100), list(Y100), bag_size=bag_size)
dataset150 = MILDataset(list(X150), list(Y150), bag_size=bag_size)
dataset200 = MILDataset(list(X200), list(Y200), bag_size=bag_size)
dataset250 = MILDataset(list(X250), list(Y250), bag_size=bag_size)
dataset300 = MILDataset(list(X300), list(Y300),        bag_size=bag_size)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=collate_bag)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, collate_fn=collate_bag)
loader50  = DataLoader(dataset50,  batch_size=1, shuffle=True,  collate_fn=collate_bag)
loader100 = DataLoader(dataset100, batch_size=1, shuffle=False, collate_fn=collate_bag)
loader150 = DataLoader(dataset150, batch_size=1, shuffle=True,  collate_fn=collate_bag)
loader200 = DataLoader(dataset200, batch_size=1, shuffle=False, collate_fn=collate_bag)
loader250 = DataLoader(dataset250, batch_size=1, shuffle=True,  collate_fn=collate_bag)
loader300 = DataLoader(dataset300, batch_size=1, shuffle=False, collate_fn=collate_bag)


# ═══════════════════════════════════════════════════════════════
# Model / Optimizer
# ═══════════════════════════════════════════════════════════════
lr     = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = MILModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = 10


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
            output, _ = model(bag)   # attention重みは評価時は不使用
            all_probs.append(output.cpu())
            all_labels.append(label.cpu())
    if not all_labels:
        return 0.0
    return roc_auc_score(torch.stack(all_labels).numpy(),
                         torch.stack(all_probs).numpy())


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════
for epoch in range(num_epochs):
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test & Cross-voltage Evaluation
# ═══════════════════════════════════════════════════════════════
print(f"Test AUC:  {evaluate(test_loader):.4f}")
print(f" -50mV AUC: {evaluate(loader50):.4f}")
print(f"-100mV AUC: {evaluate(loader100):.4f}")
print(f"-150mV AUC: {evaluate(loader150):.4f}")
print(f"-200mV AUC: {evaluate(loader200):.4f}")
print(f"-250mV AUC: {evaluate(loader250):.4f}")
print(f"-300mV AUC: {evaluate(loader300):.4f}")


# ═══════════════════════════════════════════════════════════════
# Attention Visualization
#
# 各バッグについて2種類の図を描画する：
#   上段: バッグ内インスタンスへのattention重み（棒グラフ）
#   下段: 各インスタンスの波形（attention重みが高いほど赤・太く表示）
# ═══════════════════════════════════════════════════════════════
CLASS_NAMES = {0: "CA15-3", 1: "CEA"}
CMAP = cm.get_cmap("RdYlBu_r")   # 低attention=青、高attention=赤


def visualize_attention_bags(loader, title, n_bags=4):
    """
    loader から n_bags 個のバッグを取り出し、
    attention重みと対応する波形を可視化する。
    """
    model.eval()
    per_class = max(1, n_bags // 2)
    collected = {0: [], 1: []}

    # --- 第1パス: クラスごとに per_class 個ずつ収集 ---
    with torch.no_grad():
        for bags, labels in loader:
            label = int(labels[0].item())
            if len(collected[label]) >= per_class:
                continue
            bag     = [inst.to(device) for inst in bags[0]]
            output, A = model(bag)
            prob    = output.item()
            weights = A.squeeze().cpu().numpy()
            waves   = [inst.cpu().numpy() for inst in bag]
            collected[label].append((waves, label, prob, weights))
            if all(len(v) >= per_class for v in collected.values()):
                break

    samples = collected[0] + collected[1]

    # --- 全バッグ共通のy軸範囲を計算 ---
    all_vals = np.concatenate([w for s in samples for w in s[0]])
    y_min = all_vals.min()
    y_max = all_vals.max()
    y_pad = (y_max - y_min) * 0.05 or 0.01

    # --- 第2パス: 描画 ---
    for waves, label, prob, weights in samples:
        n_inst = len(waves)
        w_norm = weights / (weights.max() + 1e-8)

        fig = plt.figure(figsize=(3 * n_inst, 6))
        gs  = fig.add_gridspec(2, n_inst,
                               height_ratios=[1, 2.5],
                               hspace=0.45, wspace=0.35)

        fig.suptitle(
            f"{title}  |  真ラベル: {CLASS_NAMES[label]}"
            f"  |  予測確率: {prob:.3f}",
            fontsize=11
        )

        ax_bar = fig.add_subplot(gs[0, :])
        bar_colors = [CMAP(w) for w in w_norm]
        ax_bar.bar(range(n_inst), weights, color=bar_colors, edgecolor="gray")
        ax_bar.set_xticks(range(n_inst))
        ax_bar.set_xlabel("Instance index", fontsize=9)
        ax_bar.set_ylabel("Attention weight", fontsize=9)
        ax_bar.set_title("Attention weights per instance", fontsize=10)
        max_idx = int(weights.argmax())
        ax_bar.annotate("★ max", xy=(max_idx, weights[max_idx]),
                        xytext=(max_idx, weights[max_idx] * 1.05),
                        ha="center", fontsize=8, color="red")

        for i, (wave, w, wn) in enumerate(zip(waves, weights, w_norm)):
            ax = fig.add_subplot(gs[1, i])
            lw = 0.8 + 2.0 * wn
            ax.plot(wave, color=CMAP(wn), linewidth=lw)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_title(f"a = {w:.3f}", fontsize=9,
                         color="red" if i == max_idx else "black",
                         fontweight="bold" if i == max_idx else "normal")
            ax.set_xlabel("Time step", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel("Amplitude", fontsize=8)

        plt.show()


def plot_attention_distribution(loaders_dict):
    """
    各電圧条件のローダーについて、クラスごとのattention重み分布を
    バイオリンプロットで比較する。
    attention重みは全バッグ × 全インスタンスを集約して描画する。
    """
    model.eval()
    fig, axes = plt.subplots(1, len(loaders_dict), figsize=(4 * len(loaders_dict), 4),
                             sharey=True)
    fig.suptitle("Attention weight distribution by class & voltage", fontsize=12)

    for ax, (volt_name, loader) in zip(axes, loaders_dict.items()):
        weights_by_class = {0: [], 1: []}

        with torch.no_grad():
            for bags, labels in loader:
                bag   = [inst.to(device) for inst in bags[0]]
                label = int(labels[0].item())
                _, A  = model(bag)
                weights_by_class[label].extend(
                    A.squeeze().cpu().numpy().tolist()
                )

        data   = [weights_by_class[0], weights_by_class[1]]
        parts  = ax.violinplot(data, positions=[0, 1], showmedians=True)
        colors = ["tab:orange", "tab:blue"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CA15-3", "CEA"])
        ax.set_title(volt_name, fontsize=10)
        ax.set_ylabel("Attention weight", fontsize=9)
        ax.axhline(1 / bag_size, color="gray", linestyle="--", linewidth=0.8,
                   label=f"uniform (1/{bag_size})")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# --- 実行 ---
# 各電圧条件から4バッグずつ attention 可視化
for volt, ldr in [("-300mV (train)", loader300)]:
    visualize_attention_bags(ldr, title=volt, n_bags=5)

# 全電圧のattention重み分布を比較
# plot_attention_distribution({
#     "-50mV":  loader50,
#     "-100mV": loader100,
#     "-150mV": loader150,
#     "-200mV": loader200,
#     "-250mV": loader250,
#     "-300mV": loader300,
# })
