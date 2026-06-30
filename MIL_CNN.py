import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
import random

# --- 1. MIL用データセットの変更 (パディングなし) ---
class MILDataset(Dataset):
    """
    可変長のインスタンス（波形）を「バッグ」として扱うデータセット。
    パディングは行わない。

    Args:
        data (list): 波形データのリスト。
        labels (list): 各波形に対応するラベルのリスト。
        bag_size (int): 1つのバッグに含めるインスタンスの数。
    """
    def __init__(self, data, labels, bag_size):
        self.data = data
        self.labels = np.array(labels)
        self.bag_size = bag_size
        
        # 同じラベルを持つデータのインデックスをグループ化
        self.label_indices = [np.where(self.labels == i)[0] for i in np.unique(self.labels)]
        self.num_bags_per_label = [len(indices) // bag_size for indices in self.label_indices]
        self.num_bags = sum(self.num_bags_per_label)
        self.labels_in_dataset = np.unique(self.labels)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        # どのラベルのバッグを生成するか決定
        for label_idx, num_bags in enumerate(self.num_bags_per_label):
            if idx < num_bags:
                class_label = self.labels_in_dataset[label_idx]
                class_indices = self.label_indices[label_idx]
                
                start = idx * self.bag_size
                end = start + self.bag_size
                bag_indices = class_indices[start:end]
                
                # バッグ内の波形をTensorのリストとして返す
                bag_waveforms = [torch.tensor(self.data[i], dtype=torch.float32) for i in bag_indices]
                
                return bag_waveforms, torch.tensor(class_label, dtype=torch.float32)
            idx -= num_bags

# --- 2. モデルの変更 (forwardメソッドの入力形式を変更) ---

class FeatureExtractor(nn.Module):
    """
    個々の波形から特徴量を抽出する。
    AdaptiveAvgPool1dにより、入力波形の長さに依らず固定長の出力を生成する。
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.feature_dim = 32

    def forward(self, x):
        # 入力xの形状: (N, L_in) or (N, 1, L_in)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (N, 1, L_in)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool1d(x, 1) # Global Average Pooling
        x = x.view(x.size(0), -1) # (N, feature_dim)
        return x

class Attention(nn.Module):
    def __init__(self, in_features, hidden_features=128):
        super(Attention, self).__init__()
        self.L = in_features
        self.D = hidden_features
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        A = self.attention(x)  # (bag_size, 1)
        A = torch.transpose(A, 1, 0)  # (1, bag_size)
        A = F.softmax(A, dim=1)  # (1, bag_size)
        
        M = torch.mm(A, x)  # (1, in_features)
        return M, A

class MILModel(nn.Module):
    """
    可変長の波形リストを処理するMILモデル。
    """
    def __init__(self):
        super(MILModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.attention = Attention(self.feature_extractor.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, bag_of_waveforms):
        # bag_of_waveforms: 波形テンソルのリスト
        
        # 各インスタンス（波形）の特徴を個別に抽出
        # バッチサイズ1で各波形を処理するためunsqueeze(0)を追加
        feature_list = [self.feature_extractor(instance.unsqueeze(0)) for instance in bag_of_waveforms]
        
        # 特徴ベクトルのリストを1つのテンソルに結合
        H = torch.cat(feature_list, dim=0)  # (bag_size, feature_dim)
        
        # アテンション・プーリングでバッグの特徴を集約
        M, A = self.attention(H) # M: (1, feature_dim)
        
        # 最終的な識別
        output = self.classifier(M)
        return output.squeeze()

# --- データ読み込み関数 (変更なし) ---
def load_and_preprocess_base(folder, label):
    data, labels = [], []
    max_len = 1000 # 長すぎる波形は除外

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            wave = df['波形'].dropna().values if '波形' in df.columns else df.iloc[:,0].dropna().values
            
            if len(wave) > max_len:
                continue

            #wave = wave / 1e15
            wave = wave - wave[0]
            data.append(wave)
            labels.append(label)
    return data, labels


folder1 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-50mV\superlow\EWMAuto_events"    
folder2 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-50mV\superlow\EWMAuto_events"    
folder3 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-100mV\superlow\EWMAuto_events"    
folder4 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-100mV\superlow\EWMAuto_events"    
folder5 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-150mV\superlow\EWMAuto_events"    
folder6 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-150mV\superlow\EWMAuto_events"    
folder7 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-200mV\superlow\EWMAuto_events"    
folder8 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-200mV\superlow\EWMAuto_events"    
folder9 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-250mV\superlow\EWMAuto_events"    
folder10 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-250mV\superlow\EWMAuto_events"    
folder11 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CA15-3\-300mV\superlow\EWMAuto_events"    
folder12 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\data_MIL\CEA\-300mV\superlow\EWMAuto_events"    


X1, y1 = load_and_preprocess_base(folder1, 0)
X2, y2 = load_and_preprocess_base(folder2, 1)
X3, y3 = load_and_preprocess_base(folder3, 0)
X4, y4 = load_and_preprocess_base(folder4, 1)
X5, y5 = load_and_preprocess_base(folder5, 0)
X6, y6 = load_and_preprocess_base(folder6, 1)
X7, y7 = load_and_preprocess_base(folder7, 0)
X8, y8 = load_and_preprocess_base(folder8, 1)
X9, y9 = load_and_preprocess_base(folder9, 0)
X10, y10 = load_and_preprocess_base(folder10, 1)
X11, y11 = load_and_preprocess_base(folder11, 0)
X12, y12 = load_and_preprocess_base(folder12, 1)

#CEA,CA15-3(-50mV ~ -300mV)
X50 = X1 + X2
Y50 = y1 + y2
X100 = X3 + X4
Y100 = y3 + y4
X150 = X5 + X6
Y150 = y5 + y6
X200 = X7 + X8
Y200 = y7 + y8
X250 = X9 + X10
Y250 = y9 + y10
X300 = X11 + X12
Y300 = y11 + y12

X_all = X1 + X2
y_all = y1 + y2



combined = list(zip(X_all, y_all))
random.shuffle(combined)
X_all, y_all = zip(*combined)

combined = list(zip(X100, Y100))
random.shuffle(combined)
X100, Y100 = zip(*combined)

combined = list(zip(X50, Y50))
random.shuffle(combined)
X50, Y50 = zip(*combined)

combined = list(zip(X150, Y150))
random.shuffle(combined)
X150, Y150 = zip(*combined)

combined = list(zip(X200, Y200))
random.shuffle(combined)
X200, Y200 = zip(*combined)

combined = list(zip(X250, Y250))
random.shuffle(combined)
X250, Y250 = zip(*combined)

combined = list(zip(X300, Y300))
random.shuffle(combined)
X300, Y300 = zip(*combined)



split_idx = int(len(X300) * 0.8)
X_train, y_train = X300[:split_idx], Y300[:split_idx]
X_test, y_test = X300[split_idx:], Y300[split_idx:]

bag_size = 5
train_dataset = MILDataset(X_train, y_train, bag_size=bag_size)
test_dataset = MILDataset(X_test, y_test, bag_size=bag_size)


dataset50 = MILDataset(X50, Y50, bag_size=bag_size)
dataset100 = MILDataset(X100, Y100, bag_size=bag_size)
dataset150 = MILDataset(X150, Y150, bag_size=bag_size)
dataset200 = MILDataset(X200, Y200, bag_size=bag_size)
dataset250 = MILDataset(X250, Y250, bag_size=bag_size)
dataset300 = MILDataset(X300, Y300, bag_size=bag_size)
# DataLoader (バッチサイズ1で、各バッチが1つのバッグに対応)
# batch_size=1の場合、カスタムのcollate_fnは不要
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loader50 = DataLoader(dataset50, batch_size=1, shuffle=True)
loader100 = DataLoader(dataset100, batch_size=1, shuffle=False)
loader150 = DataLoader(dataset150, batch_size=1, shuffle=True)
loader200 = DataLoader(dataset200, batch_size=1, shuffle=False)
loader250 = DataLoader(dataset250, batch_size=1, shuffle=True)
loader300 = DataLoader(dataset300, batch_size=1, shuffle=False)
# --- デバイス設定 ---
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MILModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = 30

# --- 3. 学習・評価ループの変更 ---
def collate_bag(batch):
    """
    DataLoaderから渡されるバッチを処理する。
    batchは [(bag_waveforms, label)] という形式のリスト。
    """
    bags = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return bags, torch.stack(labels)


def evaluate(loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for bags, labels in loader:
            # bagsはリストのリストになっているので最初の要素を取り出す
            bag = bags[0]
            label = labels[0]
            
            # バッグ内の各波形をデバイスに送る
            bag = [instance.to(device) for instance in bag]
            label = label.to(device)
            
            output = model(bag)
            
            all_probs.append(output.cpu())
            all_labels.append(label.cpu())
            
    if not all_labels:
        return 0.0

    # スカラーテンソルを結合するためにunsqueezeを使用
    all_probs = torch.stack(all_probs)
    all_labels = torch.stack(all_labels)

    return roc_auc_score(all_labels.numpy(), all_probs.numpy())


# --- 学習ループ ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    # DataLoaderは ( [list_of_tensors], tensor_label ) というタプルをリストでラップして返す
    for bags, labels in train_loader:
        bag = bags[0] # バッチサイズ1なので、リストの最初の要素がバッグ
        label = labels[0] # 同様にラベルも最初の要素

        # バッグ内の各波形をデバイスに送る
        bag = [instance.to(device) for instance in bag]
        label = label.to(device)

        optimizer.zero_grad()
        output = model(bag)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    train_auc = evaluate(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train AUC: {train_auc:.4f}")

# --- テスト ---
test_auc = evaluate(test_loader)
print(f"Test AUC: {test_auc:.4f}")

print(f"50 AUC:{evaluate(loader50):.4f}")
print(f"100 AUC:{evaluate(loader100):.4f}")
print(f"150 AUC:{evaluate(loader150):.4f}")
print(f"200 AUC:{evaluate(loader200):.4f}")
print(f"250 AUC:{evaluate(loader250):.4f}")
print(f"300 AUC:{evaluate(loader300):.4f}")
