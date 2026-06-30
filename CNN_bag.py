import os
import numpy as np
import pandas as pd
import torch.optim as optim
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split


# Dataset 
class WaveformDataset(Dataset):
    def __init__(self, data, labels):
        filtered_data = []
        filtered_labels = []
        max_len = 1000
        for w, y in zip(data, labels):
            if len(w) <= max_len:   # 長さフィルタリング
                filtered_data.append(w)
                filtered_labels.append(y)
        self.data = np.array(filtered_data, dtype=object)
        self.labels = np.array(filtered_labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # MAX_LEN = 500
        # if len(self.data[idx]) > MAX_LEN:
        #     return None
        waveform = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label



class WaveformCNN(nn.Module):
    def __init__(self):
        super(WaveformCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16,32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32,32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(32,32, kernel_size=5, padding=2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)  
        x = self.fc2(x) #線形識別機(ここで特徴を抽出する)
        x = torch.sigmoid(x)
        return x


# データ読み込み
def load_and_preprocess_base(folder, label):
    data, labels = [], []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            wave = df['波形'].dropna().values if '波形' in df.columns else df.iloc[:,0].dropna().values
            #wave = wave / 1e15
            wave = wave - wave[0]  #最初の値を0に正規化   
            data.append(wave)
            labels.append(label)
    return data, labels
def extract_all_features_from_loader(loader):
    model.eval()
    all_feats = []
    all_labels = []
    all_mv = []   

    with torch.no_grad():
        for waves, labels, mv_values in loader:
            waves = [torch.tensor(w, dtype=torch.float32) for w in waves]
            waves = torch.stack(waves).to(device)

            labels = torch.tensor(labels, dtype=torch.float32)
            mv_values = torch.tensor(mv_values)

            _, feats = model(waves)

            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_mv.append(mv_values)

    return (
        torch.cat(all_feats).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_mv).numpy()
    )

# フォルダ指定
# folder1 = "/home/htamura/sotsu/sotsu/CA15-3-50mV/autonanopore_events"
# folder2 = "/home/htamura/sotsu/sotsu/CEA-50mV/autonanopore_events"
# folder3 = "/home/htamura/sotsu/sotsu/CA15-3-100mV/autonanopore_events"
# folder4 = "/home/htamura/sotsu/sotsu/CEA-100mV/autonanopore_events"
# folder5 = "/home/htamura/sotsu/sotsu/CA15-3-150mV/autonanopore_events"
# folder6 = "/home/htamura/sotsu/sotsu/CEA-150mV/autonanopore_events"
# folder7 = "/home/htamura/sotsu/sotsu/CA15-3-200mV/autonanopore_events"
# folder8 = "/home/htamura/sotsu/sotsu/CEA-200mV/autonanopore_events"
# folder9 = "/home/htamura/sotsu/sotsu/CA15-3-250mV/autonanopore_events"
# folder10 = "/home/htamura/sotsu/sotsu/CEA-250mV/autonanopore_events"
# folder11 = "/home/htamura/sotsu/sotsu/CA15-3-300mV/autonanopore_events"
# folder12 = "/home/htamura/sotsu/sotsu/CEA-300mV/autonanopore_events"


folder1 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\101\EWMAuto_events"    
folder2 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\102\EWMAuto_events"
folder3 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\103\EWMAuto_events"    
folder4 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\401\EWMAuto_events" 
folder5 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\402\EWMAuto_events"    
folder6 = r"C:\Users\tamur\OneDrive\ドキュメント\sotsu\bag\403\EWMAuto_events" 


X1, y1 = load_and_preprocess_base(folder1, 0)
X2, y2 = load_and_preprocess_base(folder4, 1)
X3, y3 = load_and_preprocess_base(folder2, 0)
X4, y4 = load_and_preprocess_base(folder5, 1)
X5, y5 = load_and_preprocess_base(folder3, 0)
X6, y6 = load_and_preprocess_base(folder6, 1)

#CEA,CA15-3(-50mV ~ -300mV)
XA = X1 + X2
YA = y1 + y2
XB = X3 + X4
YB = y3 + y4
XC = X5 + X6
YC = y5 + y6

datasetA = WaveformDataset(XA, YA)
datasetB = WaveformDataset(XB, YB)
datasetC = WaveformDataset(XC, YC)



#訓練用とテスト用で分ける
train_dataset, test_dataset = random_split(datasetC, [0.8, 0.2])

# DataLoader 
batch_size = 1

loaderA = DataLoader(datasetA, batch_size, shuffle=True)
loaderB = DataLoader(datasetB, batch_size, shuffle=True)
loaderC = DataLoader(datasetC, batch_size, shuffle=True)


#訓練用とテスト用
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

#  デバイス設定 
lr= 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WaveformCNN()  
model = model.to(device)  
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
#電圧によってエポック数変更
num_epochs = 10
#8インスタンスごとにパラメータ更新
accum_steps = 8
epoch_loss_list = []
epoch_train_auc = []

#評価
def evaluate(loader):
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for waves, labels in loader:
                waves = waves.to(device)
                labels = labels.to(device)

                outputs = model(waves).squeeze(-1)

                all_probs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        return roc_auc_score(all_labels.numpy(), all_probs.numpy())

total_samples = 0
# 訓練
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for waves, labels in train_loader:
        for w, l in zip(waves, labels):
            if isinstance(w, np.ndarray):
                w = torch.tensor(w, dtype=torch.float32)
            w = w.unsqueeze(0).to(device)
            l = torch.tensor([l], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(w)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs,l)
            loss = loss / accum_steps
            loss.backward()

            total_loss += loss.item() * accum_steps
            total_samples += 1

            #  accum_steps 回たまったら更新
            if total_samples % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
    if total_samples % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_samples 
    epoch_loss_list.append(avg_loss)

    train_auc = evaluate(train_loader)
    epoch_train_auc.append(train_auc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Auc: {train_auc:.4f}")
    scheduler.step()
    if(train_auc > 0.98):
        break

for name, loader in zip(["A","B","C"], 
                        [loaderA, loaderB, loaderC]):
    print(f"Loader {name}: AUC = {evaluate(loader):.4f}")

auc = evaluate(test_loader)
print("Test Accuracy:", auc)
