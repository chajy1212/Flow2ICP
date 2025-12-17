# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt, random, os
from PIL import Image
import glob
from collections import OrderedDict


# ============================================================
# Seed
# ============================================================
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ============================================================
# Model
# ============================================================
class Flow2ICP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# Loss
# ============================================================
def corr_loss(pred, target):
    vx, vy = pred - pred.mean(), target - target.mean()
    denom = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum())
    return 1 - (vx * vy).sum() / (denom + 1e-8)


# ============================================================
# Utils
# ============================================================
def smooth(x, k=2):
    if k <= 1:
        return x
    return np.convolve(
        np.pad(x, (1,1), mode='edge'),
        np.ones(k)/k,
        mode='valid'
    )[:len(x)]


def plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir):
    pred_s = smooth(pred)
    idx = np.arange(len(true))

    plt.figure(figsize=(10,4))
    plt.fill_between(idx, true, pred_s, color='gray', alpha=0.25)
    plt.plot(idx, true, 'k-', lw=2, label="True ICP")
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP")

    plt.title(
        f"[Train] {sheet_name}\n"
        f"Corr={corr:.2f} | RMSE={rmse:.2f} | Acc={acc:.1f}%",
        fontsize=11
    )
    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{sheet_name}.png"), dpi=200)
    plt.close()


# ============================================================
# Paths
# ============================================================
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir  = "/home/brainlab/Workspace/jycha/ICP/results"
model_dir = "/home/brainlab/Workspace/jycha/ICP/models"
os.makedirs(model_dir, exist_ok=True)

results = []


# ============================================================
# 1) Subject-wise LOOCV (1~50)
# ============================================================
for i in range(1, 51):
    sheet_name = f"HM_P_REV_24_{i:03d}"

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except Exception as e:
        print(f"[{sheet_name}] skipped ({e})")
        continue

    if "PIC at rest (mmHg)" not in df.columns:
        print(f"[{sheet_name}] no target column")
        continue

    y = pd.to_numeric(
        df["PIC at rest (mmHg)"], errors="coerce"
    ).ffill().astype(np.float32).values.reshape(-1,1)

    X = df.select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values

    if len(X) < 5:
        print(f"[{sheet_name}] insufficient samples")
        continue

    print(f"\n==== {sheet_name} | Data: {X.shape} ====")

    scaler_y = RobustScaler()
    y_s = scaler_y.fit_transform(y)

    all_true, all_pred = [], []

    # ---------- LOOCV ----------
    for j in range(len(X)):
        train_idx = [k for k in range(len(X)) if k != j]

        scaler_x = RobustScaler()
        X_train = scaler_x.fit_transform(X[train_idx])
        X_test  = scaler_x.transform(X[[j]])

        y_train = y_s[train_idx]
        y_test  = y_s[[j]]

        model = Flow2ICP(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        x_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for _ in range(3000):
            opt.zero_grad()
            pred = model(x_t)
            loss = 0.8 * F.mse_loss(pred, y_t) + 0.2 * corr_loss(pred, y_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

        all_pred.append(float(scaler_y.inverse_transform(y_pred)[0,0]))
        all_true.append(float(scaler_y.inverse_transform(y_test)[0,0]))

    # ---------- Metrics ----------
    true = np.array(all_true)
    pred = np.array(all_pred)

    # clipping + smoothing
    for k in range(1, len(pred)):
        if pred[k-1] - pred[k] > 1.0:
            pred[k] = pred[k-1] - 1.0
    pred = smooth(pred, k=2)

    corr, _ = pearsonr(true, pred)
    rmse = np.sqrt(((true - pred)**2).mean())
    acc = np.clip(100 * (1 - np.mean(np.abs(true - pred) / np.abs(true))), 0, 100)

    print(f"[{sheet_name}] Corr={corr:.3f} | RMSE={rmse:.3f} | Acc={acc:.2f}%")

    results.append([sheet_name, corr, rmse, acc])
    plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir)

    # save model
    torch.save(model.state_dict(), os.path.join(model_dir, f"{sheet_name}.pth"))


# ============================================================
# 2) Summary CSV
# ============================================================
pd.DataFrame(results, columns=["Sheet","Corr","RMSE","Acc"]) \
  .to_csv(os.path.join(save_dir, "summary.csv"), index=False)


# ============================================================
# 3) Model Weight Averaging (50 models → 1 model)
# ============================================================
print("\n==== Averaging 50 subject models ====")

model_paths = [
    os.path.join(model_dir, f"HM_P_REV_24_{i:03d}.pth")
    for i in range(1, 51)
    if os.path.exists(os.path.join(model_dir, f"HM_P_REV_24_{i:03d}.pth"))
]

avg_state = None

for idx, path in enumerate(model_paths):
    state = torch.load(path, map_location="cpu")

    if avg_state is None:
        avg_state = OrderedDict({k: v.clone() for k, v in state.items()})
    else:
        for k in avg_state:
            avg_state[k] += state[k]

for k in avg_state:
    avg_state[k] /= len(model_paths)

final_model_path = os.path.join(model_dir, "final_model.pth")
torch.save(avg_state, final_model_path)

print(f">>> Averaged model saved → {final_model_path}")
