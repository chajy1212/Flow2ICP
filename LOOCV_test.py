# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt, random
import os, glob


# ==== Seed ====
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ==== Model ====
class Flow2ICP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)


# ==== Visualization ====
def smooth(x, k=2):
    if k <= 1: return x
    return np.convolve(np.pad(x, (1,1), mode='edge'), np.ones(k)/k, mode='valid')[:len(x)]

def plot_prediction_with_uncertainty(pred_mean, pred_std, sheet_name, save_dir):
    idx = np.arange(len(pred_mean))
    pred_s = smooth(pred_mean)

    plt.figure(figsize=(10, 4))

    # --- Uncertainty band ---
    plt.fill_between(
        idx,
        pred_s - pred_std,
        pred_s + pred_std,
        color='gray', alpha=0.25, label='Uncertainty (±STD)'
    )

    # --- Mean prediction ---
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP (Ensemble Mean)")

    plt.title(f"[Inference] — Sheet: {sheet_name}\n", fontsize=11)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("Predicted ICP (mmHg)", fontsize=11)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{sheet_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    plt.close()

    print(f"Saved plot → {save_path}")


# ==== Load all trained models ====
model_dir = "/home/brainlab/Workspace/jycha/ICP/models"
model_paths = sorted(glob.glob(os.path.join(model_dir, "HM_P_REV_24_*.pth")))

print(f"Loaded {len(model_paths)} trained models")

# Detect input dim (fixed: 29 features)
INPUT_DIM = 29
models = []

for p in model_paths:
    model = Flow2ICP(INPUT_DIM)
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    models.append(model)


# ==== Inference 51~100 ====
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir = "/home/brainlab/Workspace/jycha/ICP/test_results"

for i in range(51, 101):
    sheet_name = f"HM_P_REV_24_{i:03d}"

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
        print(f"[{sheet_name}] sheet missing → skip")
        continue

    print(f"\n==== Inference {sheet_name} ====")

    # Extract numerical features
    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy().astype(np.float32)

    scaler_x = RobustScaler()
    X_scaled = scaler_x.fit_transform(X)

    x_t = torch.tensor(X_scaled, dtype=torch.float32)

    # ========================================================
    # Ensemble predictions: 50 models → (50, N)
    # ========================================================
    preds = []
    for model in models:
        with torch.no_grad():
            pred = model(x_t).numpy().reshape(-1)
        preds.append(pred)

    preds = np.array(preds)        # shape = (num_models, num_samples)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    # ========================================================
    # Save prediction curves
    # ========================================================
    out_csv = os.path.join(save_dir, f"{sheet_name}_pred_mean.csv")
    pd.DataFrame({"pred_mean": pred_mean}).to_csv(out_csv, index=False)
    print(f"Saved CSV → {out_csv}")

    # ========================================================
    # Plot with uncertainty band
    # ========================================================
    plot_prediction_with_uncertainty(pred_mean, pred_std, sheet_name, save_dir)