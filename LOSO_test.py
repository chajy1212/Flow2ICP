# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt, random
import os, glob
from PIL import Image


# ==========================================================
# Seed
# ==========================================================
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ==========================================================
# Model
# ==========================================================
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


# ==========================================================
# Utils
# ==========================================================
def smooth(x, k=2):
    if k <= 1:
        return x
    return np.convolve(
        np.pad(x, (1,1), mode='edge'),
        np.ones(k)/k,
        mode='valid'
    )[:len(x)]


def edge_smooth(x, k=2, edge=3):
    x = x.copy()
    x[:edge]  = smooth(x[:edge], k)
    x[-edge:] = smooth(x[-edge:], k)
    return x


def load_subject_X(excel_path, sheet_name):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).iloc[:32]
    except:
        return None

    X = df.select_dtypes(include=[np.number]) \
          .fillna(0).astype(np.float32).values

    if len(X) < 5:
        return None

    return X


def plot_pred(pred, sheet_name, save_dir):
    pred_s = edge_smooth(pred, k=2, edge=3)
    idx = np.arange(len(pred_s))

    plt.figure(figsize=(10,4))
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP")

    plt.title(f"[Inference] {sheet_name}", fontsize=11)
    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{sheet_name}.png"), dpi=200)
    plt.show()
    plt.close()


# ==========================================================
# Paths
# ==========================================================
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
model_dir  = "/home/brainlab/Workspace/jycha/ICP/models_LOSO"
save_dir   = "/home/brainlab/Workspace/jycha/ICP/inference_LOSO_51_100"


# ==========================================================
# Load LOSO models
# ==========================================================
print("\n==============================")
print(" Loading LOSO models (1–50)")
print("==============================\n")

checkpoints = []

for i in range(1, 51):
    path = os.path.join(model_dir, f"HM_P_REV_24_{i:03d}.pth")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints.append(ckpt)

assert len(checkpoints) > 0, "No LOSO models found."

print(f"Loaded {len(checkpoints)} LOSO models.")


# ==========================================================
# Inference (51–100) — ENSEMBLE
# ==========================================================
summary_rows = []

print("\n==============================")
print(" Final Inference")
print(" Subjects: 51–100")
print("==============================\n")

for i in range(51, 101):
    sheet = f"HM_P_REV_24_{i:03d}"
    print(f"[Inference] {sheet}")

    X = load_subject_X(excel_path, sheet)
    if X is None:
        print(" -> skipped (no data)")
        continue

    preds = []  # ### [MODIFIED] 모델별 예측 저장

    for ckpt in checkpoints:
        model = Flow2ICP(d_in=X.shape[1])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        scaler_x = ckpt["scaler_x"]
        scaler_y = ckpt["scaler_y"]

        X_s = scaler_x.transform(X)

        with torch.no_grad():
            y_hat = model(torch.tensor(X_s, dtype=torch.float32)).numpy()

        y_hat = scaler_y.inverse_transform(y_hat).ravel()
        preds.append(y_hat)

    preds = np.stack(preds, axis=0)     # (50, T)
    pred_mean = preds.mean(axis=0)      # ensemble mean
    pred_mean = np.round(pred_mean, 2)

    # --- Clipping ---
    if len(pred_mean) > 6:
        ref = pred_mean[:-3]
        lower = np.percentile(ref, 5)
        upper = np.percentile(ref, 95)
        pred_mean = np.clip(pred_mean, lower, upper)

    # --- Save per-subject CSV ---
    np.savetxt(
        os.path.join(save_dir, f"{sheet}.csv"),
        pred_mean,
        delimiter=","
    )

    # --- Plot ---
    plot_pred(pred_mean, sheet, save_dir)

    summary_rows.append([sheet] + pred_mean.tolist())


# ==========================================================
# Summary CSV
# ==========================================================
max_len = max(len(r) - 1 for r in summary_rows)

rows = []
for r in summary_rows:
    name, vals = r[0], r[1:]
    vals = vals + [np.nan] * (max_len - len(vals))
    rows.append([name] + vals)

columns = ["sheet"] + [str(i) for i in range(1, max_len + 1)]

df = pd.DataFrame(rows, columns=columns)

df.to_csv(
    os.path.join(save_dir, "summary_51_100.csv"),
    index=False
)

print("\n>>> LOSO inference finished.")


# ============================================================
# Merge prediction plots (5×10 grid)
# ============================================================
def merge_results(save_dir, grid_cols=5, grid_rows=10,
                  out_name="summary_grid_51_100.png"):

    paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(paths) == 0:
        return

    imgs = [Image.open(p) for p in paths[:grid_cols * grid_rows]]
    w, h = imgs[0].size

    grid = Image.new("RGB", (grid_cols*w, grid_rows*h), "white")

    for i, img in enumerate(imgs):
        x = (i % grid_cols) * w
        y = (i // grid_cols) * h
        grid.paste(img, (x, y))

    grid.save(os.path.join(save_dir, out_name), dpi=(200,200))
    print(f">>> Grid saved → {out_name}")

merge_results(save_dir)