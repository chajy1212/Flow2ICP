# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
from sklearn.preprocessing import RobustScaler


# ============================================================
# Seed
# ============================================================
def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()


# ============================================================
# Model (same architecture)
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
# Utils
# ============================================================
def smooth(x, k=2):
    if k <= 1:
        return x
    return np.convolve(
        np.pad(x, (1, 1), mode="edge"),
        np.ones(k) / k,
        mode="valid"
    )[:len(x)]


def plot_pred_curve(pred, sheet, save_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(pred, "r-", lw=2)
    plt.title(f"[Predicted ICP] {sheet}")
    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{sheet}.png"), dpi=200)
    plt.show()
    plt.close()


# ============================================================
# Paths
# ============================================================
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
model_path = "/home/brainlab/Workspace/jycha/ICP/models/final_model.pth"

infer_dir = "/home/brainlab/Workspace/jycha/ICP/inference_51_100"
os.makedirs(infer_dir, exist_ok=True)


# ============================================================
# Load averaged model
# ============================================================
state = torch.load(model_path, map_location="cpu")

# infer input dimension automatically
first_key = list(state.keys())[0]
d_in = state[first_key].shape[1]

model = Flow2ICP(d_in)
model.load_state_dict(state)
model.eval()

print(f">>> Averaged model loaded (input dim = {d_in})")


# ============================================================
# 51~100 ICP Prediction (PIC at rest is missing)
# ============================================================
summary_rows = []

for i in range(51, 101):
    sheet = f"HM_P_REV_24_{i:03d}"
    print(f"[Predict] {sheet}")

    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            header=6
        ).iloc[:32]

    except Exception as e:
        print(f"  - skipped ({e})")
        continue

    # target column is missing → only features used
    X = df.select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values

    if len(X) < 5:
        print("  - insufficient samples")
        continue

    # subject-wise scaling (same policy as training)
    scaler_x = RobustScaler()
    X_s = scaler_x.fit_transform(X)

    with torch.no_grad():
        pred = model(torch.tensor(X_s, dtype=torch.float32)).numpy().ravel()

    # post-processing (same rule)
    for t in range(1, len(pred)):
        if pred[t-1] - pred[t] > 1.0:
            pred[t] = pred[t-1] - 1.0

    pred = smooth(pred, k=2)

    # save CSV
    np.savetxt(
        os.path.join(infer_dir, f"{sheet}.csv"),
        pred,
        delimiter=","
    )

    # save plot
    plot_pred_curve(pred, sheet, infer_dir)

    summary_rows.append([sheet] + pred.tolist())


# ============================================================
# Save summary CSV
# ============================================================
max_len = max(len(r) - 1 for r in summary_rows)

aligned = []
for row in summary_rows:
    name = row[0]
    vals = row[1:]
    if len(vals) < max_len:
        vals += [""] * (max_len - len(vals))
    aligned.append([name] + vals)

pd.DataFrame(aligned).to_csv(
    os.path.join(infer_dir, "summary_51_100.csv"),
    index=False,
    header=False
)

print("\n>>> Summary CSV saved (51~100)")


# ============================================================
# Merge prediction plots (5×10 grid)
# ============================================================
def merge_results(save_dir, grid_cols=5, grid_rows=10,
                  out_name="summary_grid_51_100.png"):

    paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(paths) == 0:
        print("No images to merge.")
        return

    imgs = [Image.open(p) for p in paths[:grid_cols * grid_rows]]
    w, h = imgs[0].size

    grid_img = Image.new("RGB", (grid_cols * w, grid_rows * h), "white")

    for idx, img in enumerate(imgs):
        x = (idx % grid_cols) * w
        y = (idx // grid_cols) * h
        grid_img.paste(img, (x, y))

    out_path = os.path.join(save_dir, out_name)
    grid_img.save(out_path, dpi=(200, 200))
    print(f">>> Grid saved → {out_path}")

merge_results(infer_dir)