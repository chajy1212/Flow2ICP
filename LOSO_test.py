# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import RobustScaler
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


def load_subject_X(excel_path, sheet_name):
    """
    test-only:
    - PIC at rest column이 없어도 됨
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
        return None

    X = df.select_dtypes(include=[np.number]) \
          .fillna(0).astype(np.float32).values

    if len(X) < 5:
        return None

    return X


def plot_pred_curve(pred, sheet, save_dir):
    pred_s = smooth(pred)

    plt.figure(figsize=(8,3))
    plt.plot(pred_s, 'r-', lw=2)
    plt.title(f"[Predicted ICP] {sheet}")
    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{sheet}.png"), dpi=200)
    plt.show()
    plt.close()


# ==========================================================
# Paths
# ==========================================================
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
model_path = "/home/brainlab/Workspace/jycha/ICP/models_LOSO/final_loso_model.pth"

save_dir   = "/home/brainlab/Workspace/jycha/ICP/inference_LOSO_51_100"
os.makedirs(save_dir, exist_ok=True)


# ==========================================================
# Load model
# ==========================================================
# input dim은 아무 학습 subject 하나로 결정
X_ref = load_subject_X(excel_path, "HM_P_REV_24_001")
assert X_ref is not None, "Reference subject load failed."

model = Flow2ICP(X_ref.shape[1])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("\n==============================")
print(" LOSO Final Model Inference")
print(" Subjects: 51–100 (test-only)")
print("==============================\n")


# ==========================================================
# Inference (51–100)
# ==========================================================
summary_rows = []

for i in range(51, 101):
    sheet = f"HM_P_REV_24_{i:03d}"
    print(f"[Predict] {sheet}")

    X = load_subject_X(excel_path, sheet)
    if X is None:
        print("  -> skipped")
        continue

    # subject-wise robust scaling (test-time normalization)
    scaler_x = RobustScaler().fit(X)
    X_s = scaler_x.transform(X)

    with torch.no_grad():
        pred_s = model(torch.tensor(X_s, dtype=torch.float32)).numpy()

    pred = pred_s.ravel()

    # clipping + smoothing (same rule as LOSO test)
    for k in range(1, len(pred)):
        if pred[k-1] - pred[k] > 1.0:
            pred[k] = pred[k-1] - 1.0

    pred = smooth(pred, k=2)

    # save per-subject CSV
    np.savetxt(
        os.path.join(save_dir, f"{sheet}.csv"),
        pred,
        delimiter=","
    )

    # plot
    plot_pred_curve(pred, sheet, save_dir)

    summary_rows.append([sheet] + pred.tolist())


# ==========================================================
# Summary CSV (aligned)
# ==========================================================
max_len = max(len(r)-1 for r in summary_rows)

aligned = []
for r in summary_rows:
    name, vals = r[0], r[1:]
    vals += [""] * (max_len - len(vals))
    aligned.append([name] + vals)

pd.DataFrame(aligned).to_csv(
    os.path.join(save_dir, "summary_LOSO_51_100.csv"),
    index=False,
    header=False
)

print("\n>>> Test-only inference completed.")
print(f">>> Results saved to: {save_dir}")


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

merge_results(save_dir)