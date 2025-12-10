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


# ==== Correlation Loss ====
def corr_loss(pred, target):
    vx, vy = pred - pred.mean(), target - target.mean()
    denom = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum())
    return 1 - (vx * vy).sum() / (denom + 1e-8)


# ==== Visualization ====
def smooth(x, k=2):
    if k <= 1: return x
    return np.convolve(np.pad(x, (1,1), mode='edge'), np.ones(k)/k, mode='valid')[:len(x)]

def plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir):
    pred_s = smooth(pred)
    plt.figure(figsize=(10,4))
    idx = np.arange(len(true))

    plt.fill_between(idx, true, pred_s, where=true>pred_s, color='gray', alpha=0.25, label='Error region')

    plt.fill_between(idx, true, pred_s, where=true<pred_s, color='gray', alpha=0.25)
    plt.plot(idx, true, 'k-', lw=2, label="True ICP")
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP")
    plt.title(f"[Train] — Sheet: {sheet_name}\n"
              f"Corr={corr:.2f} | RMSE={rmse:.2f} | Acc={acc:.1f}%", fontsize=11)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("ICP (mmHg)", fontsize=11)

    plt.legend(fontsize=10, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{sheet_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()
    plt.close()


# ==== Loop over sheets ====
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir = "/home/brainlab/Workspace/jycha/ICP/results"
model_dir  = "/home/brainlab/Workspace/jycha/ICP/models"

results = []

for i in range(1, 51):  # 1~50
    sheet_name = f"HM_P_REV_24_{i:03d}"

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except Exception as e:
        print(f"[{sheet_name}] skipped — {type(e).__name__}: {e}")
        continue

    if "PIC at rest (mmHg)" not in df.columns:
        print(f"[{sheet_name}] no 'PIC at rest' column, skipped.")
        continue

    y = pd.to_numeric(df["PIC at rest (mmHg)"], errors="coerce").ffill().to_numpy().astype(np.float32).reshape(-1, 1)
    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy().astype(np.float32)

    if len(X) < 5:
        print(f"[{sheet_name}] insufficient samples ({len(X)}).")
        continue

    print(f"\n==== {sheet_name} | Data: {X.shape} ====")
    scaler_y = RobustScaler()
    y_s = scaler_y.fit_transform(y)

    all_true, all_pred = [], []

    # ---- LOOCV ----
    for j in range(len(X)):
        train_idx = [k for k in range(len(X)) if k != j]

        sx = RobustScaler()

        X_train, X_test = sx.fit_transform(X[train_idx]), sx.transform(X[[j]])
        y_train, y_test = y_s[train_idx], y_s[[j]]

        model = Flow2ICP(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

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

        all_pred.append(float(scaler_y.inverse_transform(y_pred).ravel()[0]))
        all_true.append(float(scaler_y.inverse_transform(y_test).ravel()[0]))


    # ---- Metrics ----
    true, pred = np.array(all_true), np.array(all_pred)

    # ---- Clipping + Smoothing ----
    max_drop = 1.0
    for k in range(1, len(pred)):
        if pred[k - 1] - pred[k] > max_drop:
            pred[k] = pred[k - 1] - max_drop

    pred = np.convolve(np.pad(pred, (1, 1), mode='edge'), np.ones(2) / 2, mode='valid')[:len(pred)]
    # pred = smooth(pred, k=2)

    corr, _ = pearsonr(true, pred)
    rmse = np.sqrt(((true - pred) ** 2).mean())
    acc = np.clip(100 * (1 - np.mean(np.abs(true - pred) / np.abs(true))), 0, 100)

    print(f"[{sheet_name}] Corr={corr:.3f} | RMSE={rmse:.3f} | Acc={acc:.2f}%")

    results.append([sheet_name, corr, rmse, acc])

    plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir)

    # ---- Save final model for this patient ----
    model_path = os.path.join(model_dir, f"{sheet_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model → {model_path}")


# ==== Summary Table ====
res_df = pd.DataFrame(results, columns=["Sheet", "Corr", "RMSE", "Acc"])
res_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
print("\nSummary saved to:", os.path.join(save_dir, "summary.csv"))


# ==== Merge all images ====
def merge_results(save_dir, grid_cols=5, grid_rows=10, out_name="summary_grid_vertical.png"):
    image_paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(image_paths) == 0:
        print("No result images found to merge.")
        return

    imgs = [Image.open(p) for p in image_paths]
    w, h = imgs[0].size

    grid_w, grid_h = grid_cols * w, grid_rows * h
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    for idx, img in enumerate(imgs[:grid_cols * grid_rows]):
        x = (idx % grid_cols) * w
        y = (idx // grid_cols) * h
        grid_img.paste(img, (x, y))

    grid_path = os.path.join(save_dir, out_name)
    grid_img.save(grid_path, dpi=(200, 200))
    print(f"Combined image saved: {grid_path}")

merge_results(save_dir, grid_cols=5, grid_rows=10)