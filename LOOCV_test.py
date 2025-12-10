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
import joblib


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

    plt.fill_between(idx, true, pred_s, where=true>pred_s, color='gray', alpha=0.25)
    plt.fill_between(idx, true, pred_s, where=true<pred_s, color='gray', alpha=0.25)

    plt.plot(idx, true, 'k-', lw=2, label="True ICP")
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP")

    plt.title(f"[Train] — Sheet: {sheet_name}\n"
              f"Corr={corr:.2f} | RMSE={rmse:.2f} | Acc={acc:.1f}%", fontsize=11)

    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{sheet_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


# ==== Paths ====
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir = "/home/brainlab/Workspace/jycha/ICP/results"
model_dir  = "/home/brainlab/Workspace/jycha/ICP/models"


# ===========================================================
# 1) LOOCV (1~50번 환자) 학습
# ===========================================================
results = []

for i in range(1, 51):  # 1~50
    sheet_name = f"HM_P_REV_24_{i:03d}"

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
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


    # ---- Clipping + smoothing ----
    max_drop = 1.0
    for k in range(1, len(pred)):
        if pred[k - 1] - pred[k] > max_drop:
            pred[k] = pred[k - 1] - max_drop

    # pred = np.convolve(np.pad(pred, (1, 1), mode='edge'), np.ones(2) / 2, mode='valid')[:len(pred)]
    pred = smooth(pred, k=2)

    corr, _ = pearsonr(true, pred)
    rmse = np.sqrt(((true - pred) ** 2).mean())
    acc = np.clip(100 * (1 - np.mean(np.abs(true - pred) / np.abs(true))), 0, 100)

    print(f"[{sheet_name}] Corr={corr:.3f} | RMSE={rmse:.3f} | Acc={acc:.2f}%")

    results.append([sheet_name, corr, rmse, acc])

    plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, f"{sheet_name}.pth"))


# === Save summary ===
res_df = pd.DataFrame(results, columns=["Sheet", "Corr", "RMSE", "Acc"])
res_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
print("Summary saved.")


# === Merge images ===
def merge_results(save_dir, grid_cols=5, grid_rows=10, out_name="summary_grid_vertical.png"):
    paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(paths) == 0:
        print("No result images found to merge.")
        return

    imgs = [Image.open(p) for p in paths]
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


# ==================================================================
# 2) 1~50번 전체로 “최종 모델” 하나를 새로 학습
# ==================================================================
print("\n==== Training FINAL MODEL on subjects 1~50 ====\n")

X_all, y_all = [], []

for i in range(1, 51):
    sheet_name = f"HM_P_REV_24_{i:03d}"
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
        continue

    if "PIC at rest (mmHg)" not in df.columns:
        continue

    y = pd.to_numeric(df["PIC at rest (mmHg)"], errors="coerce").ffill().to_numpy().astype(np.float32).reshape(-1,1)
    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy().astype(np.float32)

    if len(X) < 5:
        continue

    X_all.append(X)
    y_all.append(y)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("Final Train Shape:", X_all.shape, y_all.shape)

# Scaling
scaler_x = RobustScaler()
scaler_y = RobustScaler()

X_s = scaler_x.fit_transform(X_all)
y_s = scaler_y.fit_transform(y_all)

# Convert
x_t = torch.tensor(X_s, dtype=torch.float32)
y_t = torch.tensor(y_s, dtype=torch.float32)

# Model
final_model = Flow2ICP(X_all.shape[1])
opt = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)

# Train
for epoch in range(3000):
    opt.zero_grad()
    pred = final_model(x_t)
    loss = 0.8 * F.mse_loss(pred, y_t) + 0.2 * corr_loss(pred, y_t)
    loss.backward()
    opt.step()

# Save
torch.save(final_model.state_dict(), os.path.join(model_dir, "final_model.pth"))
joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

print(">>> FINAL MODEL SAVED")


# ==================================================================
# 3) 시트 51~100번에 대해 ICP 예측
# ==================================================================
infer_dir = "/home/brainlab/Workspace/jycha/ICP/inference_51_100"
os.makedirs(infer_dir, exist_ok=True)

print("\n==== Predicting ICP for subjects 51~100 ====\n")

# Load trained model + scalers
final_model = Flow2ICP(X_all.shape[1])
final_model.load_state_dict(torch.load(os.path.join(model_dir, "final_model.pth")))
final_model.eval()

scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

for i in range(51, 101):
    sheet_name = f"HM_P_REV_24_{i:03d}"

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
        print(f"[{sheet_name}] skipped")
        continue

    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy().astype(np.float32)

    if len(X) < 5:
        print(f"[{sheet_name}] insufficient samples.")
        continue

    # Scaling
    X_s = scaler_x.transform(X)

    with torch.no_grad():
        pred_s = final_model(torch.tensor(X_s, dtype=torch.float32)).numpy()

    pred = scaler_y.inverse_transform(pred_s).ravel()

    # Clipping + smoothing
    for t in range(1, len(pred)):
        if pred[t-1] - pred[t] > 1.0:
            pred[t] = pred[t-1] - 1.0

    pred = smooth(pred, k=2)

    # Plot predicted curve
    plt.figure(figsize=(10,4))
    plt.plot(pred, 'r-', lw=2)
    plt.title(f"[Predicted ICP] {sheet_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(infer_dir, f"{sheet_name}.png"), dpi=200)
    plt.show()
    plt.close()

    # Save CSV for each subject
    np.savetxt(os.path.join(infer_dir, f"{sheet_name}.csv"), pred, delimiter=",")

    print(f"[{sheet_name}] prediction saved.")


# ==================================================================
# 4) 모든 subject(51~100)의 예측 값을 summary CSV로 저장
# ==================================================================
summary_list = []

for i in range(51, 101):
    sheet_name = f"HM_P_REV_24_{i:03d}"
    csv_path = os.path.join(infer_dir, f"{sheet_name}.csv")

    if not os.path.exists(csv_path):
        continue

    pred_vals = np.loadtxt(csv_path, delimiter=",")
    row = [sheet_name] + pred_vals.tolist()
    summary_list.append(row)

# 가장 긴 time-series 길이에 맞춰 패딩 (CSV 직렬화 안정성 확보)
max_len = max(len(row) - 1 for row in summary_list)

summary_aligned = []
for row in summary_list:
    name = row[0]
    vals = row[1:]
    if len(vals) < max_len:
        vals = vals + ["" for _ in range(max_len - len(vals))]
    summary_aligned.append([name] + vals)

summary_df = pd.DataFrame(summary_aligned)
summary_csv_path = os.path.join(infer_dir, "summary_51_100.csv")
summary_df.to_csv(summary_csv_path, index=False, header=False)

print(f"\n>>> Summary CSV saved → {summary_csv_path}")


# ==================================================================
# 5) 51~100 이미지 5×10 grid로 병합
# ==================================================================
def merge_inference_results(infer_dir, grid_cols=5, grid_rows=10,
                            out_name="summary_grid_51_100.png"):
    img_paths = sorted(glob.glob(os.path.join(infer_dir, "HM_P_REV_24_*.png")))
    img_paths = [p for p in img_paths if any(f"{i:03d}" in p for i in range(51, 101))]

    if len(img_paths) == 0:
        print("No inference images found.")
        return

    imgs = [Image.open(p) for p in img_paths]
    w, h = imgs[0].size
    grid_w, grid_h = grid_cols * w, grid_rows * h

    grid_img = Image.new("RGB", (grid_w, grid_h), color="white")

    for idx, img in enumerate(imgs[:grid_cols * grid_rows]):
        x = (idx % grid_cols) * w
        y = (idx // grid_cols) * h
        grid_img.paste(img, (x, y))

    out_path = os.path.join(infer_dir, out_name)
    grid_img.save(out_path, dpi=(200, 200))
    print(f">>> Combined inference image saved → {out_path}")

merge_inference_results(infer_dir)