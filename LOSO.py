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

    def forward(self, x):
        return self.net(x)


# ==== Correlation Loss ====
def corr_loss(pred, target):
    vx, vy = pred - pred.mean(), target - target.mean()
    denom = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    return 1 - (vx * vy).sum() / (denom + 1e-8)


# ==== Visualization ====
def smooth(x, k=2):
    if k <= 1: return x
    return np.convolve(np.pad(x, (1, 1), mode='edge'), np.ones(k) / k, mode='valid')[:len(x)]


def plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir):
    pred_s = smooth(pred)
    plt.figure(figsize=(10, 4))
    idx = np.arange(len(true))

    plt.fill_between(idx, true, pred_s, where=true > pred_s, color='gray', alpha=0.25)
    plt.fill_between(idx, true, pred_s, where=true < pred_s, color='gray', alpha=0.25)

    plt.plot(idx, true, 'k-', lw=2, label="True ICP")
    plt.plot(idx, pred_s, 'r--', lw=2, label="Predicted ICP")

    plt.title(f"[LOSO Test] — Sheet: {sheet_name}\n"
              f"Corr={corr:.2f} | RMSE={rmse:.2f} | Acc={acc:.1f}%", fontsize=11)

    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{sheet_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


# ==== Paths ====
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir = "/home/brainlab/Workspace/jycha/ICP/results_LOSO"
model_dir = "/home/brainlab/Workspace/jycha/ICP/models_LOSO"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ==========================================================================
# 1) LOSO TRAINING (1~50번 환자를 subject-out 방식으로 평가)
# ==========================================================================
print("\n========================")
print("   RUNNING LOSO CV")
print("========================\n")

results = []


def load_subject(sheet_name):
    """ 한 subject의 (X, y) 로드 """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).dropna(how="all")
    except:
        return None, None

    if "PIC at rest (mmHg)" not in df.columns:
        return None, None

    y = pd.to_numeric(df["PIC at rest (mmHg)"], errors="coerce").ffill().to_numpy().astype(np.float32).reshape(-1, 1)
    X = df.select_dtypes(include=[np.number]).fillna(0).to_numpy().astype(np.float32)

    if len(X) < 5:
        return None, None

    return X, y


subjects = list(range(1, 51))

for test_id in subjects:

    sheet_test = f"HM_P_REV_24_{test_id:03d}"
    print(f"\n===== LOSO TEST SUBJECT: {sheet_test} =====")

    # --- Load Test Subject ---
    X_test, y_test = load_subject(sheet_test)
    if X_test is None:
        print(f"[{sheet_test}] skipped (no data)")
        continue

    # --- Load Train Subjects ---
    X_train_list, y_train_list = [], []

    for train_id in subjects:
        if train_id == test_id:
            continue

        sheet_train = f"HM_P_REV_24_{train_id:03d}"
        Xt, yt = load_subject(sheet_train)

        if Xt is not None:
            X_train_list.append(Xt)
            y_train_list.append(yt)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

    # --- Scaling (Train fit → Test transform) ---
    scaler_x = RobustScaler().fit(X_train)
    scaler_y = RobustScaler().fit(y_train)

    X_train_s = scaler_x.transform(X_train)
    y_train_s = scaler_y.transform(y_train)

    X_test_s = scaler_x.transform(X_test)

    # --- Train Model ---
    model = Flow2ICP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    x_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train_s, dtype=torch.float32)

    for epoch in range(3000):
        opt.zero_grad()
        pred = model(x_t)
        loss = 0.8 * F.mse_loss(pred, y_t) + 0.2 * corr_loss(pred, y_t)
        loss.backward()
        opt.step()

    # --- Prediction ---
    with torch.no_grad():
        pred_s = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()

    pred = scaler_y.inverse_transform(pred_s).ravel()
    true = y_test.ravel()

    # --- Clipping + smoothing ---
    for k in range(1, len(pred)):
        if pred[k - 1] - pred[k] > 1.0:
            pred[k] = pred[k - 1] - 1.0

    pred = smooth(pred, k=2)

    # --- Metrics ---
    corr, _ = pearsonr(true, pred)
    rmse = np.sqrt(((true - pred) ** 2).mean())
    acc = np.clip(100 * (1 - np.mean(np.abs(true - pred) / np.abs(true))), 0, 100)

    print(f"[{sheet_test}] Corr={corr:.3f} | RMSE={rmse:.3f} | Acc={acc:.2f}%")

    results.append([sheet_test, corr, rmse, acc])

    # --- Plot ---
    plot_result(true, pred, corr, rmse, acc, sheet_test, save_dir)

    # --- Save model ---
    torch.save(model.state_dict(), os.path.join(model_dir, f"{sheet_test}.pth"))


# === Save Summary CSV ===
summary_df = pd.DataFrame(results, columns=["Sheet", "Corr", "RMSE", "Acc"])
summary_df.to_csv(os.path.join(save_dir, "summary_LOSO.csv"), index=False)
print("\n>>> LOSO Summary saved.")


# === Merge Images ===
def merge_results(save_dir, grid_cols=5, grid_rows=10, out_name="summary_grid_LOSO.png"):
    paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(paths) == 0:
        return

    imgs = [Image.open(p) for p in paths]
    w, h = imgs[0].size

    grid_img = Image.new("RGB", (grid_cols * w, grid_rows * h), "white")

    for idx, img in enumerate(imgs[:grid_cols * grid_rows]):
        x = (idx % grid_cols) * w
        y = (idx // grid_cols) * h
        grid_img.paste(img, (x, y))

    grid_img.save(os.path.join(save_dir, out_name), dpi=(200, 200))


merge_results(save_dir)


# ==========================================================================
# 2) FINAL MODEL (1~50 전체 훈련) → 51~100 예측
# ==========================================================================
print("\n==== Training FINAL MODEL on subjects 1~50 ====\n")

X_all, y_all = [], []

for i in range(1, 51):
    Xd, yd = load_subject(f"HM_P_REV_24_{i:03d}")
    if Xd is not None:
        X_all.append(Xd)
        y_all.append(yd)

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)

scaler_x = RobustScaler().fit(X_all)
scaler_y = RobustScaler().fit(y_all)

X_s = scaler_x.transform(X_all)
y_s = scaler_y.transform(y_all)

x_t = torch.tensor(X_s, dtype=torch.float32)
y_t = torch.tensor(y_s, dtype=torch.float32)

final_model = Flow2ICP(X_all.shape[1])
opt = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(3000):
    opt.zero_grad()
    pred = final_model(x_t)
    loss = 0.8 * F.mse_loss(pred, y_t) + 0.2 * corr_loss(pred, y_t)
    loss.backward()
    opt.step()

torch.save(final_model.state_dict(), os.path.join(model_dir, "final_model.pth"))
joblib.dump(scaler_x, os.path.join(model_dir, "scaler_x.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

print(">>> FINAL MODEL SAVED")


# ======================================================================
# 3) 51~100 ICP PREDICTION USING FINAL MODEL
# ======================================================================
print("\n==== Predicting ICP for subjects 51~100 ====\n")

infer_dir = "/home/brainlab/Workspace/jycha/ICP/inference_51_100"
os.makedirs(infer_dir, exist_ok=True)

# Load model + scalers
final_model = Flow2ICP(X_all.shape[1])
final_model.load_state_dict(torch.load(os.path.join(model_dir, "final_model.pth")))
final_model.eval()

scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

summary_rows = []

for i in range(51, 101):
    sheet = f"HM_P_REV_24_{i:03d}"
    print(f"[Predict] {sheet}")

    X, _ = load_subject(sheet)
    if X is None:
        print(f" - skipped (no numeric data)")
        continue

    # Scaling
    X_s = scaler_x.transform(X)

    # Predict
    with torch.no_grad():
        pred_s = final_model(torch.tensor(X_s, dtype=torch.float32)).numpy()

    pred = scaler_y.inverse_transform(pred_s).ravel()

    # Clipping + smoothing
    for t in range(1, len(pred)):
        if pred[t-1] - pred[t] > 1.0:
            pred[t] = pred[t-1] - 1.0

    pred = smooth(pred, k=2)

    # Save CSV
    csv_path = os.path.join(infer_dir, f"{sheet}.csv")
    np.savetxt(csv_path, pred, delimiter=",")

    # Save plot
    plt.figure(figsize=(10,4))
    plt.plot(pred, "r-", lw=2)
    plt.title(f"[Predicted ICP] {sheet}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(infer_dir, f"{sheet}.png"), dpi=200)
    plt.show()
    plt.close()

    summary_rows.append([sheet] + pred.tolist())

# ---- Save summary CSV (aligned length) ----
max_len = max(len(row) - 1 for row in summary_rows)

aligned = []
for row in summary_rows:
    sheet, vals = row[0], row[1:]
    if len(vals) < max_len:
        vals = vals + ["" for _ in range(max_len - len(vals))]
    aligned.append([sheet] + vals)

pd.DataFrame(aligned).to_csv(os.path.join(infer_dir, "summary_51_100.csv"),
                             index=False, header=False)

print("\n>>> Summary for 51~100 saved.")


# ======================================================================
# 4) Merge 51~100 images
# ======================================================================
def merge_inference(infer_dir, grid_cols=5, grid_rows=10, out="summary_grid_51_100.png"):
    img_paths = sorted(glob.glob(os.path.join(infer_dir, "HM_P_REV_24_*.png")))
    img_paths = [p for p in img_paths if any(f"{i:03d}" in p for i in range(51, 101))]

    if len(img_paths) == 0:
        print("No images to merge.")
        return

    imgs = [Image.open(p) for p in img_paths]
    w, h = imgs[0].size

    grid_img = Image.new("RGB", (grid_cols*w, grid_rows*h), "white")

    for idx, img in enumerate(imgs[:grid_cols*grid_rows]):
        x = (idx % grid_cols) * w
        y = (idx // grid_cols) * h
        grid_img.paste(img, (x, y))

    grid_img.save(os.path.join(infer_dir, out), dpi=(200,200))
    print(f">>> Merged image saved → {out}")

merge_inference(infer_dir)