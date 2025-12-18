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
# Loss
# ==========================================================
def corr_loss(pred, target):
    vx, vy = pred - pred.mean(), target - target.mean()
    denom = torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum())
    return 1 - (vx * vy).sum() / (denom + 1e-8)


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


def load_subject(excel_path, sheet_name):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=6).iloc[:32]
    except:
        return None, None

    if "PIC at rest (mmHg)" not in df.columns:
        return None, None

    y = pd.to_numeric(
        df["PIC at rest (mmHg)"],
        errors="coerce"
    ).ffill().astype(np.float32).values.reshape(-1,1)

    X = df.select_dtypes(include=[np.number]) \
          .fillna(0).astype(np.float32).values

    if len(X) < 5:
        return None, None

    return X, y


def plot_result(true, pred, corr, rmse, acc, sheet_name, save_dir):
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)

    idx = np.arange(len(true))

    plt.figure(figsize=(10,4))

    plt.fill_between(
        idx, true, pred,
        where=(true > pred),
        color='gray', alpha=0.25
    )
    plt.fill_between(
        idx, true, pred,
        where=(true < pred),
        color='gray', alpha=0.25
    )

    plt.plot(idx, true, 'k-', lw=2, label="True ICP")
    plt.plot(idx, pred, 'r--', lw=2, label="Predicted ICP")

    plt.title(
        f"[LOSO Test] — {sheet_name}\n"
        f"Corr={corr:.2f} | RMSE={rmse:.2f} | Acc={acc:.1f}%",
        fontsize=11
    )

    plt.xlabel("Sample Index")
    plt.ylabel("ICP (mmHg)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{sheet_name}.png"), dpi=200)
    plt.show()
    plt.close()


# ==========================================================
# Paths
# ==========================================================
excel_path = "/home/brainlab/Workspace/jycha/ICP/CH patients data_v2.xlsx"
save_dir   = "/home/brainlab/Workspace/jycha/ICP/results_LOSO"
model_dir  = "/home/brainlab/Workspace/jycha/ICP/models_LOSO"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ==========================================================
# LOSO CV
# ==========================================================
subjects = list(range(1, 51))
results = []

print("\n==============================")
print(" Running LOSO (Subject-level)")
print("==============================\n")

for test_id in subjects:

    sheet_test = f"HM_P_REV_24_{test_id:03d}"
    print(f"\n===== LOSO TEST SUBJECT: {sheet_test} =====")

    # --- Test subject ---
    X_test, y_test = load_subject(excel_path, sheet_test)
    if X_test is None:
        print(" -> skipped (no data)")
        continue

    # --- Train subjects ---
    X_train_list, y_train_list = [], []

    for train_id in subjects:
        if train_id == test_id:
            continue

        sheet_train = f"HM_P_REV_24_{train_id:03d}"
        Xt, yt = load_subject(excel_path, sheet_train)

        if Xt is not None:
            X_train_list.append(Xt)
            y_train_list.append(yt)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

    # --- Scaling ---
    scaler_x = RobustScaler().fit(X_train)
    scaler_y = RobustScaler().fit(y_train)

    X_train_s = scaler_x.transform(X_train)
    y_train_s = scaler_y.transform(y_train)
    X_test_s  = scaler_x.transform(X_test)

    # --- Train model ---
    model = Flow2ICP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    x_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train_s, dtype=torch.float32)

    for _ in range(3000):
        opt.zero_grad()
        pred = model(x_t)
        loss = 0.8 * F.mse_loss(pred, y_t) + 0.2 * corr_loss(pred, y_t)
        loss.backward()
        opt.step()

    # --- Test ---
    with torch.no_grad():
        pred_s = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()

    pred = scaler_y.inverse_transform(pred_s).ravel()
    true = y_test.ravel()

    # --- Clipping + smoothing ---
    # for k in range(1, len(pred)):
    #     if pred[k-1] - pred[k] > 1.0:
    #         pred[k] = pred[k-1] - 1.0
    #
    # pred = smooth(pred, k=2)

    # --- Metrics ---
    corr, _ = pearsonr(true, pred)
    rmse = np.sqrt(((true - pred)**2).mean())
    acc  = np.clip(100 * (1 - np.mean(np.abs(true - pred) / np.abs(true))), 0, 100)

    print(f"[{sheet_test}] Corr={corr:.3f} | RMSE={rmse:.3f} | Acc={acc:.2f}%")

    results.append([sheet_test, corr, rmse, acc])

    plot_result(true, pred, corr, rmse, acc, sheet_test, save_dir)

    # --- Save full checkpoint (model + scalers) ---
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_x": scaler_x,
            "scaler_y": scaler_y
        },
        os.path.join(model_dir, f"{sheet_test}.pth")
    )

    # --- Save weights only ---
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, f"{sheet_test}_state.pth")
    )


# ==========================================================
# Summary CSV
# ==========================================================
summary_df = pd.DataFrame(results,
                          columns=["Sheet", "Corr", "RMSE", "Acc"])
summary_df.to_csv(os.path.join(save_dir, "summary_LOSO.csv"), index=False)
print("\n>>> LOSO Summary saved.")


# ==========================================================
# Merge images (5×10)
# ==========================================================
def merge_results(save_dir, grid_cols=5, grid_rows=10,
                  out_name="summary_grid_LOSO.png"):

    paths = sorted(glob.glob(os.path.join(save_dir, "HM_P_REV_24_*.png")))
    if len(paths) == 0:
        return

    imgs = [Image.open(p) for p in paths]
    w, h = imgs[0].size

    grid = Image.new("RGB", (grid_cols*w, grid_rows*h), "white")

    for i, img in enumerate(imgs[:grid_cols*grid_rows]):
        x = (i % grid_cols) * w
        y = (i // grid_cols) * h
        grid.paste(img, (x, y))

    grid.save(os.path.join(save_dir, out_name), dpi=(200,200))
    print(f">>> Grid saved → {out_name}")

merge_results(save_dir)


# ==========================================================
# Build averaged LOSO model
# ==========================================================
def build_avg_loso_model(model_dir, d_in, out_path):
    state_files = sorted(glob.glob(os.path.join(model_dir, "*_state.pth")))
    assert len(state_files) > 0, "No LOSO states found"

    avg_state = None
    for f in state_files:
        state = torch.load(f, map_location="cpu")
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k]

    for k in avg_state:
        avg_state[k] /= len(state_files)

    avg_model = Flow2ICP(d_in)
    avg_model.load_state_dict(avg_state)

    torch.save(avg_model.state_dict(), out_path)
    print(f">>> Averaged LOSO model saved → {out_path}")


build_avg_loso_model(
    model_dir=model_dir,
    d_in=X_train.shape[1],
    out_path=os.path.join(model_dir, "Flow2ICP_LOSO_avg.pth")
)