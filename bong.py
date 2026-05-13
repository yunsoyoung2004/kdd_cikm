import os
import random
import argparse
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from huggingface_hub import login, snapshot_download


# ============================================================
# Utils
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_list(xs):
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))


def mean_rows(rows):
    if len(rows) == 0:
        return [float("nan")] * 4
    n = len(rows[0])
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def safe_corr_from_lists(e, u):
    if len(e) <= 1:
        return None

    e_mean = sum(e) / len(e)
    u_mean = sum(u) / len(u)

    e_var = sum((v - e_mean) ** 2 for v in e) / len(e)
    u_var = sum((v - u_mean) ** 2 for v in u) / len(u)

    if e_var <= 1e-12 or u_var <= 1e-12:
        return None

    cov = sum((e[i] - e_mean) * (u[i] - u_mean) for i in range(len(e))) / len(e)
    return cov / ((e_var ** 0.5) * (u_var ** 0.5))


def minmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def entropy_np(values):
    x = np.asarray(values, dtype=np.float32)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    p = x / (x.sum() + 1e-6)
    return float(-(p * np.log(p + 1e-6)).sum())


# ============================================================
# HF
# ============================================================

def hf_download_three_repos(args):
    if args.hf_token:
        login(token=args.hf_token)

    os.makedirs(args.local_dir, exist_ok=True)

    snapshot_download(
        repo_id=args.subway_daily_repo,
        repo_type="dataset",
        local_dir=os.path.join(args.local_dir, "subway_daily"),
        local_dir_use_symlinks=False,
    )

    snapshot_download(
        repo_id=args.subway_30min_repo,
        repo_type="dataset",
        local_dir=os.path.join(args.local_dir, "subway_30min"),
        local_dir_use_symlinks=False,
    )

    snapshot_download(
        repo_id=args.mobility_cell_repo,
        repo_type="dataset",
        local_dir=os.path.join(args.local_dir, "mobility_cell"),
        local_dir_use_symlinks=False,
    )

    print("[HF DOWNLOAD DONE]", args.local_dir)


# ============================================================
# Loaders
# ============================================================

def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(path)


def find_files(root, suffix=None, keyword=None):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if suffix and not f.endswith(suffix):
                continue
            if keyword and keyword.lower() not in f.lower():
                continue
            out.append(p)
    return sorted(out)


def pick_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Cannot find {candidates}. Available={df.columns.tolist()}")
    return None


def load_mapping(local_dir):
    files = find_files(local_dir, suffix=".csv", keyword="cell_to_nearest_station")
    if len(files) == 0:
        files = find_files(local_dir, suffix=".csv", keyword="mapping")
    if len(files) == 0:
        files = find_files(local_dir, suffix=".csv")

    if len(files) == 0:
        raise FileNotFoundError("mapping csv not found")

    df = read_table(files[0])
    print("[MAPPING]", files[0], df.shape)
    print("[MAPPING COLS]", df.columns.tolist())
    return df


def load_many_parquets(folder, max_files=None):
    files = find_files(folder, suffix=".parquet")
    if max_files:
        files = files[:max_files]

    print(f"[PARQUET FILES] {folder}: {len(files)}")

    dfs = []
    for i, p in enumerate(files):
        try:
            d = pd.read_parquet(p)
            d["source_file"] = os.path.basename(p)
            dfs.append(d)

            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print(f"[LOAD PROGRESS] {i+1}/{len(files)}")

        except Exception as e:
            print("[SKIP]", p, e)

    if len(dfs) == 0:
        raise RuntimeError(f"No parquet loaded from {folder}")

    df = pd.concat(dfs, ignore_index=True)
    print("[LOAD]", folder, df.shape)
    print(df.columns.tolist())
    return df


# ============================================================
# Feature Engineering
# ============================================================

def build_station_tensor(subway_df):
    station_col = pick_col(
        subway_df,
        ["MASTER_STATION_ID", "station_id", "STATION_ID", "역사_ID", "역ID", "station"]
    )

    time_col = pick_col(
        subway_df,
        ["datetime", "date", "BASE_DATE", "USE_DATE", "use_date", "기준일자", "time"],
        required=False,
    )

    if time_col is None:
        subway_df["time_key"] = subway_df["source_file"]
        time_col = "time_key"

    out_col = pick_col(subway_df, ["od_out_cnt", "out_cnt", "outflow", "out"], required=False)
    in_col = pick_col(subway_df, ["od_in_cnt", "in_cnt", "inflow", "in"], required=False)

    if out_col is None or in_col is None:
        numeric_cols = [c for c in subway_df.columns if pd.api.types.is_numeric_dtype(subway_df[c])]
        if len(numeric_cols) < 2:
            raise ValueError("Need at least two numeric flow columns.")
        out_col, in_col = numeric_cols[:2]

    g = subway_df.groupby([time_col, station_col])[[out_col, in_col]].sum().reset_index()

    g["out_flow"] = g[out_col].astype(float)
    g["in_flow"] = g[in_col].astype(float)
    g["net_flow"] = g["in_flow"] - g["out_flow"]
    g["total_flow"] = g["in_flow"] + g["out_flow"]
    g["imbalance"] = np.abs(g["net_flow"]) / (g["total_flow"] + 1e-6)

    flow_cols = ["out_flow", "in_flow", "net_flow", "total_flow", "imbalance"]

    times = sorted(g[time_col].unique())
    stations = sorted(g[station_col].unique())

    t2i = {t: i for i, t in enumerate(times)}
    s2i = {s: i for i, s in enumerate(stations)}

    X = np.zeros((len(times), len(stations), len(flow_cols)), dtype=np.float32)

    for _, r in g.iterrows():
        X[t2i[r[time_col]], s2i[r[station_col]], :] = r[flow_cols].values.astype(np.float32)

    X[:, :, 0] = np.log1p(np.maximum(X[:, :, 0], 0))
    X[:, :, 1] = np.log1p(np.maximum(X[:, :, 1], 0))
    X[:, :, 2] = np.sign(X[:, :, 2]) * np.log1p(np.abs(X[:, :, 2]))
    X[:, :, 3] = np.log1p(np.maximum(X[:, :, 3], 0))
    X[:, :, 4] = np.clip(X[:, :, 4], 0, 1)

    print("[X]", X.shape)
    print("[FLOW COLS]", flow_cols)
    return X, stations, times, flow_cols, s2i


def build_cell_context(cell_df, mapping_df, station_to_idx):
    map_station_col = pick_col(mapping_df, ["MASTER_STATION_ID", "station_id", "STATION_ID"])
    map_cell_col = pick_col(mapping_df, ["CELL_ID", "cell_id", "CELL_ID_BASE", "o_cell", "d_cell"])

    cell_col = pick_col(cell_df, ["CELL_ID", "cell_id", "CELL_ID_BASE", "o_cell", "d_cell"])

    out_col = pick_col(cell_df, ["out_cnt", "mobility_out_sum", "outflow", "out"], required=False)
    in_col = pick_col(cell_df, ["in_cnt", "mobility_in_sum", "inflow", "in"], required=False)

    if out_col is None or in_col is None:
        numeric_cols = [c for c in cell_df.columns if pd.api.types.is_numeric_dtype(cell_df[c])]
        if len(numeric_cols) < 2:
            raise ValueError("Need at least two numeric cell flow columns.")
        out_col, in_col = numeric_cols[:2]

    cell_agg = cell_df.groupby(cell_col)[[out_col, in_col]].sum().reset_index()
    cell_agg["cell_out"] = cell_agg[out_col].astype(float)
    cell_agg["cell_in"] = cell_agg[in_col].astype(float)
    cell_agg["cell_net"] = cell_agg["cell_in"] - cell_agg["cell_out"]
    cell_agg["cell_total"] = cell_agg["cell_in"] + cell_agg["cell_out"]
    cell_agg["cell_imbalance"] = np.abs(cell_agg["cell_net"]) / (cell_agg["cell_total"] + 1e-6)

    cell_cols = ["cell_out", "cell_in", "cell_net", "cell_total", "cell_imbalance"]

    merged = mapping_df[[map_station_col, map_cell_col]].merge(
        cell_agg[[cell_col] + cell_cols],
        left_on=map_cell_col,
        right_on=cell_col,
        how="left",
    )

    merged[cell_cols] = merged[cell_cols].fillna(0)

    ctx = merged.groupby(map_station_col)[cell_cols].agg(["mean", "sum", "max"])
    ctx.columns = [f"{a}_{b}" for a, b in ctx.columns]
    ctx = ctx.reset_index()

    C = np.zeros((len(station_to_idx), len(ctx.columns) - 1), dtype=np.float32)

    for _, r in ctx.iterrows():
        sid = r[map_station_col]
        if sid in station_to_idx:
            C[station_to_idx[sid]] = r.drop(map_station_col).values.astype(np.float32)

    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    for j in range(C.shape[1]):
        if "imbalance" not in ctx.columns[j + 1]:
            C[:, j] = np.sign(C[:, j]) * np.log1p(np.abs(C[:, j]))

    C = (C - C.mean(axis=0, keepdims=True)) / (C.std(axis=0, keepdims=True) + 1e-6)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    print("[CELL CONTEXT]", C.shape)
    return C


# ============================================================
# RQ Output Analysis
# ============================================================

def save_rq1_observation_bias(subway_df, cell_df, mapping_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    station_col = pick_col(subway_df, ["STATION_ID", "MASTER_STATION_ID", "station_id"])
    time_col = pick_col(subway_df, ["datetime", "date", "BASE_DATE", "time"], required=False)
    if time_col is None:
        subway_df["time_key"] = subway_df["source_file"]
        time_col = "time_key"

    out_col = pick_col(subway_df, ["od_out_cnt", "out_cnt", "outflow"], required=False)
    in_col = pick_col(subway_df, ["od_in_cnt", "in_cnt", "inflow"], required=False)

    cell_col = pick_col(cell_df, ["CELL_ID_BASE", "CELL_ID", "cell_id"])
    cell_out = pick_col(cell_df, ["out_cnt", "mobility_out_sum", "outflow"], required=False)
    cell_in = pick_col(cell_df, ["in_cnt", "mobility_in_sum", "inflow"], required=False)

    map_station_col = pick_col(mapping_df, ["MASTER_STATION_ID", "STATION_ID", "station_id"])
    map_cell_col = pick_col(mapping_df, ["CELL_ID", "CELL_ID_BASE", "cell_id"])

    subway_df["subway_total"] = subway_df[out_col].fillna(0) + subway_df[in_col].fillna(0)
    subway_df["observed"] = (subway_df["subway_total"] > 0).astype(int)

    station_time = subway_df.groupby([station_col, time_col])["observed"].max().reset_index()
    T = station_time[time_col].nunique()

    coverage = station_time.groupby(station_col)["observed"].sum().reset_index(name="observed_steps")
    coverage["coverage"] = coverage["observed_steps"] / max(T, 1)
    coverage["sparsity"] = 1 - coverage["coverage"]

    temporal = subway_df.groupby([station_col, time_col])["subway_total"].sum().reset_index()
    ent = temporal.groupby(station_col)["subway_total"].apply(entropy_np).reset_index(name="temporal_entropy")

    station_subway = subway_df.groupby(station_col)["subway_total"].sum().reset_index()
    station_subway["subway_norm"] = minmax_np(station_subway["subway_total"].values)

    cell_agg = cell_df.groupby(cell_col)[[cell_out, cell_in]].sum().reset_index()
    cell_agg["cell_total"] = cell_agg[cell_out] + cell_agg[cell_in]
    cell_agg["cell_imbalance"] = (
        (cell_agg[cell_in] - cell_agg[cell_out]).abs()
        / (cell_agg["cell_total"] + 1e-6)
    )

    mapped = mapping_df[[map_station_col, map_cell_col]].merge(
        cell_agg[[cell_col, "cell_total", "cell_imbalance"]],
        left_on=map_cell_col,
        right_on=cell_col,
        how="left",
    ).fillna(0)

    station_cell = mapped.groupby(map_station_col)[["cell_total", "cell_imbalance"]].agg(
        {"cell_total": "sum", "cell_imbalance": "mean"}
    ).reset_index()
    station_cell["cell_norm"] = minmax_np(station_cell["cell_total"].values)

    diag = coverage.merge(ent, on=station_col, how="left")
    diag = diag.merge(station_subway, on=station_col, how="left")
    diag = diag.merge(station_cell, left_on=station_col, right_on=map_station_col, how="left")

    diag["hidden_demand_score"] = diag["cell_norm"].fillna(0) - diag["subway_norm"].fillna(0)
    diag["observation_bias_score"] = (
        diag["sparsity"].fillna(0)
        + minmax_np(diag["cell_imbalance"].fillna(0).values)
        + minmax_np(diag["hidden_demand_score"].fillna(0).values)
    ) / 3

    diag.to_csv(os.path.join(out_dir, "rq1_station_observation_bias.csv"), index=False, encoding="utf-8-sig")
    diag.sort_values("hidden_demand_score", ascending=False).head(50).to_csv(
        os.path.join(out_dir, "rq4_top50_hidden_demand.csv"), index=False, encoding="utf-8-sig"
    )
    diag.sort_values("observation_bias_score", ascending=False).head(50).to_csv(
        os.path.join(out_dir, "rq4_top50_observation_bias.csv"), index=False, encoding="utf-8-sig"
    )

    print("[RQ1/RQ4 SAVED]", out_dir)
    return diag, station_col


@torch.no_grad()
def save_station_uncertainty(model, loader, A, device, station_ids, out_dir, phase="TEST"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    n = len(station_ids)
    err_sum = [0.0] * n
    unc_sum = [0.0] * n
    cnt = [0] * n

    for b in loader:
        x = b["x"].to(device)
        y = b["y"].to(device)
        context = b["context"].to(device)

        pred, _, unc = model(x, context, A)

        err = torch.abs(pred - y).detach().cpu().tolist()
        uu = unc.detach().cpu().tolist()

        for bi in range(len(err)):
            for si in range(n):
                vals_e = err[bi][si]
                vals_u = uu[bi][si]
                err_sum[si] += sum(vals_e) / max(len(vals_e), 1)
                unc_sum[si] += sum(vals_u) / max(len(vals_u), 1)
                cnt[si] += 1

    rows = []
    for i, sid in enumerate(station_ids):
        c = max(cnt[i], 1)
        rows.append({
            "station_id": sid,
            "mean_abs_error": err_sum[i] / c,
            "mean_uncertainty": unc_sum[i] / c,
        })

    df = pd.DataFrame(rows)
    df["error_rank"] = df["mean_abs_error"].rank(ascending=False)
    df["uncertainty_rank"] = df["mean_uncertainty"].rank(ascending=False)

    path = os.path.join(out_dir, f"{phase.lower()}_station_uncertainty.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("[RQ3/RQ4 STATION UNCERTAINTY SAVED]", path)
    return df


# ============================================================
# Graph
# ============================================================

def build_adjacency(mapping_df, station_ids, graph_topk=10, graph_threshold=0.0, graph_type="topk"):
    station_col = pick_col(mapping_df, ["MASTER_STATION_ID", "station_id", "STATION_ID"])
    cell_col = pick_col(mapping_df, ["CELL_ID", "cell_id", "CELL_ID_BASE", "o_cell", "d_cell"])

    N = len(station_ids)
    station_cells = mapping_df.groupby(station_col)[cell_col].apply(set).to_dict()

    sim_mat = np.eye(N, dtype=np.float32)

    for i, si in enumerate(station_ids):
        ci = station_cells.get(si, set())
        for j in range(i + 1, N):
            sj = station_ids[j]
            cj = station_cells.get(sj, set())

            if len(ci) == 0 or len(cj) == 0:
                continue

            sim = len(ci & cj) / max(len(ci | cj), 1)

            if sim > graph_threshold:
                sim_mat[i, j] = sim
                sim_mat[j, i] = sim

    if graph_type == "identity":
        A = np.eye(N, dtype=np.float32)

    elif graph_type == "full":
        A = sim_mat.copy()
        if A.sum() <= N:
            A = A + 0.01

    else:
        A = np.eye(N, dtype=np.float32)

        for i in range(N):
            row = sim_mat[i].copy()
            row[i] = -1

            idx = np.argsort(row)[::-1][:graph_topk]

            for j in idx:
                if row[j] > graph_threshold:
                    A[i, j] = row[j]

        A = np.maximum(A, A.T)

    D = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(D + 1e-6))
    A = D_inv @ A @ D_inv

    density = float((A > 0).mean())
    print("[A]", A.shape, "density", density, "graph_type", graph_type, "topk", graph_topk)
    return A.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class UrbanDataset(Dataset):
    def __init__(self, X, C, window=12, horizon=1, mask_ratio=0.3):
        self.X = X.astype(np.float32)
        self.C = C.astype(np.float32)
        self.C_list = self.C.tolist()

        self.window = window
        self.horizon = horizon
        self.mask_ratio = mask_ratio

        self.mean = self.X.mean(axis=(0, 1), keepdims=True)
        self.std = self.X.std(axis=(0, 1), keepdims=True) + 1e-6
        self.Xn = (self.X - self.mean) / self.std
        self.Xn = np.nan_to_num(self.Xn, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return max(0, len(self.Xn) - self.window - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.Xn[idx:idx + self.window]
        y = self.Xn[idx + self.window + self.horizon - 1]

        mask = (np.random.rand(*x.shape) > self.mask_ratio).astype(np.float32)
        x_masked = x * mask

        return {
            "x": torch.tensor(x_masked.tolist(), dtype=torch.float32),
            "x_full": torch.tensor(x.tolist(), dtype=torch.float32),
            "mask": torch.tensor(mask.tolist(), dtype=torch.float32),
            "y": torch.tensor(y.tolist(), dtype=torch.float32),
            "context": torch.tensor(self.C_list, dtype=torch.float32),
        }


# ============================================================
# Model
# ============================================================

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        h = torch.einsum("ij,bjf->bif", A, x)
        h = self.lin(h)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class POUR(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim=128, latent_dim=128, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gconvs = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.temporal = nn.GRU(hidden_dim, latent_dim, batch_first=True, num_layers=1)
        self.z_norm = nn.LayerNorm(latent_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.unc_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),
        )

    def forward(self, x, context, A):
        B, W, N, Fdim = x.shape
        hs = []

        for t in range(W):
            h = torch.cat([x[:, t], context], dim=-1)
            h = self.input_proj(h)

            for g in self.gconvs:
                h = h + g(h, A)

            hs.append(h)

        h = torch.stack(hs, dim=1)
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, W, -1)

        out, _ = self.temporal(h)
        z = self.z_norm(out[:, -1])

        pred = self.pred_head(z).view(B, N, Fdim)
        recon = self.recon_head(z).view(B, N, Fdim)
        unc = self.unc_head(z).view(B, N, Fdim) + 1e-6

        return pred, recon, unc


# ============================================================
# Loss
# ============================================================

def masked_recon_loss(recon, target_last, mask_last):
    missing = 1.0 - mask_last
    return (((recon - target_last) ** 2) * missing).sum() / (missing.sum() + 1e-6)


def uncertainty_loss(pred, y, unc):
    return torch.mean(((pred - y) ** 2) / unc + torch.log(unc))


# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, A, opt, device, lambda_recon, lambda_unc, epoch_idx=0, total_epochs=0):
    model.train()
    logs = []
    total_batches = len(loader)

    print("\n" + "=" * 90)
    print(f"[TRAIN] Epoch {epoch_idx}/{total_epochs} | batches={total_batches}")
    print("=" * 90)

    for batch_idx, b in enumerate(loader):
        x = b["x"].to(device)
        x_full = b["x_full"].to(device)
        mask = b["mask"].to(device)
        y = b["y"].to(device)
        context = b["context"].to(device)

        opt.zero_grad()
        pred, recon, unc = model(x, context, A)

        lp = F.mse_loss(pred, y)
        lr = masked_recon_loss(recon, x_full[:, -1], mask[:, -1])
        lu = uncertainty_loss(pred, y, unc)

        loss = lp + lambda_recon * lr + lambda_unc * lu
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        logs.append([loss.item(), lp.item(), lr.item(), lu.item()])

        if batch_idx == 0 or (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            gpu_mem = 0.0
            if torch.cuda.is_available() and device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**3

            print(
                f"[E{epoch_idx:03d} B{batch_idx+1:04d}/{total_batches}] "
                f"loss={loss.item():.4f} pred={lp.item():.4f} "
                f"recon={lr.item():.4f} unc={lu.item():.4f} "
                f"grad={float(grad_norm):.4f} gpu={gpu_mem:.2f}GB"
            )

    loss, lp, lr, lu = mean_rows(logs)
    print(f"[TRAIN SUMMARY] loss={loss:.4f} pred={lp:.4f} recon={lr:.4f} unc={lu:.4f}")
    return loss, lp, lr, lu


@torch.no_grad()
def eval_model(model, loader, A, device, phase="VAL"):
    model.eval()
    maes, rmses, corrs = [], [], []

    print(f"\n[{phase}] batches={len(loader)}")

    for b in loader:
        x = b["x"].to(device)
        y = b["y"].to(device)
        context = b["context"].to(device)

        pred, _, unc = model(x, context, A)

        err = torch.abs(pred - y)
        mae = err.mean().item()
        rmse = torch.sqrt(((pred - y) ** 2).mean()).item()

        maes.append(mae)
        rmses.append(rmse)

        e = err.detach().cpu().flatten().tolist()
        u = unc.detach().cpu().flatten().tolist()
        corr = safe_corr_from_lists(e, u)

        if corr is not None:
            corrs.append(corr)

    return {
        "mae": mean_list(maes),
        "rmse": mean_list(rmses),
        "uncertainty_error_corr": mean_list(corrs) if len(corrs) else float("nan"),
    }


# ============================================================
# Main
# ============================================================

def main(args):
    set_seed(args.seed)

    if not args.skip_download:
        hf_download_three_repos(args)

    mapping = load_mapping(args.local_dir)

    subway_30min = load_many_parquets(
        os.path.join(args.local_dir, "subway_30min"),
        max_files=args.max_subway_files,
    )

    cell = load_many_parquets(
        os.path.join(args.local_dir, "mobility_cell"),
        max_files=args.max_cell_files,
    )

    rq_dir = os.path.join(args.output_dir, "rq_outputs")
    rq_diag = None
    rq_station_col = None

    if args.save_rq_outputs:
        rq_diag, rq_station_col = save_rq1_observation_bias(
            subway_30min.copy(),
            cell.copy(),
            mapping.copy(),
            rq_dir,
        )

    X, station_ids, time_ids, flow_cols, s2i = build_station_tensor(subway_30min)
    C = build_cell_context(cell, mapping, s2i)

    A_np = build_adjacency(
        mapping,
        station_ids,
        graph_topk=args.graph_topk,
        graph_threshold=args.graph_threshold,
        graph_type=args.graph_type,
    )

    T = X.shape[0]
    tr = int(T * 0.7)
    va = int(T * 0.85)

    train_ds = UrbanDataset(X[:tr], C, args.window, args.horizon, args.mask_ratio)
    val_ds = UrbanDataset(X[tr - args.window:va], C, args.window, args.horizon, args.mask_ratio)
    test_ds = UrbanDataset(X[va - args.window:], C, args.window, args.horizon, args.mask_ratio)

    print("[SPLIT]", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[DEVICE]", device)

    A = torch.tensor(A_np.tolist(), dtype=torch.float32).to(device)

    model = POUR(
        input_dim=X.shape[-1],
        context_dim=C.shape[-1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best = 1e18
    history = []

    for ep in range(1, args.epochs + 1):
        loss, lp, lr, lu = train_epoch(
            model, train_loader, A, opt, device,
            args.lambda_recon, args.lambda_unc,
            epoch_idx=ep,
            total_epochs=args.epochs,
        )

        val = eval_model(model, val_loader, A, device, phase="VAL")

        row = {
            "epoch": ep,
            "loss": loss,
            "pred_loss": lp,
            "recon_loss": lr,
            "unc_loss": lu,
            "val_mae": val["mae"],
            "val_rmse": val["rmse"],
            "val_uncertainty_error_corr": val["uncertainty_error_corr"],
        }
        history.append(row)

        print(
            f"[EPOCH {ep:03d} FINAL] "
            f"loss={loss:.4f} pred={lp:.4f} recon={lr:.4f} unc={lu:.4f} "
            f"val_mae={val['mae']:.4f} val_rmse={val['rmse']:.4f} "
            f"corr={val['uncertainty_error_corr']:.4f}"
        )

        if val["mae"] < best:
            best = val["mae"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "station_ids": station_ids,
                    "time_ids": time_ids,
                    "flow_cols": flow_cols,
                    "C": C.tolist(),
                    "A": A_np.tolist(),
                    "best_val_mae": best,
                },
                os.path.join(args.output_dir, "best_pour.pt"),
            )
            print(f"[BEST UPDATED] val_mae={best:.4f}")

        pd.DataFrame(history).to_csv(
            os.path.join(args.output_dir, "train_history.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    ckpt = torch.load(os.path.join(args.output_dir, "best_pour.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    test = eval_model(model, test_loader, A, device, phase="TEST")
    print("[TEST]", test)

    pd.DataFrame([test]).to_csv(
        os.path.join(args.output_dir, "test_result.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_rq_outputs:
        station_unc = save_station_uncertainty(
            model,
            test_loader,
            A,
            device,
            station_ids,
            rq_dir,
            phase="TEST",
        )

        if rq_diag is not None and rq_station_col is not None:
            merged = station_unc.merge(
                rq_diag,
                left_on="station_id",
                right_on=rq_station_col,
                how="left",
            )
            merged.to_csv(
                os.path.join(rq_dir, "rq4_hidden_demand_uncertainty_merged.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            print("[RQ4 MERGED SAVED]", os.path.join(rq_dir, "rq4_hidden_demand_uncertainty_merged.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--subway_daily_repo", type=str, default="youngbongbong/subway_od")
    parser.add_argument("--subway_30min_repo", type=str, default="youngbongbong/subway_30min_od")
    parser.add_argument("--mobility_cell_repo", type=str, default="youngbongbong/mobility_dataset")

    parser.add_argument("--local_dir", type=str, default="./hf_data")
    parser.add_argument("--output_dir", type=str, default="./outputs_pour")
    parser.add_argument("--skip_download", action="store_true")

    parser.add_argument("--max_subway_files", type=int, default=None)
    parser.add_argument("--max_cell_files", type=int, default=None)

    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--mask_ratio", type=float, default=0.3)

    parser.add_argument("--graph_type", type=str, default="topk", choices=["topk", "full", "identity"])
    parser.add_argument("--graph_topk", type=int, default=10)
    parser.add_argument("--graph_threshold", type=float, default=0.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--lambda_unc", type=float, default=0.05)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_rq_outputs", action="store_true")

    args = parser.parse_args()
    main(args)