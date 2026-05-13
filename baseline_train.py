import os, random, argparse, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_list(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


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


def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(path)


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

    if not dfs:
        raise RuntimeError(f"No parquet loaded from {folder}")

    df = pd.concat(dfs, ignore_index=True)
    print("[LOAD]", folder, df.shape)
    print(df.columns.tolist())
    return df


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
    return df


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
    return X, stations, times, flow_cols


def build_adjacency(mapping_df, station_ids, graph_topk=10, graph_type="topk"):
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
            if sim > 0:
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
                if row[j] > 0:
                    A[i, j] = row[j]
        A = np.maximum(A, A.T)

    D = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(D + 1e-6))
    A = D_inv @ A @ D_inv
    print("[A]", A.shape, "density", float((A > 0).mean()))
    return A.astype(np.float32)


class UrbanDataset(Dataset):
    def __init__(self, X, window=24, horizon=1, mask_ratio=0.1):
        self.X = X.astype(np.float32)
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
            "y": torch.tensor(y.tolist(), dtype=torch.float32),
        }


class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, A=None):
        B, W, N, Fdim = x.shape
        h = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, Fdim)
        out, _ = self.gru(h)
        z = self.norm(out[:, -1])
        pred = self.head(z).view(B, N, Fdim)
        return pred


class TemporalTransformerBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, A=None):
        B, W, N, Fdim = x.shape
        h = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, Fdim)
        h = self.proj(h)

        pos = torch.arange(W, device=x.device).float().view(1, W, 1)
        pe = torch.sin(pos / 10.0)
        h = h + pe

        z = self.encoder(h)[:, -1]
        z = self.norm(z)
        pred = self.head(z).view(B, N, Fdim)
        return pred


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


class STGCBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gconvs = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.gru = nn.GRU(hidden_dim, latent_dim, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, A):
        B, W, N, Fdim = x.shape
        hs = []

        for t in range(W):
            h = self.in_proj(x[:, t])
            for g in self.gconvs:
                h = h + g(h, A)
            hs.append(h)

        h = torch.stack(hs, dim=1)
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, W, -1)

        out, _ = self.gru(h)
        z = self.norm(out[:, -1])
        pred = self.head(z).view(B, N, Fdim)
        return pred


@torch.no_grad()
def eval_model(model, loader, A, device):
    model.eval()
    maes, rmses = [], []

    for b in loader:
        x = b["x"].to(device)
        y = b["y"].to(device)

        pred = model(x, A)
        err = torch.abs(pred - y)

        maes.append(err.mean().item())
        rmses.append(torch.sqrt(((pred - y) ** 2).mean()).item())

    return {
        "mae": mean_list(maes),
        "rmse": mean_list(rmses),
    }


def train_epoch(model, loader, A, opt, device, epoch, total_epochs):
    model.train()
    losses = []

    print("\n" + "=" * 80)
    print(f"[TRAIN] epoch={epoch}/{total_epochs} batches={len(loader)}")
    print("=" * 80)

    for i, b in enumerate(loader):
        x = b["x"].to(device)
        y = b["y"].to(device)

        opt.zero_grad()
        pred = model(x, A)

        loss = F.mse_loss(pred, y)

        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())

        if i == 0 or (i + 1) % 5 == 0 or (i + 1) == len(loader):
            gpu_mem = 0.0
            if torch.cuda.is_available() and device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
            print(
                f"[E{epoch:03d} B{i+1:04d}/{len(loader)}] "
                f"loss={loss.item():.4f} grad={float(grad):.4f} gpu={gpu_mem:.2f}GB"
            )

    return mean_list(losses)


def main(args):
    set_seed(args.seed)

    mapping = load_mapping(args.local_dir)

    subway = load_many_parquets(
        os.path.join(args.local_dir, "subway_30min"),
        max_files=args.max_subway_files,
    )

    X, station_ids, time_ids, flow_cols = build_station_tensor(subway)

    A_np = build_adjacency(
        mapping,
        station_ids,
        graph_topk=args.graph_topk,
        graph_type=args.graph_type,
    )

    T = X.shape[0]
    tr = int(T * 0.7)
    va = int(T * 0.85)

    train_ds = UrbanDataset(X[:tr], args.window, args.horizon, args.mask_ratio)
    val_ds = UrbanDataset(X[tr - args.window:va], args.window, args.horizon, args.mask_ratio)
    test_ds = UrbanDataset(X[va - args.window:], args.window, args.horizon, args.mask_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[DEVICE]", device)

    A = torch.tensor(A_np.tolist(), dtype=torch.float32).to(device)

    input_dim = X.shape[-1]

    if args.model == "gru":
        model = GRUBaseline(input_dim, args.hidden_dim, args.dropout)
    elif args.model == "transformer":
        model = TemporalTransformerBaseline(input_dim, args.hidden_dim, args.nhead, args.num_layers, args.dropout)
    elif args.model == "stgc":
        model = STGCBaseline(input_dim, args.hidden_dim, args.latent_dim, args.num_layers, args.dropout)
    else:
        raise ValueError(args.model)

    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)

    best = 1e18
    hist = []

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, A, opt, device, ep, args.epochs)
        val = eval_model(model, val_loader, A, device)

        row = {
            "epoch": ep,
            "loss": loss,
            "val_mae": val["mae"],
            "val_rmse": val["rmse"],
        }
        hist.append(row)

        print(f"[EPOCH {ep:03d}] loss={loss:.4f} val_mae={val['mae']:.4f} val_rmse={val['rmse']:.4f}")

        if val["mae"] < best:
            best = val["mae"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "station_ids": station_ids,
                    "time_ids": time_ids,
                    "flow_cols": flow_cols,
                    "A": A_np.tolist(),
                },
                os.path.join(args.output_dir, "best_baseline.pt")
            )
            print("[BEST UPDATED]", best)

        pd.DataFrame(hist).to_csv(
            os.path.join(args.output_dir, "train_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    ckpt = torch.load(os.path.join(args.output_dir, "best_baseline.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    test = eval_model(model, test_loader, A, device)
    test["model"] = args.model

    print("[TEST]", test)

    pd.DataFrame([test]).to_csv(
        os.path.join(args.output_dir, "test_result.csv"),
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["gru", "transformer", "stgc"])

    parser.add_argument("--local_dir", type=str, default="./hf_data")
    parser.add_argument("--output_dir", type=str, default="./baseline_outputs")

    parser.add_argument("--max_subway_files", type=int, default=20)

    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--mask_ratio", type=float, default=0.1)

    parser.add_argument("--graph_type", type=str, default="topk", choices=["topk", "full", "identity"])
    parser.add_argument("--graph_topk", type=int, default=10)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)