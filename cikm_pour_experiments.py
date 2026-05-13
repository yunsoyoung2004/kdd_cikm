import os
import glob
import math
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


LOCAL_DIR = "./hf_data"
SCRIPT = "bong.py"
OUT_ROOT = "./cikm_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)


# ============================================================
# Utils
# ============================================================

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


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def entropy_from_values(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    p = x / (x.sum() + 1e-9)
    return float(-(p * np.log(p + 1e-9)).sum())


def run(cmd):
    print("\n" + "=" * 100)
    print(cmd)
    print("=" * 100)
    subprocess.run(cmd, shell=True, check=False)


# ============================================================
# Exp 1. Observation Bias Diagnosis
# ============================================================

def experiment_1_observation_bias():
    print("\n[EXP1] Observation Bias Diagnosis")

    mapping_files = find_files(LOCAL_DIR, suffix=".csv", keyword="cell_to_nearest_station")
    if len(mapping_files) == 0:
        mapping_files = find_files(LOCAL_DIR, suffix=".csv", keyword="mapping")
    mapping = read_table(mapping_files[0])

    subway_files = find_files(os.path.join(LOCAL_DIR, "subway_30min"), suffix=".parquet")
    cell_files = find_files(os.path.join(LOCAL_DIR, "mobility_cell"), suffix=".parquet")

    subway = pd.concat([read_table(p) for p in subway_files[:50]], ignore_index=True)
    cell = pd.concat([read_table(p) for p in cell_files[:50]], ignore_index=True)

    station_col = pick_col(subway, ["STATION_ID", "MASTER_STATION_ID", "station_id"])
    time_col = pick_col(subway, ["datetime", "date", "time", "BASE_DATE"])
    out_col = pick_col(subway, ["od_out_cnt", "out_cnt", "outflow"])
    in_col = pick_col(subway, ["od_in_cnt", "in_cnt", "inflow"])

    cell_col = pick_col(cell, ["CELL_ID_BASE", "CELL_ID", "cell_id"])
    cell_out = pick_col(cell, ["out_cnt", "mobility_out_sum", "outflow"])
    cell_in = pick_col(cell, ["in_cnt", "mobility_in_sum", "inflow"])

    map_station_col = pick_col(mapping, ["MASTER_STATION_ID", "STATION_ID", "station_id"])
    map_cell_col = pick_col(mapping, ["CELL_ID", "CELL_ID_BASE", "cell_id"])

    subway["total_flow"] = subway[out_col].fillna(0) + subway[in_col].fillna(0)
    subway["observed"] = (subway["total_flow"] > 0).astype(int)

    # Coverage(s_i) = sum_t 1(x_i,t > 0) / T
    station_time = (
        subway.groupby([station_col, time_col])["observed"]
        .max()
        .reset_index()
    )
    T = station_time[time_col].nunique()

    coverage = (
        station_time.groupby(station_col)["observed"]
        .sum()
        .reset_index(name="observed_steps")
    )
    coverage["coverage"] = coverage["observed_steps"] / max(T, 1)
    coverage["sparsity"] = 1 - coverage["coverage"]

    # Temporal entropy H_i = - sum_t p_i,t log p_i,t
    flow_by_time = (
        subway.groupby([station_col, time_col])["total_flow"]
        .sum()
        .reset_index()
    )
    entropy = (
        flow_by_time.groupby(station_col)["total_flow"]
        .apply(entropy_from_values)
        .reset_index(name="temporal_entropy")
    )

    station_flow = (
        subway.groupby(station_col)[[out_col, in_col, "total_flow"]]
        .sum()
        .reset_index()
    )
    station_flow["subway_observed_norm"] = minmax(station_flow["total_flow"])

    # Cell imbalance = |in-out| / (in+out)
    cell_agg = (
        cell.groupby(cell_col)[[cell_out, cell_in]]
        .sum()
        .reset_index()
    )
    cell_agg["cell_total"] = cell_agg[cell_out] + cell_agg[cell_in]
    cell_agg["cell_imbalance"] = (
        (cell_agg[cell_in] - cell_agg[cell_out]).abs()
        / (cell_agg["cell_total"] + 1e-9)
    )

    mapped = mapping[[map_station_col, map_cell_col]].merge(
        cell_agg,
        left_on=map_cell_col,
        right_on=cell_col,
        how="left"
    )
    mapped = mapped.fillna(0)

    station_cell = (
        mapped.groupby(map_station_col)[["cell_total", "cell_imbalance"]]
        .agg({"cell_total": "sum", "cell_imbalance": "mean"})
        .reset_index()
    )
    station_cell["cell_mobility_norm"] = minmax(station_cell["cell_total"])

    diag = coverage.merge(entropy, on=station_col, how="left")
    diag = diag.merge(station_flow, on=station_col, how="left")
    diag = diag.merge(
        station_cell,
        left_on=station_col,
        right_on=map_station_col,
        how="left"
    )

    diag["hidden_demand_score"] = (
        diag["cell_mobility_norm"].fillna(0)
        - diag["subway_observed_norm"].fillna(0)
    )
    diag["observation_bias_score"] = (
        diag["sparsity"].fillna(0)
        + minmax(diag["cell_imbalance"].fillna(0))
        + minmax(diag["hidden_demand_score"].fillna(0))
    ) / 3

    out_dir = os.path.join(OUT_ROOT, "exp1_observation_bias")
    os.makedirs(out_dir, exist_ok=True)

    diag.to_csv(
        os.path.join(out_dir, "station_observation_bias.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # Figure 1 coverage
    plt.figure(figsize=(8, 5))
    diag.sort_values("coverage")["coverage"].reset_index(drop=True).plot()
    plt.title("Station Observation Coverage")
    plt.xlabel("Stations sorted by coverage")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_station_coverage_curve.png"), dpi=200)
    plt.close()

    # Figure 2 entropy
    plt.figure(figsize=(8, 5))
    diag.sort_values("temporal_entropy")["temporal_entropy"].reset_index(drop=True).plot()
    plt.title("Temporal Observation Entropy")
    plt.xlabel("Stations sorted by entropy")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_temporal_entropy_curve.png"), dpi=200)
    plt.close()

    # Figure 3 hidden demand mismatch
    plt.figure(figsize=(6, 6))
    plt.scatter(
        diag["subway_observed_norm"],
        diag["cell_mobility_norm"],
        alpha=0.6
    )
    plt.xlabel("Normalized subway-observed mobility")
    plt.ylabel("Normalized cell-level mobility")
    plt.title("Hidden Demand Mismatch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_hidden_demand_mismatch.png"), dpi=200)
    plt.close()

    print("[EXP1 DONE]", out_dir)
    return diag


# ============================================================
# Exp 2. Masked Reconstruction Robustness
# ============================================================

def experiment_2_mask_sweep():
    print("\n[EXP2] Masked Reconstruction Robustness")

    mask_ratios = [0.1, 0.3, 0.5, 0.7]

    for mr in mask_ratios:
        out = os.path.join(OUT_ROOT, f"exp2_mask_{mr}")
        cmd = f"""
python {SCRIPT} \
  --skip_download \
  --local_dir "{LOCAL_DIR}" \
  --output_dir "{out}" \
  --max_subway_files 20 \
  --max_cell_files 50 \
  --window 12 \
  --horizon 1 \
  --mask_ratio {mr} \
  --batch_size 4 \
  --epochs 10 \
  --num_workers 0
"""
        run(cmd)


# ============================================================
# Exp 3. Ablation Study
# ============================================================

def experiment_3_ablation():
    print("\n[EXP3] Ablation Study")

    settings = {
        "FULL": {
            "lambda_recon": 1.0,
            "lambda_unc": 0.1,
        },
        "NO_RECON": {
            "lambda_recon": 0.0,
            "lambda_unc": 0.1,
        },
        "NO_UNCERTAINTY": {
            "lambda_recon": 1.0,
            "lambda_unc": 0.0,
        },
        "VANILLA": {
            "lambda_recon": 0.0,
            "lambda_unc": 0.0,
        },
    }

    for name, cfg in settings.items():
        out = os.path.join(OUT_ROOT, f"exp3_{name}")
        cmd = f"""
python {SCRIPT} \
  --skip_download \
  --local_dir "{LOCAL_DIR}" \
  --output_dir "{out}" \
  --max_subway_files 20 \
  --max_cell_files 50 \
  --window 12 \
  --horizon 1 \
  --mask_ratio 0.3 \
  --batch_size 4 \
  --epochs 10 \
  --lambda_recon {cfg["lambda_recon"]} \
  --lambda_unc {cfg["lambda_unc"]} \
  --num_workers 0
"""
        run(cmd)


# ============================================================
# Exp 4. Uncertainty Calibration Analysis
# ============================================================

def experiment_4_uncertainty_summary():
    print("\n[EXP4] Uncertainty Calibration Summary")

    rows = []

    for f in glob.glob(os.path.join(OUT_ROOT, "**", "test_result.csv"), recursive=True):
        try:
            df = pd.read_csv(f)
            row = df.iloc[0].to_dict()
            row["exp"] = os.path.basename(os.path.dirname(f))
            rows.append(row)
        except Exception as e:
            print("[SKIP RESULT]", f, e)

    summary = pd.DataFrame(rows)

    out_dir = os.path.join(OUT_ROOT, "summary")
    os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(
        os.path.join(out_dir, "all_experiment_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if len(summary) == 0:
        print("[NO RESULTS FOUND]")
        return summary

    # Figure 4 mask robustness
    mask_df = summary[summary["exp"].str.contains("mask", case=False, na=False)].copy()
    if len(mask_df):
        mask_df["mask_ratio"] = (
            mask_df["exp"]
            .str.extract(r"mask_([0-9.]+)")[0]
            .astype(float)
        )
        mask_df = mask_df.sort_values("mask_ratio")

        plt.figure(figsize=(7, 5))
        plt.plot(mask_df["mask_ratio"], mask_df["mae"], marker="o")
        plt.xlabel("Mask ratio")
        plt.ylabel("MAE")
        plt.title("Robustness under Partial Observation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig4_mask_ratio_vs_mae.png"), dpi=200)
        plt.close()

    # Figure 5 ablation
    ablation_df = summary[summary["exp"].str.contains("FULL|VANILLA|NO_", case=False, na=False)].copy()
    if len(ablation_df):
        ablation_df = ablation_df.sort_values("mae")
        plt.figure(figsize=(8, 5))
        plt.bar(ablation_df["exp"], ablation_df["mae"])
        plt.xticks(rotation=25)
        plt.ylabel("MAE")
        plt.title("Ablation Study")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig5_ablation_mae.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(ablation_df["exp"], ablation_df["uncertainty_error_corr"])
        plt.xticks(rotation=25)
        plt.ylabel("Corr(|error|, uncertainty)")
        plt.title("Uncertainty Calibration by Ablation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig6_uncertainty_calibration.png"), dpi=200)
        plt.close()

    print("[SUMMARY DONE]", out_dir)
    print(summary)
    return summary


# ============================================================
# Exp 5. Hidden Demand × Model Uncertainty Connection
# ============================================================

def experiment_5_hidden_demand_connection(diag, summary):
    print("\n[EXP5] Hidden Demand Analysis")

    out_dir = os.path.join(OUT_ROOT, "exp5_hidden_demand")
    os.makedirs(out_dir, exist_ok=True)

    top_hidden = diag.sort_values("hidden_demand_score", ascending=False).head(30)
    top_sparse = diag.sort_values("sparsity", ascending=False).head(30)
    top_bias = diag.sort_values("observation_bias_score", ascending=False).head(30)

    top_hidden.to_csv(os.path.join(out_dir, "top_hidden_demand_stations.csv"), index=False, encoding="utf-8-sig")
    top_sparse.to_csv(os.path.join(out_dir, "top_sparse_stations.csv"), index=False, encoding="utf-8-sig")
    top_bias.to_csv(os.path.join(out_dir, "top_observation_bias_stations.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 6))
    plt.scatter(diag["hidden_demand_score"], diag["observation_bias_score"], alpha=0.6)
    plt.xlabel("Hidden demand score")
    plt.ylabel("Observation bias score")
    plt.title("Hidden Demand and Observation Bias")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_hidden_demand_vs_bias.png"), dpi=200)
    plt.close()

    print("[TOP HIDDEN DEMAND]")
    print(top_hidden.head(10))

    print("[EXP5 DONE]", out_dir)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    diag = experiment_1_observation_bias()

    experiment_2_mask_sweep()
    experiment_3_ablation()

    summary = experiment_4_uncertainty_summary()
    experiment_5_hidden_demand_connection(diag, summary)

    print("\nALL CIKM EXPERIMENTS DONE.")