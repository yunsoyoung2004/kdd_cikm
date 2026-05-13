import os
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOCAL_DIR = "./hf_data"
SCRIPT = "bong.py"
OUT_ROOT = "./cikm_outputs_safe"

MAX_SUBWAY_FILES = 20
MAX_CELL_FILES = 50
EPOCHS = 10
BATCH_SIZE = 4

os.makedirs(OUT_ROOT, exist_ok=True)


def run(cmd):
    print("\n" + "=" * 100)
    print(cmd)
    print("=" * 100)
    subprocess.run(cmd, shell=True, check=False)


def run_mask_sweep():
    for mr in [0.1, 0.3, 0.5, 0.7]:
        out = os.path.join(OUT_ROOT, f"mask_{mr}")
        cmd = f"""
python {SCRIPT} \
  --skip_download \
  --local_dir "{LOCAL_DIR}" \
  --output_dir "{out}" \
  --max_subway_files {MAX_SUBWAY_FILES} \
  --max_cell_files {MAX_CELL_FILES} \
  --window 12 \
  --horizon 1 \
  --mask_ratio {mr} \
  --batch_size {BATCH_SIZE} \
  --epochs {EPOCHS} \
  --num_workers 0
"""
        run(cmd)


def run_ablation():
    settings = {
        "FULL": (1.0, 0.1),
        "NO_RECON": (0.0, 0.1),
        "NO_UNCERTAINTY": (1.0, 0.0),
        "VANILLA": (0.0, 0.0),
    }

    for name, (lr, lu) in settings.items():
        out = os.path.join(OUT_ROOT, name)
        cmd = f"""
python {SCRIPT} \
  --skip_download \
  --local_dir "{LOCAL_DIR}" \
  --output_dir "{out}" \
  --max_subway_files {MAX_SUBWAY_FILES} \
  --max_cell_files {MAX_CELL_FILES} \
  --window 12 \
  --horizon 1 \
  --mask_ratio 0.3 \
  --batch_size {BATCH_SIZE} \
  --epochs {EPOCHS} \
  --lambda_recon {lr} \
  --lambda_unc {lu} \
  --num_workers 0
"""
        run(cmd)


def collect_results():
    rows = []

    for f in glob.glob(os.path.join(OUT_ROOT, "**", "test_result.csv"), recursive=True):
        try:
            df = pd.read_csv(f)
            row = df.iloc[0].to_dict()
            row["exp"] = os.path.basename(os.path.dirname(f))
            rows.append(row)
        except Exception as e:
            print("[SKIP]", f, e)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_ROOT, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary)
    print("[SAVED]", summary_path)

    if len(summary) > 0:
        plt.figure(figsize=(8, 5))
        summary.sort_values("mae").plot(
            x="exp",
            y="mae",
            kind="bar",
            legend=False,
        )
        plt.ylabel("MAE")
        plt.title("POUR CIKM Experiments")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, "summary_mae.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        summary.sort_values("uncertainty_error_corr").plot(
            x="exp",
            y="uncertainty_error_corr",
            kind="bar",
            legend=False,
        )
        plt.ylabel("Corr(|error|, uncertainty)")
        plt.title("Uncertainty Calibration")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, "summary_uncertainty_corr.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    run_mask_sweep()
    run_ablation()
    collect_results()