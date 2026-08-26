#!/usr/bin/env python3
"""
compare_llama2_vs_patchtst.py

1. For each date where both Llama2 and PatchTST predictions exist,
   compute average MAE across iterations (itr0/itr1/itr2).
2. Select dates where Llama2 avg MAE < PatchTST avg MAE.
3. Plot each selected date: 13-week forecast window with both models overlaid.

Usage:
  python compare_llama2_vs_patchtst.py
  python compare_llama2_vs_patchtst.py --prediction_dir ./predictions --dataset_dir ./dataset
"""

import matplotlib
matplotlib.use("Agg")

import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# ── dataset / color constants ─────────────────────────────────────────────────
DATASET_DEFS = [
    dict(label="Flu_NorthChina",
         raw="Flu_in_NorthChina(1).csv", raw_fb="Flu_in_NorthChina.csv",
         diff="NorthChina_diff(1).csv",  diff_fb="NorthChina_diff.csv",
         scale=0.01),
]

_DIFF_TO_DEF = {}
for _d in DATASET_DEFS:
    _DIFF_TO_DEF[_d["diff"]]    = _d
    _DIFF_TO_DEF[_d["diff_fb"]] = _d

TRUE_COLOR      = "#333333"
FORECAST_BG_COLOR = "#FCE4D6"
MODEL_COLORS = {
    "Llama2":   "#1f77b4",
    "PatchTST": "#ff7f0e",
}

FILE_RE = re.compile(
    r"flu_north_(Llama2|PatchTST)_pl\d+_(\d{4}-\d{2}-\d{2})_itr(\d+)_predictions\.csv"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve(dataset_dir, preferred, fallback):
    p = dataset_dir / preferred
    if p.exists():
        return p
    fb = dataset_dir / fallback
    if fb.exists():
        return fb
    raise FileNotFoundError(f"Neither '{preferred}' nor '{fallback}' in {dataset_dir}")


def load_raw_series(dataset_dir, ds_def):
    raw_path = _resolve(dataset_dir, ds_def["raw"], ds_def["raw_fb"])
    df = pd.read_csv(raw_path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["positive_rate"]


def inverse_first_and_52week_difference(z_series, raw_ref, forecast_dates):
    raw_ref = raw_ref.copy().sort_index()
    z = np.asarray(z_series, dtype=float)
    forecast_dates = pd.to_datetime(forecast_dates)
    n = len(z)
    y = np.full(n, np.nan)

    def _lookup(d):
        if d in raw_ref.index:
            return float(raw_ref[d])
        before = raw_ref.index[raw_ref.index <= d]
        return float(raw_ref[before[-1]]) if len(before) else float(raw_ref.iloc[0])

    for t in range(n):
        fd  = forecast_dates[t]
        y52 = _lookup(fd - pd.DateOffset(weeks=52))
        y53 = _lookup(fd - pd.DateOffset(weeks=53))
        y1  = _lookup(fd - pd.DateOffset(weeks=1)) if t == 0 else y[t - 1]
        y[t] = z[t] + y1 + y52 - y53
    return y


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ── file scanning ─────────────────────────────────────────────────────────────

def scan_files(pred_dir):
    """Returns {date: {model: [(itr, Path), ...]}}"""
    by_date = defaultdict(lambda: defaultdict(list))
    for f in sorted(pred_dir.glob("*_predictions.csv")):
        m = FILE_RE.match(f.name)
        if m:
            model, date, itr = m.group(1), m.group(2), int(m.group(3))
            by_date[date][model].append((itr, f))
    return by_date


# ── per-date helpers ──────────────────────────────────────────────────────────

def avg_mae_for_date(itr_files, raw_ref, scale):
    """Average MAE across iterations for one model+date."""
    maes = []
    for _itr, fpath in sorted(itr_files):
        df = pd.read_csv(fpath, parse_dates=["date"])
        fdates = df["date"].tolist()
        y_pred = inverse_first_and_52week_difference(
            df["y_pred_diff"].values, raw_ref, fdates) * scale
        y_true = inverse_first_and_52week_difference(
            df["y_true_diff"].values, raw_ref, fdates) * scale
        maes.append(compute_mae(y_true, y_pred))
    return float(np.mean(maes))


def load_avg_pred(itr_files, raw_ref, scale):
    """Average predictions across iterations; returns (fcast_dates, y_true, y_pred_avg)."""
    all_pred, fdates, y_true = [], None, None
    for _itr, fpath in sorted(itr_files):
        df = pd.read_csv(fpath, parse_dates=["date"])
        fdates = df["date"].tolist()
        y_pred = inverse_first_and_52week_difference(
            df["y_pred_diff"].values, raw_ref, fdates) * scale
        y_true = inverse_first_and_52week_difference(
            df["y_true_diff"].values, raw_ref, fdates) * scale
        all_pred.append(y_pred)
    return fdates, y_true, np.mean(all_pred, axis=0)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_date(date, llama2_files, patchtst_files, raw_ref, scale, figure_dir):
    fdates, y_true, llama2_pred  = load_avg_pred(llama2_files,   raw_ref, scale)
    _,      _,      patchtst_pred = load_avg_pred(patchtst_files, raw_ref, scale)

    fcast_dt = pd.to_datetime(fdates)
    cutoff   = fcast_dt[0]
    inp      = raw_ref[raw_ref.index <= cutoff].iloc[-52:] * scale

    l_mae = compute_mae(y_true, llama2_pred)
    p_mae = compute_mae(y_true, patchtst_pred)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(inp.index, inp.values,   color=TRUE_COLOR,              linewidth=1.8, label="Observed (history)")
    ax.plot(fcast_dt,  y_true,       color=TRUE_COLOR,              linewidth=1.8, label="Observed (forecast window)")
    ax.plot(fcast_dt,  llama2_pred,  color=MODEL_COLORS["Llama2"],  linewidth=2.0,
            linestyle="--", label=f"Llama2   MAE={l_mae:.4f}")
    ax.plot(fcast_dt,  patchtst_pred, color=MODEL_COLORS["PatchTST"], linewidth=2.0,
            linestyle="--", label=f"PatchTST MAE={p_mae:.4f}")

    ax.axvspan(fcast_dt[0], fcast_dt[-1], color=FORECAST_BG_COLOR, alpha=0.5, zorder=0)

    y_all = np.concatenate([inp.values, y_true, llama2_pred, patchtst_pred])
    pad   = (y_all.max() - y_all.min()) * 0.08 or 0.005
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25); lbl.set_ha("right"); lbl.set_fontsize(9)

    ax.set_ylabel("Positive rate", fontsize=11)
    ax.set_title(
        f"Flu NorthChina — forecast ending on {date}  "
        f"(Llama2 MAE={l_mae:.4f} < PatchTST MAE={p_mae:.4f})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = figure_dir / f"{date}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.name}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare Llama2 vs PatchTST and plot Llama2 wins")
    p.add_argument("--prediction_dir", default="./predictions", type=Path)
    p.add_argument("--dataset_dir",    default="./dataset",    type=Path)
    p.add_argument("--figure_dir",     default="./figures", type=Path)
    cli = p.parse_args()

    cli.figure_dir.mkdir(parents=True, exist_ok=True)

    ds_def  = _DIFF_TO_DEF["NorthChina_diff.csv"]
    scale   = ds_def["scale"]
    raw_ref = load_raw_series(cli.dataset_dir, ds_def)

    by_date = scan_files(cli.prediction_dir)
    common  = sorted(
        d for d, models in by_date.items()
        if "Llama2" in models and "PatchTST" in models
    )
    print(f"Common dates (both models present): {len(common)}\n")

    wins = []
    for date in common:
        l_mae = avg_mae_for_date(by_date[date]["Llama2"],   raw_ref, scale)
        p_mae = avg_mae_for_date(by_date[date]["PatchTST"], raw_ref, scale)
        tag   = "*** LLAMA2 WINS ***" if l_mae < p_mae else "patchtst wins"
        print(f"  {date}  Llama2={l_mae:.4f}  PatchTST={p_mae:.4f}  {tag}")
        if l_mae < p_mae:
            wins.append(date)

    print(f"\nLlama2 wins: {len(wins)} / {len(common)} common dates\n")

    if not wins:
        print("No dates where Llama2 MAE < PatchTST MAE.")
        return

    print("Generating plots ...")
    for date in wins:
        plot_date(
            date,
            by_date[date]["Llama2"],
            by_date[date]["PatchTST"],
            raw_ref, scale,
            cli.figure_dir,
        )

    print(f"\nDone. {len(wins)} figures saved to: {cli.figure_dir}/")


if __name__ == "__main__":
    main()
