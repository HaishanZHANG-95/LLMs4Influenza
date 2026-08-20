#!/usr/bin/env python3
"""
run_gpt4ts_and_plot.py — multi-model influenza forecast comparison plots.

Reads prediction CSVs saved by main.py (./predictions/*.csv) and produces
Figure 2-style overlaid comparison plots for all models and datasets.

Usage:
  python run_gpt4ts_and_plot.py
  python run_gpt4ts_and_plot.py --prediction_dir ./predictions --figure_dir ./figures
  python run_gpt4ts_and_plot.py --dataset_dir /path/to/data
"""

import matplotlib
matplotlib.use("Agg")

import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from utils.tools import plot_gpt4ts_inset

# ── dataset definitions: maps diff-csv filename → raw csv + display scale ────
DATASET_DEFS = [
    dict(label="Flu_NorthChina",
         raw="Flu_in_NorthChina(1).csv", raw_fb="Flu_in_NorthChina.csv",
         diff="NorthChina_diff(1).csv",  diff_fb="NorthChina_diff.csv",
         scale=0.01),
    dict(label="Flu_SouthChina",
         raw="Flu_in_SouthChina(1).csv", raw_fb="Flu_in_SouthChina.csv",
         diff="SouthChina_diff(1).csv",  diff_fb="SouthChina_diff.csv",
         scale=0.01),
    dict(label="Flu_USA",
         raw="Flu_in_USA(1).csv",        raw_fb="Flu_in_USA.csv",
         diff="USA_diff(1).csv",         diff_fb="USA_diff.csv",
         scale=0.01),
    dict(label="ILI_NorthChina",
         raw="ILI_in_NorthChina(1).csv", raw_fb="ILI_in_NorthChina.csv",
         diff="ILI_NorthChina_diff(1).csv", diff_fb="ILI_NorthChina_diff.csv",
         scale=1.0),
    dict(label="ILI_SouthChina",
         raw="ILI_in_SouthChina(1).csv", raw_fb="ILI_in_SouthChina.csv",
         diff="ILI_SouthChina_diff(1).csv", diff_fb="ILI_SouthChina_diff.csv",
         scale=1.0),
]

# diff filename (with or without "(1)") → dataset def
_DIFF_TO_DEF = {}
# label → dataset def
_LABEL_TO_DEF = {}
for _d in DATASET_DEFS:
    _DIFF_TO_DEF[_d["diff"]]    = _d
    _DIFF_TO_DEF[_d["diff_fb"]] = _d
    _LABEL_TO_DEF[_d["label"]]  = _d

# color palette
GROUNDTRUTH_COLOR  = "#333333"
GPT2_COLOR         = "#7B3294"
LLAMA2_COLOR       = "#D7191C"
FORECAST_BG_COLOR  = "#FCE4D6"
TRUE_COLOR         = GROUNDTRUTH_COLOR

MODEL_COLORS = {
    "GPT4TS":  GPT2_COLOR,
    "Llama2":  LLAMA2_COLOR,
    "Llama3":  "#E87040",
    "Gemma2":  "#27AE60",
    "PatchTST":"#1A9641",
    "DLinear": "#F39C12",
}
_FALLBACK_COLORS = ["#3498DB", "#1ABC9C", "#E67E22", "#95A5A6"]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

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


def _model_color(model_name, used):
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    for c in _FALLBACK_COLORS:
        if c not in used:
            return c
    return "#888888"


# ════════════════════════════════════════════════════════════════════════════
# Inverse differencing
# ════════════════════════════════════════════════════════════════════════════

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


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    sp,_ = stats.spearmanr(y_true, y_pred)
    pe,_ = stats.pearsonr(y_true, y_pred)
    return dict(mae=mae, mse=mse, spearman=float(sp), pearson=float(pe))


# ════════════════════════════════════════════════════════════════════════════
# Load predictions saved by main.py
# ════════════════════════════════════════════════════════════════════════════

def load_all_predictions(pred_dir, dataset_dir, seq_len=52):
    """
    Scan pred_dir for *_predictions.csv files saved by main.py.
    Returns dict: dataset_label -> {
        "input_dates", "input_values", "forecast_dates", "y_true_orig",
        "models": [ {model_name, color, y_pred_orig, metrics} ]
    }
    If multiple itr CSVs exist for the same model+dataset, the last one is used.
    """
    csvs = sorted(Path(pred_dir).glob("*_predictions.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_predictions.csv files found in {pred_dir}")

    # normalise each CSV into (model_name, label, fcast_dates, y_pred_orig, y_true_orig, ds_def)
    # keep last file when duplicate (model_name, label) keys appear
    grouped = {}
    for csv_path in csvs:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        cols = set(df.columns)

        if "model_name" in cols and "data_path" in cols:
            # new format saved by main.py
            model_name = df["model_name"].iloc[0]
            data_path  = df["data_path"].iloc[0]
            ds_def     = _DIFF_TO_DEF.get(data_path)
            if ds_def is None:
                print(f"  [skip] unknown data_path '{data_path}' in {csv_path.name}")
                continue
            grouped[(model_name, ds_def["label"])] = ("new", df, ds_def)

        elif "dataset_id" in cols:
            # old format saved by run_gpt4ts_and_plot.py
            label = df["dataset_id"].iloc[0]
            ds_def = _LABEL_TO_DEF.get(label)
            if ds_def is None:
                print(f"  [skip] unknown dataset_id '{label}' in {csv_path.name}")
                continue
            # infer model name from filename: {label}_{ModelName}_predictions.csv
            stem = csv_path.stem  # e.g. Flu_NorthChina_GPT4TS_predictions -> wrong, stem has no _predictions
            # stem is e.g. "Flu_NorthChina_GPT4TS_predictions"
            suffix = "_predictions"
            base = stem[:-len(suffix)] if stem.endswith(suffix) else stem
            model_name = base[len(label) + 1:] if base.startswith(label + "_") else base
            grouped[(model_name, label)] = ("old", df, ds_def)

        else:
            print(f"  [skip] unrecognised CSV format: {csv_path.name}")

    # build per-dataset result dict
    by_dataset = {}
    used_colors = set()
    for (model_name, label), (fmt, df, ds_def) in grouped.items():
        scale = ds_def.get("scale", 1.0)
        fcast_dates = df["date"].tolist()

        if fmt == "new":
            raw_ref = load_raw_series(Path(dataset_dir), ds_def)
            y_pred_orig = inverse_first_and_52week_difference(
                df["y_pred_diff"].values, raw_ref, fcast_dates) * scale
            y_true_orig = inverse_first_and_52week_difference(
                df["y_true_diff"].values, raw_ref, fcast_dates) * scale
        else:
            # old format already has original-scale values
            raw_ref     = load_raw_series(Path(dataset_dir), ds_def)
            y_pred_orig = df["y_pred_original"].values
            y_true_orig = df["y_true_original"].values

        metrics = compute_metrics(y_true_orig, y_pred_orig)
        color   = _model_color(model_name, used_colors)
        used_colors.add(color)

        if label not in by_dataset:
            cutoff = pd.to_datetime(fcast_dates[0])
            inp    = raw_ref[raw_ref.index < cutoff].iloc[-seq_len:]
            by_dataset[label] = dict(
                label         = label,
                forecast_dates= [str(d.date()) for d in pd.to_datetime(fcast_dates)],
                input_dates   = list(inp.index),
                input_values  = inp.values * scale,
                y_true_orig   = y_true_orig,
                models        = [],
            )

        by_dataset[label]["models"].append(dict(
            model_name  = model_name,
            color       = color,
            y_pred_orig = y_pred_orig,
            metrics     = metrics,
        ))
        print(f"  [loaded] {label} / {model_name}  "
              f"MAE={metrics['mae']:.4f}  Pearson={metrics['pearson']:.4f}")

    return by_dataset


# ════════════════════════════════════════════════════════════════════════════
# Plot: single dataset, all models overlaid
# ════════════════════════════════════════════════════════════════════════════

def plot_single_dataset(ds_result, figure_dir):
    label        = ds_result["label"]
    input_dates  = pd.to_datetime(ds_result["input_dates"])
    input_values = ds_result["input_values"]
    fcast_dates  = pd.to_datetime(ds_result["forecast_dates"])
    y_true       = ds_result["y_true_orig"]
    model_list   = ds_result["models"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [2, 1.2]})
    fig.subplots_adjust(hspace=0.38)

    # top panel: historical + all model forecasts
    ax1.plot(input_dates, input_values, color=TRUE_COLOR, linewidth=1.8, label="Observed")
    ax1.plot(fcast_dates, y_true, color=TRUE_COLOR, linewidth=1.8)
    for m in model_list:
        ax1.plot(fcast_dates, m["y_pred_orig"], color=m["color"], linewidth=2.0,
                 linestyle="--", label=f"{m['model_name']} (MAE={m['metrics']['mae']:.3f})")
    ax1.axvspan(fcast_dates[0], fcast_dates[-1], color=FORECAST_BG_COLOR, alpha=0.5, zorder=0)

    y_all = np.concatenate([input_values, y_true] + [m["y_pred_orig"] for m in model_list])
    pad = (y_all.max() - y_all.min()) * 0.08 or 0.05
    ax1.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax1.set_xlim(input_dates[0] - pd.Timedelta(days=5),
                 fcast_dates[-1] + pd.Timedelta(days=5))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(25); lbl.set_ha("right"); lbl.set_fontsize(9)
    ax1.set_ylabel("Positive rate / ILI count", fontsize=11)
    ax1.set_title(label, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # bottom panel: zoomed 13-week window, best-MAE model annotated
    best_m = min(model_list, key=lambda m: m["metrics"]["mae"])
    plot_gpt4ts_inset(ax2, fcast_dates, y_true, best_m["y_pred_orig"],
                      model_name=best_m["model_name"], model_color=best_m["color"],
                      true_color=TRUE_COLOR, mae=best_m["metrics"]["mae"])
    for m in model_list:
        if m is not best_m:
            ax2.plot(fcast_dates, m["y_pred_orig"], color=m["color"],
                     linewidth=1.8, linestyle="--", label=m["model_name"])
    ax2.set_title("13-week forecast window", fontsize=11)
    ax2.tick_params(axis="y", left=True, labelleft=True)
    ax2.legend(fontsize=9, loc="upper left")

    for suffix in (".png", ".pdf"):
        out = Path(figure_dir) / f"{label}_multimodel{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  [figure] {out.name}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Plot: combined N-row summary (one row per dataset)
# ════════════════════════════════════════════════════════════════════════════

def plot_combined_summary(by_dataset, figure_dir):
    datasets = list(by_dataset.values())
    n = len(datasets)
    fig, axes = plt.subplots(n, 2, figsize=(18, n * 4),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.5, wspace=0.06)

    for row, ds in enumerate(datasets):
        ax_left  = axes[row][0]
        ax_right = axes[row][1]
        input_dates  = pd.to_datetime(ds["input_dates"])
        input_values = ds["input_values"]
        fcast_dates  = pd.to_datetime(ds["forecast_dates"])
        y_true       = ds["y_true_orig"]

        ax_left.plot(input_dates, input_values, color=TRUE_COLOR, linewidth=1.6)
        ax_left.plot(fcast_dates, y_true, color=TRUE_COLOR, linewidth=1.6)
        for m in ds["models"]:
            ax_left.plot(fcast_dates, m["y_pred_orig"], color=m["color"],
                         linewidth=2.0, linestyle="--",
                         label=f"{m['model_name']}  r={m['metrics']['pearson']:.2f}")
        ax_left.axvspan(fcast_dates[0], fcast_dates[-1], color=FORECAST_BG_COLOR, alpha=0.5, zorder=0)
        y_all = np.concatenate([input_values, y_true] + [m["y_pred_orig"] for m in ds["models"]])
        pad = (y_all.max() - y_all.min()) * 0.08 or 0.05
        ax_left.set_ylim(y_all.min() - pad, y_all.max() + pad)
        ax_left.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl in ax_left.get_xticklabels():
            lbl.set_rotation(20); lbl.set_ha("right"); lbl.set_fontsize(8)
        ax_left.set_ylabel(ds["label"], fontsize=10, fontweight="bold")
        ax_left.spines["top"].set_visible(False); ax_left.spines["right"].set_visible(False)
        ax_left.legend(fontsize=8, loc="upper left")

        best_m = min(ds["models"], key=lambda m: m["metrics"]["mae"])
        plot_gpt4ts_inset(ax_right, fcast_dates, y_true, best_m["y_pred_orig"],
                          model_name=best_m["model_name"], model_color=best_m["color"],
                          true_color=TRUE_COLOR, mae=best_m["metrics"]["mae"])
        for m in ds["models"]:
            if m is not best_m:
                ax_right.plot(fcast_dates, m["y_pred_orig"], color=m["color"],
                              linewidth=1.6, linestyle="--")

    axes[0][0].set_title("Observed + model forecasts", fontsize=12, fontweight="bold")
    axes[0][1].set_title("13-week zoom", fontsize=12, fontweight="bold")

    for suffix in (".png", ".pdf"):
        out = Path(figure_dir) / f"all_datasets_multimodel{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  [figure] {out.name}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Multi-model influenza forecast plots")
    p.add_argument("--prediction_dir", default="./predictions", type=Path,
                   help="Directory with *_predictions.csv files saved by main.py")
    p.add_argument("--dataset_dir",    default="./dataset",    type=Path,
                   help="Directory with raw CSV files for inverse differencing")
    p.add_argument("--figure_dir",     default="./figures",    type=Path)
    p.add_argument("--seq_len",        default=52, type=int,
                   help="Input window length (must match the value used in main.py)")
    cli = p.parse_args()

    cli.figure_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading prediction CSVs ...")
    by_dataset = load_all_predictions(cli.prediction_dir, cli.dataset_dir, cli.seq_len)

    if not by_dataset:
        print("No predictions loaded — nothing to plot.")
        return

    print(f"\n[2] Generating figures for {len(by_dataset)} dataset(s) ...")
    for ds in by_dataset.values():
        plot_single_dataset(ds, cli.figure_dir)
    plot_combined_summary(by_dataset, cli.figure_dir)

    print(f"\nAll figures saved to: {cli.figure_dir}/")


if __name__ == "__main__":
    main()

