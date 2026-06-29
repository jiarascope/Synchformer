#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


FLOAT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def extract_metric(line: str, name: str):
    """
    Extracts:
      Metric: current (running_avg)
      Metric: current

    Returns:
      current, running_avg
    """
    pattern = rf"\b{re.escape(name)}\s*:\s*({FLOAT})(?:\s*\(({FLOAT})\))?"
    m = re.search(pattern, line)
    if not m:
        return None, None

    cur = float(m.group(1))
    avg = float(m.group(2)) if m.group(2) is not None else None
    return cur, avg


def parse_log(log_path: Path) -> pd.DataFrame:
    metric_names = [
        "Loss",
        "Segment_contrastive_loss",
        "Precision",
        "Precision_a",
        "Precision_v",
        "Accuracy",
        "Acc",
        "LR",
    ]

    rows = []

    progress_re = re.compile(
        rf"\|\s+INFO\s+\|\s+"
        rf"(?P<phase>Train|Valid|Val|Eval|Test)"
        rf".*?Epoch:\s*(?P<epoch>\d+)"
        rf".*?\[\s*(?P<seen>\d+)\s*/\s*(?P<total>\d+)"
    )

    with log_path.open("r", errors="replace") as f:
        for line in f:
            m = progress_re.search(line)
            if not m:
                continue

            phase = m.group("phase")
            if phase in {"Valid", "Val"}:
                phase = "Eval"

            row = {
                "phase": phase,
                "epoch": int(m.group("epoch")),
                "seen_in_phase": int(m.group("seen")),
                "phase_total": int(m.group("total")),
            }

            # This is good for the train trace.
            row["phase_progress"] = row["epoch"] * row["phase_total"] + row["seen_in_phase"]

            found = False
            for name in metric_names:
                cur, avg = extract_metric(line, name)
                if cur is not None:
                    key = name.lower()
                    row[key] = cur
                    if avg is not None:
                        row[f"{key}_avg"] = avg
                    found = True

            if found:
                rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates()

    # Prefer running-average metrics when available.
    for metric in [
        "loss",
        "segment_contrastive_loss",
        "precision",
        "precision_a",
        "precision_v",
        "accuracy",
        "acc",
    ]:
        avg_col = f"{metric}_avg"
        if avg_col in df.columns:
            df[f"{metric}_plot"] = df[avg_col]
        elif metric in df.columns:
            df[f"{metric}_plot"] = df[metric]

    return df


def epoch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the last logged point for each phase/epoch.
    This removes the noisy mid-validation running-average points.
    """
    if df.empty:
        return df

    df = df.sort_values(["phase", "epoch", "seen_in_phase"])
    return df.groupby(["phase", "epoch"], as_index=False).tail(1)


def save_plot(out_path: Path):
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def plot_loss_by_epoch(summary: pd.DataFrame, out_dir: Path):
    metric = "loss_plot"
    if metric not in summary.columns:
        print("[skip] loss not found")
        return

    plt.figure(figsize=(8, 5))

    for phase in ["Train", "Eval"]:
        d = summary[summary["phase"] == phase].dropna(subset=[metric])
        if not d.empty:
            plt.plot(d["epoch"], d[metric], marker="o", linewidth=2, label=phase)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss by epoch")
    plt.legend()
    save_plot(out_dir / "loss_by_epoch.png")


def plot_precision_by_epoch(summary: pd.DataFrame, out_dir: Path):
    metric = "precision_plot"
    if metric not in summary.columns:
        print("[skip] precision not found")
        return

    plt.figure(figsize=(8, 5))

    for phase in ["Train", "Eval"]:
        d = summary[summary["phase"] == phase].dropna(subset=[metric])
        if not d.empty:
            plt.plot(d["epoch"], d[metric], marker="o", linewidth=2, label=phase)

    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision by epoch")
    plt.ylim(0, 1.05)
    plt.legend()
    save_plot(out_dir / "precision_by_epoch.png")


def plot_train_loss_trace(df: pd.DataFrame, out_dir: Path):
    metric = "loss_plot"
    d = df[(df["phase"] == "Train") & df[metric].notna()].copy()

    if d.empty:
        print("[skip] train loss trace not found")
        return

    # Better global x-axis for train.
    train_total = d["phase_total"].max()
    d["global_train_seen"] = d["epoch"] * train_total + d["seen_in_phase"]

    plt.figure(figsize=(9, 5))
    plt.plot(d["global_train_seen"], d[metric], linewidth=1.5)
    plt.xlabel("Global training samples seen")
    plt.ylabel("Training loss")
    plt.title("Training loss trace")
    save_plot(out_dir / "train_loss_trace.png")


def plot_lr(df: pd.DataFrame, out_dir: Path):
    if "lr" not in df.columns:
        print("[skip] lr not found")
        return

    d = df[(df["phase"] == "Train") & df["lr"].notna()].copy()
    if d.empty:
        print("[skip] train lr not found")
        return

    train_total = d["phase_total"].max()
    d["global_train_seen"] = d["epoch"] * train_total + d["seen_in_phase"]

    plt.figure(figsize=(9, 5))
    plt.plot(d["global_train_seen"], d["lr"], linewidth=1.5)
    plt.xlabel("Global training samples seen")
    plt.ylabel("Learning rate")
    plt.title("Learning rate schedule")
    save_plot(out_dir / "lr_schedule.png")


def plot_directional_precision(summary: pd.DataFrame, out_dir: Path):
    needed = ["precision_a_plot", "precision_v_plot"]
    if not all(c in summary.columns for c in needed):
        print("[skip] directional precision not found")
        return

    d = summary[summary["phase"] == "Eval"].copy()
    if d.empty:
        print("[skip] no eval directional precision")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(d["epoch"], d["precision_a_plot"], marker="o", linewidth=2, label="Precision_a")
    plt.plot(d["epoch"], d["precision_v_plot"], marker="o", linewidth=2, label="Precision_v")
    plt.xlabel("Epoch")
    plt.ylabel("Eval precision")
    plt.title("Eval directional precision")
    plt.ylim(0, 1.05)
    plt.legend()
    save_plot(out_dir / "eval_directional_precision.png")


def print_loss_range(df: pd.DataFrame):
    print("\nLoss ranges from parsed log:")

    for phase in ["Train", "Eval"]:
        d = df[df["phase"] == phase]

        if "loss_plot" in d.columns and d["loss_plot"].notna().any():
            print(
                f"  {phase} loss_avg: "
                f"min={d['loss_plot'].min():.4f}, "
                f"max={d['loss_plot'].max():.4f}, "
                f"last={d['loss_plot'].dropna().iloc[-1]:.4f}"
            )

        if "segment_contrastive_loss_plot" in d.columns and d["segment_contrastive_loss_plot"].notna().any():
            print(
                f"  {phase} segment_contrastive_loss_avg: "
                f"min={d['segment_contrastive_loss_plot'].min():.4f}, "
                f"max={d['segment_contrastive_loss_plot'].max():.4f}, "
                f"last={d['segment_contrastive_loss_plot'].dropna().iloc[-1]:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("essential_plots"))

    parser.add_argument(
        "--include-lr",
        action="store_true",
        help="Also output lr_schedule.png",
    )
    parser.add_argument(
        "--include-directions",
        action="store_true",
        help="Also output eval_directional_precision.png",
    )

    args = parser.parse_args()

    df = parse_log(args.log_file)
    if df.empty:
        raise SystemExit("No metric rows found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.out_dir / "metrics_all.csv", index=False)

    summary = epoch_summary(df)
    summary.to_csv(args.out_dir / "metrics_epoch_summary.csv", index=False)

    # Essential plots only.
    plot_loss_by_epoch(summary, args.out_dir)
    plot_precision_by_epoch(summary, args.out_dir)
    plot_train_loss_trace(df, args.out_dir)

    # Optional plots.
    if args.include_lr:
        plot_lr(df, args.out_dir)

    if args.include_directions:
        plot_directional_precision(summary, args.out_dir)

    print_loss_range(df)


if __name__ == "__main__":
    main()