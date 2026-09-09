#!/usr/bin/env python3
"""Plot raw GPU MLUPS for the Milestone 6 distributed lid-driven cavity, per
https://pastewka.github.io/Accelerators/project/milestone06.html : total
lattice updates per second across the whole distributed grid (one A100 GPU
per MPI rank), as a function of GPU count.

Unlike plot_mlups_scaling.py (which derives speedup/efficiency relative to
the 1-GPU run), this reports the MLUPS numbers themselves, since that is the
throughput metric the milestone asks to measure and report.

Reads milestone6_results/mlups_report.csv (median/min/max per configuration,
already summarized from milestone6_results/scaling/a100-6822867/raw.csv).
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("milestone6_results/mlups_report.csv"))
    parser.add_argument("--output", type=Path, default=Path("milestone6_results/gpu_mlups"))
    args = parser.parse_args()

    with args.csv.open() as file:
        rows = [row for row in csv.DictReader(file) if row["backend"] == "gpu"]
    if not rows:
        raise ValueError(f"No GPU rows found in {args.csv}")
    rows.sort(key=lambda r: int(r["ranks"]))

    gpus = [int(r["ranks"]) for r in rows]
    median = [float(r["median_mlups"]) for r in rows]
    lower = [float(r["median_mlups"]) - float(r["min_mlups"]) for r in rows]
    upper = [float(r["max_mlups"]) - float(r["median_mlups"]) for r in rows]
    grid = f"{rows[0]['nx']}x{rows[0]['ny']}"
    steps = rows[0]["steps"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    color = "#eb6834"  # matches the GPU series color in plot_mlups_scaling.py

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    ax.errorbar(gpus, median, yerr=[lower, upper], fmt="o-", capsize=5,
                linewidth=2.4, markersize=9, color=color,
                label="Measured median; bars: run-to-run range")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus, labels=[str(g) for g in gpus])
    ax.set_xlabel("A100 GPUs (one MPI rank each)")
    ax.set_ylabel("MLUPS")
    ax.set_title(f"Milestone 6: GPU MLUPS, {grid} grid, {steps} steps")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")
    ax.set_ylim(top=max(median) * 1.15)

    for g, m in zip(gpus, median):
        ax.annotate(f"{m:,.0f}", xy=(g, m), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=12, color="0.25")

    for suffix in ["png", "svg"]:
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None,
                    bbox_inches="tight")
    print(f"Wrote {args.output}.{{png,svg}}")
    print("gpus  median_mlups  min_mlups     max_mlups")
    for r in rows:
        print(f"{r['ranks']:>4}  {float(r['median_mlups']):12.4f}  "
              f"{float(r['min_mlups']):11.4f}  {float(r['max_mlups']):11.4f}")


if __name__ == "__main__":
    main()
