#!/usr/bin/env python3
"""Overlay strong-scaling speedup across problem sizes on one log-log plot.

Complements plot_scaling.py (which plots one problem size). Reading multiple
cluster/run_scaling.py `size_NxN` subdirectories on the same axes is what
shows the communication-to-computation ratio effect called out in
https://pastewka.github.io/Accelerators/notes/scaling.html : at a fixed rank
count, a small grid gives each rank fewer owned columns relative to the two
halo columns it exchanges every step, so its speedup should sit further
below the ideal line than a large grid's does.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def load_size(size_dir):
    with (size_dir / "raw.csv").open() as file:
        rows = list(csv.DictReader(file))
    groups = {}
    for row in rows:
        p, time = int(row["ranks"]), float(row["seconds"])
        if not math.isfinite(time) or time <= 0:
            raise ValueError(f"Invalid timing in {size_dir}")
        groups.setdefault(p, []).append(time)
    meta = json.loads((size_dir / "metadata.json").read_text())
    if set(groups) != set(meta["ranks"]) or any(len(v) != meta["repeats"] for v in groups.values()):
        raise ValueError(f"Incomplete scaling run in {size_dir}")
    processes = sorted(groups)
    baseline = statistics.median(groups[1])
    speedup = [baseline / statistics.median(groups[p]) for p in processes]
    efficiency = [s / p for s, p in zip(speedup, processes)]
    return meta, processes, speedup, efficiency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path,
                        help="Directory containing size_NxN subdirectories, e.g. milestone6_results/scaling/JOBID")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <results_root>/strong_scaling_multisize.{png,svg,pdf}")
    args = parser.parse_args()

    size_dirs = sorted((d for d in args.results_root.glob("size_*") if d.is_dir()),
                       key=lambda d: int(d.name.split("_")[1].split("x")[0]))
    if not size_dirs:
        raise FileNotFoundError(f"No size_NxN subdirectories under {args.results_root}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })

    # Viridis, sampled across the number of problem sizes: small grids (most
    # affected by communication overhead) are dark purple, large grids yellow.
    cmap = plt.get_cmap("viridis")
    all_processes = set()
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    # Second figure for efficiency, per https://pastewka.github.io/Accelerators/notes/scaling.html :
    # log2 processes on x, efficiency as a 0-100% linear percentage on y, ideal
    # is a flat line at 100% rather than the speedup plot's diagonal.
    fig_eff, ax_eff = plt.subplots(figsize=(8, 6), layout="constrained")
    for index, size_dir in enumerate(size_dirs):
        meta, processes, speedup, efficiency = load_size(size_dir)
        all_processes.update(processes)
        color = cmap(index / max(len(size_dirs) - 1, 1))
        label = f"{meta['nx']} x {meta['ny']}"
        ax.plot(processes, speedup, "o-", color=color, linewidth=2, markersize=7, label=label)
        ax_eff.plot(processes, efficiency, "o-", color=color, linewidth=2, markersize=7, label=label)

    ideal = sorted(all_processes)
    ax.plot(ideal, ideal, "--", color="0.55", linewidth=2, label="Ideal: S(p) = p", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ideal, labels=[str(p) for p in ideal])
    ax.set_yticks(ideal, labels=[str(p) for p in ideal])
    ax.set_xlabel("MPI processes (one CPU core per process)")
    ax.set_ylabel("Speedup T(1) / T(p)")
    ax.set_title("bwUniCluster strong scaling: speedup")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper left", title="Grid size")

    ax_eff.axhline(1.0, linestyle="--", color="0.55", linewidth=2, label="Ideal: 100% efficiency", zorder=0)
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_xticks(ideal, labels=[str(p) for p in ideal])
    ax_eff.set_ylim(0, 1.1)
    ax_eff.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax_eff.set_xlabel("MPI processes (one CPU core per process)")
    ax_eff.set_ylabel("Efficiency = Speedup / p")
    ax_eff.set_title("bwUniCluster strong scaling: efficiency")
    ax_eff.grid(True, which="both", alpha=0.2)
    ax_eff.legend(loc="lower left", title="Grid size")

    stem = args.output or (args.results_root / "strong_scaling_multisize")
    eff_stem = stem.with_name(stem.name.replace("strong_scaling", "efficiency"))
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None)
        fig_eff.savefig(eff_stem.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None)
    print(f"Wrote {stem}.{{png,svg,pdf}} and {eff_stem}.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
