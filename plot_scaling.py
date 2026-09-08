#!/usr/bin/env python3
"""Plot measured strong scaling from cluster/run_scaling.py CSV output."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (args.results / "raw.csv").open() as file:
        rows = list(csv.DictReader(file))
    groups = {}
    problems = set()
    for row in rows:
        p, time = int(row["ranks"]), float(row["seconds"])
        if not math.isfinite(time) or time <= 0:
            raise ValueError("Invalid timing")
        problems.add(tuple(row[key] for key in ["nx", "ny", "steps", "warmup"]))
        groups.setdefault(p, []).append(time)
    if len(problems) != 1 or 1 not in groups:
        raise ValueError("Require one fixed problem and a one-process baseline")
    meta = json.loads((args.results / "metadata.json").read_text())
    if set(groups) != set(meta["ranks"]) or any(len(v) != meta["repeats"] for v in groups.values()):
        raise ValueError("Incomplete scaling run")
    processes = sorted(groups)
    baseline = statistics.median(groups[1])
    summary = []
    for p in processes:
        times = groups[p]
        median = statistics.median(times)
        summary.append(dict(ranks=p, median_seconds=median, min_seconds=min(times),
                            max_seconds=max(times), speedup=baseline/median,
                            efficiency=baseline/median/p))
    with (args.results / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    speedup = [r["speedup"] for r in summary]
    lower = [r["speedup"] - baseline/r["max_seconds"] for r in summary]
    upper = [baseline/r["min_seconds"] - r["speedup"] for r in summary]
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    ax.plot(processes, processes, "--", color="0.55", label="Ideal: S(p) = p")
    ax.errorbar(processes, speedup, yerr=[lower, upper], fmt="o-", capsize=4,
                color="#1565c0", label="Measured median; bars: runtime range")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(processes, labels=[str(p) for p in processes])
    ax.set_yticks(processes, labels=[str(p) for p in processes])
    if meta.get("backend") == "cuda":
        unit = f"MPI ranks (one {meta['gpu_arch'].upper()} GPU per rank)"
    else:
        unit = "MPI processes (one CPU core per process)"
    ax.set_xlabel(unit)
    ax.set_ylabel("Speedup T(1) / T(p)")
    ax.set_title(f"bwUniCluster: {meta['nx']} × {meta['ny']}, {meta['steps']:,} timesteps")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(args.results / "strong_scaling.png", dpi=180)
    fig.savefig(args.results / "strong_scaling.svg")
    fig.savefig(args.results / "strong_scaling.pdf")
    print("ranks  median_seconds  speedup  efficiency")
    for r in summary:
        print(f"{r['ranks']:5}  {r['median_seconds']:14.6f}  {r['speedup']:7.3f}  {r['efficiency']:9.2%}")


if __name__ == "__main__":
    main()
