#!/usr/bin/env python3
"""Plot CPU/GPU speedup and efficiency derived from MLUPS, per
https://pastewka.github.io/Accelerators/notes/scaling.html : speedup on
log-log axes against an ideal S(p) = p line, efficiency as its own figure
on a linear 0-100% axis against a flat 100% ideal line.

Speedup S(p) = MLUPS(p) / MLUPS(1) is equivalent to the usual T(1)/T(p)
definition, since MLUPS is inversely proportional to wall time at a fixed
problem size. Reads milestone6_results/mlups_report.csv (produced from the
bwUniCluster strong-scaling runs, see cluster/README.md).
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("milestone6_results/mlups_report.csv"))
    parser.add_argument("--output", type=Path, default=Path("milestone6_results/mlups_scaling"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    with args.csv.open() as file:
        rows = list(csv.DictReader(file))

    backends = {}
    for row in rows:
        backends.setdefault(row["backend"], []).append(
            dict(ranks=int(row["ranks"]), mlups=float(row["median_mlups"]),
                 min_mlups=float(row["min_mlups"]), max_mlups=float(row["max_mlups"]))
        )
    for entries in backends.values():
        entries.sort(key=lambda r: r["ranks"])
        base = next(r["mlups"] for r in entries if r["ranks"] == 1)
        for r in entries:
            r["speedup"] = r["mlups"] / base
            r["efficiency"] = r["speedup"] / r["ranks"]

    style = dict(cpu=dict(color="#2a78d6", label="CPU · Kokkos Serial (256×256)"),
                 gpu=dict(color="#eb6834", label="GPU · Kokkos CUDA, A100 (4096×4096)"))
    all_ranks = sorted({r["ranks"] for entries in backends.values() for r in entries})

    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax_speedup, ax_eff) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")

    # --- speedup: log-log against ideal S(p) = p ---------------------------
    ax_speedup.plot(all_ranks, all_ranks, "--", color="0.55", linewidth=2, label="Ideal: S(p) = p")
    for backend, entries in backends.items():
        ranks = [r["ranks"] for r in entries]
        speedup = [r["speedup"] for r in entries]
        entries0 = entries[0]["mlups"]
        lower = [r["speedup"] - r["min_mlups"] / entries0 for r in entries]
        upper = [r["max_mlups"] / entries0 - r["speedup"] for r in entries]
        ax_speedup.errorbar(ranks, speedup, yerr=[lower, upper], fmt="o-", capsize=4,
                            linewidth=2.2, markersize=7, color=style[backend]["color"],
                            label=style[backend]["label"])
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_yscale("log", base=2)
    ax_speedup.set_xticks(all_ranks, labels=[str(p) for p in all_ranks])
    ax_speedup.set_yticks(all_ranks, labels=[str(p) for p in all_ranks])
    ax_speedup.set_xlabel("MPI ranks (one core / one GPU each)")
    ax_speedup.set_ylabel("Speedup S(p) = MLUPS(p) / MLUPS(1)")
    ax_speedup.set_title("Strong-scaling speedup")
    ax_speedup.grid(True, which="both", alpha=0.2)
    ax_speedup.legend(loc="upper left")

    # --- efficiency: linear 0-100% against flat 100% ideal -----------------
    ax_eff.axhline(1.0, linestyle="--", color="0.55", linewidth=2, label="Ideal: 100% efficiency")
    for backend, entries in backends.items():
        ranks = [r["ranks"] for r in entries]
        efficiency = [r["efficiency"] for r in entries]
        entries0 = entries[0]["mlups"]
        eff_lower = [r["efficiency"] - r["min_mlups"] / entries0 / r["ranks"] for r in entries]
        eff_upper = [r["max_mlups"] / entries0 / r["ranks"] - r["efficiency"] for r in entries]
        ax_eff.errorbar(ranks, efficiency, yerr=[eff_lower, eff_upper], fmt="o-", capsize=4,
                        linewidth=2.2, markersize=7, color=style[backend]["color"],
                        label=style[backend]["label"])
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_xticks(all_ranks, labels=[str(p) for p in all_ranks])
    ax_eff.set_ylim(0, 1.1)
    ax_eff.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_eff.set_xlabel("MPI ranks (one core / one GPU each)")
    ax_eff.set_ylabel("Efficiency = Speedup / p")
    ax_eff.set_title("Parallel efficiency")
    ax_eff.grid(True, which="both", alpha=0.2)
    ax_eff.legend(loc="lower left")

    fig.suptitle("bwUniCluster: lid-driven cavity MLUPS strong scaling", fontsize=17, y=1.04)

    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None,
                    bbox_inches="tight")
    print(f"Wrote {args.output}.{{png,svg,pdf}}")
    print()
    print("backend  ranks  mlups        speedup  efficiency")
    for backend, entries in backends.items():
        for r in entries:
            print(f"{backend:7}  {r['ranks']:5}  {r['mlups']:10.3f}  {r['speedup']:7.3f}  {r['efficiency']:9.2%}")


if __name__ == "__main__":
    main()
