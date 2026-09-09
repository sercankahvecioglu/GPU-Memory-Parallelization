#!/usr/bin/env python3
"""Plot the Milestone 5 MLUPS benchmark: per-repeat performance of the
single-process solver, as required by
https://pastewka.github.io/Accelerators/project/milestone05.html and reused
as the single-process baseline in Milestone 6.

Reads milestone5_results/mlups_report.txt (written by the milestone5
executable) and renders a bar chart of MLUPS per timed repeat with the
median marked, so the reported number's run-to-run spread is visible
alongside the headline figure.
"""
import argparse
from pathlib import Path


def parse_report(path):
    repeats = []
    fields = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        is_repeat_line = line.startswith("repeat_")
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if is_repeat_line and key == "mlups":
                repeats.append(float(value))
            else:
                fields[key] = value
    return repeats, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("milestone5_results/mlups_report.txt"))
    parser.add_argument("--output", type=Path, default=Path("milestone5_results/mlups_benchmark"))
    args = parser.parse_args()

    repeats, fields = parse_report(args.report)
    median = float(fields["median_mlups"])
    minimum = float(fields["min_mlups"])
    maximum = float(fields["max_mlups"])
    grid = fields["grid"]
    backend = fields["backend"]
    steps = fields["solver_only_steps_per_repeat"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    color = "#2a78d6"
    repeat_numbers = list(range(1, len(repeats) + 1))

    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    ax.bar(repeat_numbers, repeats, color=color, width=0.6, zorder=3)
    ax.axhline(median, linestyle="--", color="0.35", linewidth=2, zorder=4,
               label=f"Median: {median:.2f} MLUPS")
    ax.set_ylim(0, maximum * 1.15)
    ax.set_xticks(repeat_numbers)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("MLUPS")
    ax.set_title(f"Milestone 5 solver-only MLUPS benchmark\n{grid} grid, {backend}, {steps} steps/repeat")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="lower right")

    ax.annotate(
        f"single-process baseline\nmin {minimum:.2f} · median {median:.2f} · max {maximum:.2f}",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=11, color="0.25",
    )

    for suffix in ["png", "svg"]:
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180 if suffix == "png" else None,
                    bbox_inches="tight")
    print(f"Wrote {args.output}.{{png,svg}}")
    print(f"Median MLUPS (single-process baseline for Milestone 6): {median:.4f}")


if __name__ == "__main__":
    main()
