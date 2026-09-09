#!/usr/bin/env python3
"""Create a dependency-free SVG plot from the Milestone 4 CSV output.

Reads viscosity_vs_omega.csv (written by executables/milestone4.cpp) and
draws measured viscosity points against the theoretical BGK-LBM curve
nu(omega) = (1/omega - 0.5) / 3, with proper axis ticks so the plot is
reproducible from data alone (unlike the earlier hand-edited SVG).
"""

import argparse
import csv
from pathlib import Path

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 14

# Okabe-Ito colour-blind-safe palette.
MEASURED_COLOUR = "#0072B2"
THEORY_COLOUR = "#D55E00"


def read_csv(filename):
    with filename.open(newline="") as file:
        return [
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(file)
        ]


def nice_step(span, target_ticks=5):
    """Pick a human-friendly axis step (1/2/5 * 10^k) close to span/target_ticks."""
    if span <= 0:
        return 1.0
    raw_step = span / target_ticks
    exponent = 0
    while raw_step >= 10:
        raw_step /= 10
        exponent += 1
    while raw_step < 1:
        raw_step *= 10
        exponent -= 1
    for candidate in (1, 2, 5, 10):
        if raw_step <= candidate:
            return candidate * 10 ** exponent
    return 10 * 10 ** exponent


def theoretical_viscosity(omega):
    return (1.0 / omega - 0.5) / 3.0


def plot_viscosity_vs_omega(rows, output_directory):
    width, height = 820, 520
    left, right, top, bottom = 70, 790, 30, 455

    omegas = [row["omega"] for row in rows]
    measured = [row["nu_measured"] for row in rows]
    omega_step = min(b - a for a, b in zip(omegas, omegas[1:])) if len(omegas) > 1 else 0.2
    omega_min = min(omegas) - 0.5 * omega_step
    omega_max = max(omegas) + 0.5 * omega_step

    y_max_raw = max(measured + [theoretical_viscosity(o) for o in (omega_min, omega_max)])
    y_step = nice_step(y_max_raw)
    y_max = y_step * (int(y_max_raw / y_step) + 1)

    def screen_x(omega):
        return left + (omega - omega_min) / (omega_max - omega_min) * (right - left)

    def screen_y(nu):
        return bottom - nu / y_max * (bottom - top)

    theory_samples = 100
    theory_points = " ".join(
        f"{screen_x(omega_min + i * (omega_max - omega_min) / (theory_samples - 1)):.2f},"
        f"{screen_y(theoretical_viscosity(omega_min + i * (omega_max - omega_min) / (theory_samples - 1))):.4f}"
        for i in range(theory_samples)
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<polyline fill="none" stroke="{THEORY_COLOUR}" stroke-width="2" points="{theory_points}"/>',
    ]

    for row in rows:
        svg.append(
            f'<circle cx="{screen_x(row["omega"]):.2f}" cy="{screen_y(row["nu_measured"]):.4f}" '
            f'r="5" fill="{MEASURED_COLOUR}"/>'
        )

    for omega in omegas:
        x = screen_x(omega)
        svg.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}" stroke="black"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="{TICK_SIZE}">{omega:.1f}</text>'
        )

    y_tick = 0.0
    while y_tick <= y_max + 1e-9:
        y = screen_y(y_tick)
        svg.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black"/>')
        svg.append(
            f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="{TICK_SIZE}">{y_tick:.2f}</text>'
        )
        y_tick += y_step

    svg.extend([
        f'<text x="{(left + right) / 2}" y="{height - 15}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="{LABEL_SIZE}">omega</text>',
        f'<text x="20" y="{(top + bottom) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {(top + bottom) / 2})" font-family="sans-serif" '
        f'font-size="{LABEL_SIZE}">kinematic viscosity</text>',
        f'<text x="{(left + right) / 2}" y="20" text-anchor="middle" '
        f'font-family="sans-serif" font-size="{TITLE_SIZE}">Measured and theoretical viscosity</text>',
        f'<circle cx="600" cy="52" r="5" fill="{MEASURED_COLOUR}"/>'
        f'<text x="610" y="56" font-family="sans-serif" font-size="{TICK_SIZE}" '
        f'fill="{MEASURED_COLOUR}">measured</text>',
        f'<line x1="590" y1="75" x2="610" y2="75" stroke="{THEORY_COLOUR}" stroke-width="2"/>'
        f'<text x="610" y="79" font-family="sans-serif" font-size="{TICK_SIZE}" '
        f'fill="{THEORY_COLOUR}">theory</text>',
        "</svg>",
    ])

    (output_directory / "viscosity_vs_omega.svg").write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description="Plot Milestone 4 viscosity-vs-omega CSV output.")
    parser.add_argument(
        "results_directory",
        nargs="?",
        type=Path,
        default=Path("milestone4_results"),
    )
    args = parser.parse_args()

    csv_file = args.results_directory / "viscosity_vs_omega.csv"
    if not csv_file.exists():
        raise FileNotFoundError("Run the milestone4 executable before plotting.")

    rows = read_csv(csv_file)
    plot_viscosity_vs_omega(rows, args.results_directory)
    print(f"Plot written to {args.results_directory / 'viscosity_vs_omega.svg'}")


if __name__ == "__main__":
    main()
