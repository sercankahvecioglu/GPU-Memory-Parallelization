#!/usr/bin/env python3
"""Animate the Milestone 2 streaming operator: a velocity blob traveling
diagonally across a periodic domain and re-emerging at its starting point.

Reimplements the pure streaming step from src/lbm.cpp (streaming, compute_density,
compute_velocity) in numpy with the exact same D2Q9 lattice and the same blob
initial condition used in tests/test_streaming.cpp, so no Kokkos/CMake build is
required just to visualize it. No collision is applied: this is the streaming-only
milestone, so the blob translates one lattice site per step without distortion.
"""

import argparse
from pathlib import Path

import numpy as np

NX, NY = 15, 10
Q = 9
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
DIRECTION = 5  # diagonal (+x, +y): shows motion in both axes at once.


def initialize_blob():
    """Same 2x2 blob and placement as StreamingTest in tests/test_streaming.cpp."""
    f = np.zeros((NX, NY, Q))
    f[4, 3, DIRECTION] = 1.25
    f[5, 3, DIRECTION] = 2.50
    f[4, 4, DIRECTION] = 3.75
    f[5, 4, DIRECTION] = 5.00
    return f


def stream(f):
    f_new = np.zeros_like(f)
    for i in range(Q):
        f_new[:, :, i] = np.roll(f[:, :, i], shift=(CX[i], CY[i]), axis=(0, 1))
    return f_new


def velocity_field(f):
    rho = f.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = (f * CX).sum(axis=2) / rho
        uy = (f * CY).sum(axis=2) / rho
    ux = np.nan_to_num(ux)
    uy = np.nan_to_num(uy)
    return ux, uy


def run(num_periods):
    round_trip_steps = np.lcm(NX, NY)  # blob returns to its start after this many steps
    total_steps = round_trip_steps * num_periods

    f = initialize_blob()
    frames = [velocity_field(f)]
    for _ in range(total_steps):
        f = stream(f)
        frames.append(velocity_field(f))
    return frames, round_trip_steps


def render(frames, round_trip_steps, output_path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
    })

    x_grid, y_grid = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    max_speed = max(np.hypot(ux, uy).max() for ux, uy in frames)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(top=0.85)
    ux0, uy0 = frames[0]
    speed0 = np.hypot(ux0, uy0)
    quiver = ax.quiver(
        x_grid, y_grid, ux0, uy0, speed0,
        cmap="viridis", clim=(0, max_speed), scale=1.0, scale_units="xy",
        pivot="mid",
    )
    ax.set_xlim(-0.5, NX - 0.5)
    ax.set_ylim(-0.5, NY - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (lattice sites)")
    ax.set_ylabel("y (lattice sites)")
    fig.colorbar(quiver, ax=ax, label="speed", fraction=0.046, pad=0.04)
    title = fig.suptitle("", fontsize=15, y=0.97)

    def title_text(step):
        return (
            "Milestone 2: streaming-only velocity field\n"
            f"step {step} / {len(frames) - 1}  (periodic re-emergence every {round_trip_steps} steps)"
        )

    def update(step):
        ux, uy = frames[step]
        quiver.set_UVC(ux, uy, np.hypot(ux, uy))
        title.set_text(title_text(step))
        return quiver, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("milestone2_results/velocity_field_evolution.gif"))
    parser.add_argument("--periods", type=int, default=2, help="Number of full periodic round trips to animate.")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    frames, round_trip_steps = run(args.periods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(frames, round_trip_steps, args.output, args.fps)
    print(f"Wrote {len(frames)} frames ({args.periods} period(s) of {round_trip_steps} steps) to {args.output}")


if __name__ == "__main__":
    main()
