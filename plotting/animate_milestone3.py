#!/usr/bin/env python3
"""Animate the Milestone 3 collision + streaming operator: a central density
bump relaxing via the BGK collision and spreading out as damped sound waves
on a periodic domain.

Reimplements compute_density, compute_velocity, compute_f_eq, collision, and
streaming from src/lbm.cpp in numpy (same D2Q9 lattice, same equilibrium
formula, same BGK relaxation), so no Kokkos/CMake build is required just to
visualize it.
"""

import argparse
from pathlib import Path

import numpy as np

NX, NY = 50, 50
OMEGA = 1.0
RHO0 = 0.5       # background density, 0 < rho < 1 per the milestone spec
BUMP = 0.3       # peak excess density at the center
SIGMA = 3.0      # width of the initial Gaussian bump

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
WEIGHTS = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def equilibrium(rho, ux, uy):
    usq = ux * ux + uy * uy
    feq = np.empty((NX, NY, 9))
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        feq[:, :, i] = WEIGHTS[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def initialize_bump():
    """Uniform density with a central Gaussian bump, zero initial velocity."""
    x_grid, y_grid = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    dx = x_grid - NX // 2
    dy = y_grid - NY // 2
    rho = RHO0 + BUMP * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * SIGMA ** 2))
    ux = np.zeros((NX, NY))
    uy = np.zeros((NX, NY))
    return equilibrium(rho, ux, uy)


def step(f):
    rho = f.sum(axis=2)
    ux = (f * CX).sum(axis=2) / rho
    uy = (f * CY).sum(axis=2) / rho

    f_post = f + OMEGA * (equilibrium(rho, ux, uy) - f)

    f_new = np.empty_like(f)
    for i in range(9):
        f_new[:, :, i] = np.roll(f_post[:, :, i], shift=(CX[i], CY[i]), axis=(0, 1))

    return f_new, rho, ux, uy


def run(num_steps, save_every):
    f = initialize_bump()

    frames = []
    frame_steps = []
    for current_step in range(num_steps + 1):
        rho = f.sum(axis=2)
        if current_step % save_every == 0:
            frames.append(rho.copy())
            frame_steps.append(current_step)
        if current_step < num_steps:
            f, rho, ux, uy = step(f)

    return frames, frame_steps


def render(frames, frame_steps, output_path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
    })

    vmin = min(frame.min() for frame in frames)
    vmax = max(frame.max() for frame in frames)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(top=0.85, right=0.88)
    image = ax.imshow(
        frames[0].T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
        extent=[0, NX - 1, 0, NY - 1],
    )
    ax.set_xlabel("x (lattice sites)")
    ax.set_ylabel("y (lattice sites)")
    title = fig.suptitle("", fontsize=15, y=0.97)
    fig.colorbar(image, ax=ax, label="density", fraction=0.046, pad=0.04)

    def title_text(step):
        return f"Milestone 3: collision + streaming\nstep {step} / {frame_steps[-1]} (omega = {OMEGA})"

    def update(frame_index):
        image.set_data(frames[frame_index].T)
        title.set_text(title_text(frame_steps[frame_index]))
        return image, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("milestone3_results/density_wave_evolution.gif"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    frames, frame_steps = run(args.steps, args.save_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(frames, frame_steps, args.output, args.fps)
    print(f"Wrote {len(frames)} frames over {args.steps} steps to {args.output}")


if __name__ == "__main__":
    main()
