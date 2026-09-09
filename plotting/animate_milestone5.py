#!/usr/bin/env python3
"""Animate the Milestone 5 lid-driven cavity from stationary fluid to steady state.

Reimplements the exact D2Q9 BGK solver from src/lbm.cpp / executables/milestone5.cpp
in numpy (same lattice weights, same bounce-back + moving-lid boundary condition,
same physical constants), so no Kokkos/CMake build is required just to visualize it.
Periodic velocity-magnitude snapshots are assembled into an animated GIF.
"""

import argparse
from pathlib import Path

import numpy as np

# Physical parameters, identical to executables/milestone5.cpp.
NX = NY = 128
LID_VELOCITY = 0.1
OMEGA = 1.0 / 0.596
WALL_DENSITY = 1.0
CONVERGENCE_CHECK_INTERVAL = 100
CONVERGENCE_LIMIT = 1.0e-10
MAX_STEPS = 200000
CS2 = 1.0 / 3.0

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
WEIGHTS = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def equilibrium(rho, ux, uy):
    usq = ux * ux + uy * uy
    feq = np.empty((NX, NY, 9))
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        feq[:, :, i] = WEIGHTS[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def precompute_boundary_geometry():
    """Boundary validity/lid masks depend only on lattice geometry, not on f."""
    x_grid, y_grid = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    geometry = []
    for j in range(9):
        i = OPPOSITE[j]
        src_x = x_grid - CX[j]
        src_y = y_grid - CY[j]
        valid = (src_x >= 0) & (src_x < NX) & (src_y >= 0) & (src_y < NY)
        hits_moving_lid = (~valid) & (src_y >= NY) & (x_grid > 0) & (x_grid < NX - 1)
        correction = 2.0 * WEIGHTS[i] * WALL_DENSITY * (CX[i] * LID_VELOCITY) / CS2
        lid_delta = np.where(hits_moving_lid, correction, 0.0)
        geometry.append((i, valid, lid_delta))
    return geometry


def step(f, geometry):
    rho = f.sum(axis=2)
    inv_rho = 1.0 / rho
    ux = (f * CX).sum(axis=2) * inv_rho
    uy = (f * CY).sum(axis=2) * inv_rho
    f_post = f - OMEGA * (f - equilibrium(rho, ux, uy))

    f_new = np.empty_like(f)
    for j in range(9):
        i, valid, lid_delta = geometry[j]
        gathered = np.roll(f_post[:, :, j], shift=(CX[j], CY[j]), axis=(0, 1))
        # Bounce-back: population that would have left through a wall returns
        # in the opposite direction at the same node (src invalid = wall hit).
        bounced = f_post[:, :, i] - lid_delta
        f_new[:, :, j] = np.where(valid, gathered, bounced)

    return f_new, ux, uy


def run(save_every, max_frames):
    geometry = precompute_boundary_geometry()

    f = np.empty((NX, NY, 9))
    for i in range(9):
        f[:, :, i] = WEIGHTS[i]  # rho = 1, u = 0 initial equilibrium.

    frames = []
    frame_steps = []
    ux = np.zeros((NX, NY))
    uy = np.zeros((NX, NY))
    velocity_change = np.inf

    for current_step in range(1, MAX_STEPS + 1):
        check_convergence = current_step % CONVERGENCE_CHECK_INTERVAL == 0
        if check_convergence:
            previous_ux, previous_uy = ux.copy(), uy.copy()

        f, ux, uy = step(f, geometry)

        if check_convergence:
            velocity_change = np.max(
                np.sqrt((ux - previous_ux) ** 2 + (uy - previous_uy) ** 2)
            )

        if current_step % save_every == 0 or current_step == 1:
            frames.append(np.sqrt(ux ** 2 + uy ** 2).copy())
            frame_steps.append(current_step)
            if len(frames) >= max_frames:
                save_every *= 2  # Keep the GIF short even if convergence is slow.

        if check_convergence and velocity_change < CONVERGENCE_LIMIT:
            break

    if not (velocity_change < CONVERGENCE_LIMIT):
        raise RuntimeError("cavity did not converge within MAX_STEPS")

    frames.append(np.sqrt(ux ** 2 + uy ** 2).copy())
    frame_steps.append(current_step)
    return frames, frame_steps, current_step


def render(frames, frame_steps, total_steps, output_path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    maximum_speed = max(frame.max() for frame in frames)  # Rule 4: one fixed scale.

    def title_text(step):
        return f"Lid-driven cavity, Re = 400\nstep {step:,} / {total_steps:,}"

    # Extra headroom (top=0.86) keeps the two-line title from clipping against
    # the figure edge; a small explicit colorbar keeps it from crowding the plot.
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    fig.subplots_adjust(top=0.86, right=0.88)
    image = ax.imshow(
        frames[0].T, origin="lower", cmap="viridis", vmin=0, vmax=maximum_speed,
        extent=[0, NX - 1, 0, NY - 1],
    )
    ax.set_xlabel("x (lattice sites)")
    ax.set_ylabel("y (lattice sites)")
    title = fig.suptitle(title_text(frame_steps[0]), fontsize=18, y=0.98)
    colorbar = fig.colorbar(image, ax=ax, label="speed", fraction=0.046, pad=0.04, shrink=0.85)

    def update(frame_index):
        image.set_data(frames[frame_index].T)
        title.set_text(title_text(frame_steps[frame_index]))
        return image, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("milestone5_results/cavity_evolution.gif"))
    parser.add_argument("--save-every", type=int, default=200,
                        help="Capture a frame every N solver steps (doubles if --max-frames is hit).")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--cache", type=Path, default=Path("milestone5_results/cavity_evolution_frames.npz"),
                        help="Reuse cached frames from a previous run (skips the ~minutes-long simulation) "
                             "when re-rendering after style-only changes; recomputed if missing.")
    parser.add_argument("--recompute", action="store_true", help="Ignore any existing cache.")
    args = parser.parse_args()

    if args.cache.exists() and not args.recompute:
        cached = np.load(args.cache)
        frames = list(cached["frames"])
        frame_steps = list(cached["frame_steps"])
        total_steps = int(cached["total_steps"])
        print(f"Loaded {len(frames)} cached frames from {args.cache}")
    else:
        frames, frame_steps, total_steps = run(args.save_every, args.max_frames)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, frames=np.array(frames), frame_steps=np.array(frame_steps),
                            total_steps=total_steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(frames, frame_steps, total_steps, args.output, args.fps)
    print(f"Converged after {total_steps:,} steps; wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
