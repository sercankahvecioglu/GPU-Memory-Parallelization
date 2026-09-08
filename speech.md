# YALB — Yet Another Lattice Boltzmann Solver
### 10-minute presentation speech

Project: D2Q9 Lattice Boltzmann Method solver, built incrementally across
milestones 1–6 of the Accelerators course, using Kokkos for on-node
parallelism and MPI for distributed-memory domain decomposition.

---

## 1. Introduction (≈1 min)

Good afternoon. I'm going to walk you through my LBM solver. The goal of the project was
to implement a D2Q9 lattice Boltzmann solver, validate it against known
physics, make it fast, and then make it run on more than one core and more
than one machine. I'll show you what I built,
show you a live run of the classic lid-driven cavity benchmark, and finish
with performance numbers and what I'd change next time.

## 2. What did I do? Programming model (≈1.5 min)

I implemented the full D2Q9 BGK lattice Boltzmann method: collision,
streaming, and boundary handling, in C++, on top of **Kokkos** for the
compute kernels and **raw MPI** for distributed-memory parallelism.

The reason for that combination: Kokkos gives me a single source for the
numerical kernels — `Kokkos::View` for the population, density and velocity
arrays, and `Kokkos::parallel_for` for every loop over the lattice — that
can be compiled against different execution backends (Serial, OpenMP,
CUDA, ...) without touching the kernel code. On top of that single-node
layer, I wrote the domain decomposition and halo exchange in MPI. So the mental model is: Kokkos parallelizes
*within* a rank, MPI parallelizes *across* ranks, and the two composes into
one hybrid MPI+Kokkos solver.

## 3. Methods (≈1.5 min)

The solver itself is a standard D2Q9 BGK scheme: 9 discrete velocities per
cell, an equilibrium distribution `f_eq` computed from local density and
velocity, and a relaxation step `f = f + omega * (f_eq - f)`, followed by
streaming along the 9 lattice directions. I validated it in stages, exactly
as the milestones prescribed:

- **Milestone 3**: streaming and boundary primitives, unit-tested against
  hand-computed reference states.
- **Milestone 4**: a decaying shear wave, from which I measure the
  numerically-realized viscosity by fitting the amplitude decay, and
  compare it against the analytic `nu = (1/omega - 1/2)/3` relation across
  five omega values — this is the classic sanity check that the collision
  operator produces the correct transport coefficient.
- **Milestone 5**: the lid-driven cavity at Reynolds Number = 400 on a 128×128 grid,
  run to steady state (convergence limit 1e-10 on the max velocity change,
  checked every 100 steps), and validated against the Ghia, Ghia & Shin
  reference velocity profile.
- **Milestone 6**: the same cavity solver, re-implemented for distributed
  memory: the domain is split into contiguous vertical strips across MPI
  ranks, each rank owns its columns plus two ghost columns, and only the
  post-collision populations that actually cross a rank boundary are
  exchanged.

## 4. The programming model in detail — how it differs from the lecture (≈2 min)

In the lecture, the natural path is: write a serial kernel, then bolt on
OpenMP pragmas for shared memory, and separately hand-write a CUDA kernel
if you want the GPU — three different code paths for three backends, kept
in sync by hand. I avoided that by writing every kernel exactly once
against the Kokkos execution and memory model: `compute_density`,
`compute_velocity`, `collision`, `streaming_with_walls`, and so on are all
plain functions that take `Kokkos::View`s and dispatch a `parallel_for`
over a flattened index. Swapping `Kokkos::Serial` for `Kokkos::OpenMP` or a
GPU backend is a CMake/compile-time decision, not a rewrite.

The other deliberate choice is *not* using a distributed array library. I wrote the
halo exchange by hand with two `MPI_Sendrecv` calls per timestep, moving
only the three post-collision populations that travel in each direction. 
That's `3*NY - 2` doubles per
neighbor, not the full ghost column, which cuts the exchanged volume
compared to a naive "send everything" halo. It also means the boundary
logic — physical wall vs. MPI interface — has to be resolved from global
coordinates inside the streaming kernel itself, which is the trickiest
part of the whole project: a cell doesn't know from local ghost-array
layout alone whether it's next to a wall or next to another rank.

## 5. Advantages and pitfalls (≈1 min)

**Advantages**: one kernel codebase for every backend Kokkos supports;
the MPI layer is thin and auditable — I can reason about exactly what
bytes move every step; and correctness could be checked incrementally —
milestone 6 has a `--check-solver` mode that reruns the full serial
reference on every rank and diffs against it, so the distributed code is
validated against the single-node code, not just against physics.

**Pitfalls**: Kokkos is a heavier build-time dependency and its
error messages when a lambda captures the wrong thing are brutal. Hand
rolled MPI halo exchange means I own every indexing bug — the global
vs. local coordinate translation at rank boundaries was the single
largest source of bugs in the whole project. And because I optimized the
halo message to exclude the wall-bound diagonals, that logic is now
coupled to the specific D2Q9 stencil — it's not something you could reuse
for a different velocity set without re-deriving which channels need to
travel.

## 6. Results — show it works (≈1.5 min)

*(cut to the recorded cavity-evolution GIF here —
`milestone5_results/cavity_evolution.gif`)*

This is the milestone 5 lid-driven cavity at Reynolds number 400, run to
steady state on a 128×128 grid — you can see the classic primary vortex
form under the moving lid, with the secondary corner vortices developing
in the bottom corners as the flow converges. It converged after 83,200
steps. Against the 17 benchmark points from Ghia, Ghia & Shin, the
normalized RMSE is 0.36%, with a worst-case error of 0.83% of the lid
speed. The distributed milestone 6 version reproduces the serial solver
to within 8.9e-16 in population values across 1, 2, 3 and 4 MPI ranks —
essentially machine precision — and mass and kinetic energy are conserved
across ranks to the same tolerance.

## 7. Performance — GLUPS and strong scaling (≈2 min)

The single-rank, Kokkos-Serial baseline on the milestone 6 solver sustains
**20.98 million lattice updates per second (MLUPS)** on a 256×256 cavity —
2.10e-5 GLUPS in the requested 10^9-per-second units — measured on one
bwUniCluster CPU core over 20,000 timesteps, with 100 untimed warmup steps
discarded and I/O, diagnostics, and convergence checks excluded from the
timed region.

**CPU strong scaling** (same 256×256 problem, 1→32 MPI ranks, one rank per
core, 3 repeats): throughput climbs to **616.97 MLUPS at 32 cores** — a
29.4x speedup, 92% parallel efficiency. I didn't stop at one problem size:
I swept the same 1-32 rank ladder across five grid sizes (64² up to 1024²)
to test whether the halo exchange's fixed per-step cost starts to dominate
as each rank's share of the domain shrinks. It does, exactly as predicted —
parallel efficiency at 32 ranks rises monotonically with problem size, from
61% at 64×64 up to 96% at 512×512, before edging down slightly to 95% at
1024×1024 (that grid uses fewer, longer timesteps, so its numbers are a bit
noisier). Small grids are communication-bound; large grids are not.

**GPU strong scaling**: the same Kokkos source, recompiled for CUDA, no
code changes. One A100 sustains **2998 MLUPS** on a 4096×4096 grid — about
**143x** a single CPU core — and scaling to 4 A100s reaches **10,432
MLUPS**, 87% efficiency (3.48x speedup). That's over 10 GLUPS on 4 GPUs
from one hybrid Kokkos+MPI codebase that also runs, unmodified, on a single
laptop core.

## 8. Discussion (≈1 min)

What I learned: getting the physics right (BGK collision, equilibrium
distribution, bounce-back boundaries) was the easy 60% of the project;
the hard part was the bookkeeping at parallelization boundaries — ghost
cells, global-vs-local coordinates, and making sure a distributed run
gives bit-compatible-to-precision results against the serial reference.
Writing the `--check-solver` cross-validation mode early would have saved
me time; I added it after already debugging several halo bugs by hand.
The multi-size scaling sweep confirmed a suspicion I had going in — small
grids are communication-bound at high rank counts — but I only had a
number for it after actually running the five-size ladder, not before.

What I'd do differently next time: build the correctness harness
(analytic tests, serial-vs-parallel diffing) *before* writing the parallel
kernel, not after. I'd also budget more cluster time for edge cases: my
64-rank runs hung outright (18.6% CPU efficiency over a 7.5-minute
timeout, most ranks idle) rather than degrading gracefully, most likely an
MPI launch/binding issue at that process count rather than a bug in the
halo exchange itself — I didn't have time to isolate it further this
round, so the CPU story stops at a clean 1-32 rank curve.

Thank you — happy to take questions, including on the code itself.