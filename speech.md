# YALB — Yet Another Lattice Boltzmann Solver
### 10-minute presentation speech

Project: D2Q9 Lattice Boltzmann Method solver, built incrementally across
milestones 1–6 of the Accelerators course, using Kokkos for on-node
parallelism and MPI for distributed-memory domain decomposition.

---

## 1. Introduction (≈1 min)

Good [morning/afternoon]. I'm going to walk you through YALB — my Lattice
Boltzmann solver for the Accelerators project. The goal of the project was
to implement a D2Q9 lattice Boltzmann solver, validate it against known
physics, make it fast, and then make it run on more than one core and more
than one machine. I'll show you what I built, how it's structured, what it
costs and gains me compared to the "textbook" approach from the lecture,
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
layer, I hand-wrote the domain decomposition and halo exchange in MPI
myself, rather than relying on a library that would have hidden the
communication pattern from me. So the mental model is: Kokkos parallelizes
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
- **Milestone 5**: the lid-driven cavity at Re = 400 on a 128×128 grid,
  run to steady state (convergence limit 1e-10 on the max velocity change,
  checked every 100 steps), and validated against the Ghia, Ghia & Shin
  1982 reference velocity profile.
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

The other deliberate choice is *not* using a distributed array library —
no Kokkos "remote spaces", no PETSc-style DistributedArray. I wrote the
halo exchange by hand with two `MPI_Sendrecv` calls per timestep, moving
only the three post-collision populations that travel in each direction
(channels 1, 5, 8 rightward; 3, 6, 7 leftward — the diagonals that would
hit a wall are dropped from the message). That's `3*NY - 2` doubles per
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

*(cut to the recorded video of the sliding lid here)*

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

## 7. Performance — GLUPS (≈1.5 min)

The single-rank, Kokkos-Serial baseline sustains **19.44 million lattice
updates per second (MLUPS)** on a 128×128 cavity — that's 0.0194 GLUPS, or
1.944e-5 GLUPS in the requested 10^9-per-second units, measured over 2,000
timesteps with synchronization included and I/O and convergence checks
excluded.

[If you have run cluster/strong_scaling.sbatch by presentation time, replace
this paragraph with the actual numbers: report GLUPS for 1, 2, 4, 8, 16, 32,
64 ranks on the 256×256/20,000-step strong-scaling sweep, and quote speedup
and parallel efficiency at the largest rank count from
milestone6_results/scaling/<JOBID>/summary.csv.]

The scaling study itself — 256×256 cells, 20,000 steps, Re = 800, at 1
through 64 MPI ranks with three repeats each — is designed and scripted
(`cluster/run_scaling.py`, `cluster/strong_scaling.sbatch`) to run on
bwUniCluster; [state here whether it has completed by presentation day].
It measures pure solver time — collision, halo exchange, streaming, field
update — with MPI_MAX taken over ranks so the slowest rank sets the
reported time.

## 8. Discussion (≈1 min)

What I learned: getting the physics right (BGK collision, equilibrium
distribution, bounce-back boundaries) was the easy 60% of the project;
the hard part was the bookkeeping at parallelization boundaries — ghost
cells, global-vs-local coordinates, and making sure a distributed run
gives bit-compatible-to-precision results against the serial reference.
Writing the `--check-solver` cross-validation mode early would have saved
me time; I added it after already debugging several halo bugs by hand.

What I'd do differently next time: build the correctness harness
(analytic tests, serial-vs-parallel diffing) *before* writing the
parallel kernel, not after. I'd also profile the halo exchange
communication-to-computation ratio before assuming vertical strip
decomposition scales well to 64 ranks — a 256-column domain across 64
ranks gives each rank only 4 columns, which is a communication-bound
regime I have not yet measured.

Thank you — happy to take questions, including on the code itself.