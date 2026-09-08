# YALB — Yet Another Lattice Boltzmann Solver
### Presentation speech (Introduction / Methods / Results / Discussion)

Project: D2Q9 Lattice Boltzmann Method solver, built incrementally across
milestones 1–6 of the Accelerators course, using Kokkos for on-node
parallelism and MPI for distributed-memory domain decomposition.

Structure follows the standard four-part talk: Introduction (what and how),
Methods (programming model, how it works, advantages and pitfalls), Results
(show the code working, then performance), Discussion (lessons, what's
next). Each section ends with a one-line spoken transition — say it out
loud before moving on, so people get a second to reset before the next
part starts. Learn the gist of each section, not a script word-for-word.

---

## 1. Introduction — what did I do, how did I do it? (≈2.5 min)

Good [morning/afternoon]. I'll walk you through YALB, my Lattice Boltzmann
solver for the Accelerators project. The goal was pretty simple to state:
implement a D2Q9 lattice Boltzmann solver, validate it against known
physics, make it fast, and then get it running on more than one core and
more than one machine.

So I implemented the full D2Q9 BGK method — collision, streaming, boundary
handling — in C++, using Kokkos for the actual compute kernels and raw MPI
for the distributed-memory part.

Why that combination? Kokkos lets me write the numerical kernels once —
`Kokkos::View` for the population, density and velocity arrays,
`Kokkos::parallel_for` for every loop over the lattice — and then compile
that same code against different backends: Serial, OpenMP, CUDA, without
touching the kernels themselves. On top of that I hand-wrote the domain
decomposition and halo exchange in MPI myself, instead of pulling in a
library that would've hidden the communication pattern from me. So the
mental model, roughly: Kokkos handles parallelism inside a rank, MPI
handles it across ranks, and together they make one hybrid solver.

*(Transition: that's what I built and roughly how it's shaped — now
Methods: how each piece actually works, and where it differs from what we
did in lecture.)*

## 2. Methods — how it works, and how it differs from the lecture (≈4.5 min)

The solver itself is a standard D2Q9 BGK scheme: 9 discrete velocities per
cell, an equilibrium distribution computed from the local density and
velocity, a relaxation step (`f = f + omega * (f_eq - f)`), then streaming
along the 9 directions. Nothing exotic there. I validated it in stages,
following the milestones:

- Milestone 3: streaming and boundary primitives, checked against
  hand-computed reference states.
- Milestone 4: a decaying shear wave — I measure the viscosity the solver
  actually produces by fitting the amplitude decay, and compare it to the
  analytic `nu = (1/omega - 1/2)/3` across five omega values. Classic
  sanity check that the collision operator gives you the right transport
  coefficient.
- Milestone 5: lid-driven cavity at Re = 400 on a 128×128 grid, run to
  steady state, checked against the Ghia, Ghia & Shin 1982 reference
  profile.
- Milestone 6: the same cavity solver, but rebuilt for distributed memory
  — domain split into vertical strips across MPI ranks, each rank owns its
  columns plus two ghost columns, and only the populations that actually
  cross a rank boundary get exchanged.

**How this differs from the lecture.** In lecture, the natural path is to
write a serial kernel, bolt OpenMP pragmas onto it, then separately
hand-write a CUDA kernel if you want the GPU — three code paths for three
backends that you now have to keep in sync by hand. I wanted to avoid
that, so every kernel is written exactly once against Kokkos:
`compute_density`, `compute_velocity`, `collision`, `streaming_with_walls`
are all plain functions taking `Kokkos::View`s and dispatching a
`parallel_for`. Switching from `Kokkos::Serial` to `Kokkos::OpenMP` or a
GPU backend is a CMake flag, not a rewrite.

The other choice I made on purpose was not reaching for a distributed
array library — no Kokkos remote spaces, no PETSc-style distributed
array. I wrote the halo exchange myself, two `MPI_Sendrecv` calls per
timestep, and I only send the populations that actually travel in each
direction (channels 1, 5, 8 going right; 3, 6, 7 going left — the
diagonals that would just hit a wall get dropped from the message
entirely). That's `3*NY - 2` doubles per neighbor instead of a full ghost
column. The price you pay for that is that the boundary logic — is this a
physical wall or an MPI interface — has to get resolved from global
coordinates inside the streaming kernel itself, and that was honestly the
trickiest part of the whole project: a cell can't tell from its local
ghost-array position alone whether it's sitting next to a wall or next to
another rank.

And then there's a third thing worth flagging once GPUs entered the
picture: the halo exchange stages data through a host mirror before
calling MPI, even when Kokkos is built for CUDA. That costs an extra
device-to-host and host-to-device copy on every exchange, which
CUDA-aware MPI would let you skip. I actually tried to get rid of that —
rewrote the exchange to pass device pointers straight into
`MPI_Sendrecv`, no host mirror — and then checked whether it would
actually work by querying `mpi_built_with_cuda_support` on the cluster.
It came back false on every MPI module I could find there, the default
GNU build and both NVIDIA HPC SDK variants. So I couldn't get it working
and fell back to the host-staged version rather than ship something that
would segfault or quietly corrupt data.

Advantages of this setup: one kernel codebase covers every backend
Kokkos supports, the MPI layer stays thin enough that I can actually
reason about what bytes move on every step, and correctness could be
checked incrementally — milestone 6 has a `--check-solver` mode that
reruns the full serial reference on every rank and diffs against it, so
the distributed code gets checked against the single-node code, not just
against the physics.

The pitfalls are about what you'd expect. Kokkos is a heavier build
dependency, and when a lambda captures the wrong thing the error messages
are brutal. Rolling my own MPI halo exchange means every indexing bug is
mine to find — the global-vs-local coordinate translation at rank
boundaries caused more bugs than anything else in the project. And
because I optimized the halo message to skip the wall-bound diagonals,
that logic is now tied specifically to the D2Q9 stencil — you couldn't
reuse it for a different velocity set without re-deriving which channels
need to travel.

*(Transition: that's the how and the trade-offs — now let's actually look
at it. Results: first that it works, then how fast it is.)*

## 3. Results — we have code, let's show it works, then how fast it is (≈3.5 min)

*(cut to the recorded cavity-evolution GIF here —
`milestone5_results/cavity_evolution.gif`)*

This is the milestone 5 lid-driven cavity at Reynolds number 400, run to
steady state on a 128×128 grid. You can see the primary vortex form under
the moving lid, and the secondary corner vortices show up in the bottom
corners as it converges. It took 83,200 steps to get there. Against the
17 benchmark points from Ghia, Ghia & Shin, the normalized RMSE comes out
to 0.36%, worst-case error 0.83% of the lid speed. The distributed
milestone 6 version matches the serial solver to within 8.9e-16 in
population values across 1, 2, 3 and 4 MPI ranks — basically machine
precision — and mass and kinetic energy stay conserved across ranks to
the same tolerance.

On performance: the single-rank, Kokkos-Serial baseline sustains 20.98
million lattice updates per second — MLUPS — on a 256×256 cavity, which
is 2.10e-5 GLUPS in the units that were asked for. That's measured on one
bwUniCluster CPU core over 20,000 timesteps, 100 untimed warmup steps
thrown away, and I/O, diagnostics and convergence checks excluded from
the timed region.

For CPU strong scaling — same 256×256 problem, 1 to 32 MPI ranks, one
rank per core, 3 repeats — throughput climbs to 616.97 MLUPS at 32 cores.
That's a 29.4x speedup, 92% parallel efficiency. I didn't want to stop at
one grid size, so I ran the same 1-32 rank sweep across five sizes, from
64² up to 1024², to see whether the halo exchange's fixed per-step cost
starts to matter more as each rank's slice of the domain gets smaller. It
does, exactly like you'd expect: efficiency at 32 ranks goes up
monotonically with grid size, 61% at 64×64, up to 96% at 512×512, then
drops slightly to 95% at 1024×1024 (that one uses fewer, longer
timesteps, so the numbers are a bit noisier). Small grids are
communication-bound, large ones aren't.

For the GPU side: same Kokkos source, recompiled for CUDA, no code
changes at all. One A100 sustains 2998 MLUPS on a 4096×4096 grid — about
143x a single CPU core — and scaling to 4 A100s gets to 10,432 MLUPS, 87%
efficiency, 3.48x speedup. That's over 10 GLUPS on 4 GPUs, from the same
hybrid Kokkos+MPI codebase that also runs, completely unmodified, on a
single laptop core.

That "no code changes" claim only holds because of the host-staged halo
exchange I mentioned earlier, and its cost is exactly why the GPU problem
size had to jump to 4096×4096 instead of the CPU run's 256×256 — at the
smaller size, 4 GPUs finish so fast that the extra copy per exchange
dominates and speedup actually falls below 1.

*(Transition: so the code works, it's fast, and it scales — what did that
actually cost me to learn, and what would I do differently? Discussion.)*

## 4. Discussion — what did I learn, what would I do differently? (≈1 min)

Honestly, getting the physics right — BGK collision, the equilibrium
distribution, bounce-back boundaries — was the easy 60% of this project.
The hard part was the bookkeeping at the parallelization boundaries:
ghost cells, global-vs-local coordinates, making sure a distributed run
gives results that match the serial reference to machine precision.
Writing the `--check-solver` cross-validation mode earlier would've saved
me a lot of time — I only added it after already chasing several halo
bugs by hand. The multi-size scaling sweep confirmed something I
suspected going in, that small grids are communication-bound at high rank
counts, but I didn't actually have a number for it until I ran the whole
five-size ladder.

What I'd do differently next time: build the correctness harness —
analytic tests, serial-vs-parallel diffing — before writing the parallel
kernel, not after. I'd also leave more cluster time for weird edge cases.
My 64-rank runs just hung — 18.6% CPU efficiency over a 7.5-minute
timeout, most ranks sitting idle — instead of degrading gracefully. Most
likely an MPI launch or binding issue at that process count rather than a
bug in the halo exchange itself, but I didn't have time to dig into it
this round, so the CPU story stops at a clean 1-32 rank curve.

Thanks — happy to take questions, including on the code itself.
