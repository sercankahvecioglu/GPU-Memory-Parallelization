# yalb — a D2Q9 Lattice Boltzmann Solver

[![CMake](https://github.com/sercankahvecioglu/yalb/actions/workflows/cmake.yml/badge.svg)](https://github.com/sercankahvecioglu/yalb/actions/workflows/cmake.yml)

A two-dimensional lattice Boltzmann fluid solver (D2Q9, BGK collision) built
incrementally across six milestones for a high-performance-computing course.
On-node parallelism uses [Kokkos](https://kokkos.org/) (Serial and CUDA
backends have been exercised); distributed-memory parallelism across nodes or
GPUs uses [MPI](https://www.mpi-forum.org/) with a 1-D strip domain
decomposition.

The solver is validated against the shear-wave decay analytical solution
(Milestone 4), the Ghia, Ghia & Shin (1982) lid-driven-cavity benchmark
(Milestone 5), and a serial-vs-distributed cross-check (Milestone 6), and its
throughput is reported in MLUPS (million lattice updates per second) on both
CPU and GPU.

## Requirements

* CMake >= 3.14
* A C++17 compiler
* [Kokkos](https://kokkos.org/) 4.7 (fetched automatically if not found)
* [Eigen3](https://eigen.tuxfamily.org/) 5.0 (fetched automatically if not found)
* An MPI implementation (e.g. OpenMPI)
* Python 3 with `numpy` and `matplotlib` for the plotting/animation scripts

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On bwUniCluster, load the toolchain first:

```bash
module load compiler/gnu mpi/openmpi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Use `-DCMAKE_BUILD_TYPE=Debug` while developing; always benchmark or run
production simulations with `Release`.

## Repository layout

```
src/            core library (D2Q9 kernels: streaming, collision, equilibrium, ...)
include/        public header for the library (lbm.hpp)
executables/    milestone entry points (main.cpp, milestone4/5/6.cpp)
tests/          GoogleTest unit tests, run via ctest
cluster/        bwUniCluster job scripts and strong-scaling sweep driver
milestone*_results/   generated CSV/plot/animation output per milestone
plot_*.py, animate_*.py, validate_milestone6.py   post-processing scripts
```

## Milestones

### Milestone 1-2: project setup and the streaming operator

The D2Q9 lattice constants, density/velocity moments, and the pure streaming
operator (`streaming` in `src/lbm.cpp`) are implemented and unit-tested in
`tests/test_streaming.cpp`: a localized population "blob" translates one
lattice site per step without distortion, conserves total mass to machine
precision, and re-emerges at its starting point after wrapping around the
periodic domain.

```bash
python3 animate_milestone2.py
```

reproduces that exact test setup in NumPy and renders the traveling velocity
field to `milestone2_results/velocity_field_evolution.gif`.

### Milestone 3: the BGK collision operator

The equilibrium distribution (`compute_f_eq`) and BGK relaxation
(`collision`) are implemented in `src/lbm.cpp`. Combined with streaming, a
central density perturbation on an otherwise uniform, periodic domain
relaxes and spreads outward as a damped sound wave. This behavior is
reproduced with:

```bash
python3 animate_milestone3.py
```

which writes `milestone3_results/density_wave_evolution.gif`. Collision
correctness is additionally exercised end-to-end by the Milestones 4 and 5
validations below.

### Milestone 4: shear-wave viscosity validation

```bash
cmake --build build --target milestone4
./build/executables/milestone4 milestone4_results
```

Initializes `rho = 1`, `ux(y) = epsilon * sin(2*pi*y/Ny)` and evolves it for
`omega = 0.6, 0.8, 1.0, 1.2, 1.4`. Amplitude decay is measured by projecting
the velocity profile onto its initial sine mode; the measured viscosity
`-slope / (2*pi/Ny)^2` is compared against the theoretical
`(1/omega - 1/2)/3`. Outputs: `velocity_profiles.csv`,
`amplitude_omega_*.csv`, `viscosity_vs_omega.csv`.

### Milestone 5: lid-driven cavity and single-process MLUPS baseline

```bash
cmake --build build --target milestone5
./build/executables/milestone5 milestone5_results
python3 plot_milestone5.py milestone5_results
python3 plot_milestone5_mlups.py
```

Stationary bounce-back on the left/right/bottom walls, moving-wall
bounce-back on the lid, run to convergence (max velocity change below
`1e-10`, checked every 100 steps). The saved Release/Kokkos-Serial run at
Re = 400 converged after 83,200 steps; against the 17 Ghia et al. (1982)
reference points, the normalized RMSE is 0.003645 and the maximum absolute
error is 0.008269 (0.827% of lid speed), with `benchmark_summary.txt` and
`benchmark_comparison.csv` holding the full comparison.

Solver-only throughput on this machine (Kokkos Serial, one process, 2,000
timed steps, 7 repeats, `Kokkos::fence()`-bracketed timing that excludes
convergence checks and file I/O): **median 18.38 MLUPS** (range
18.08-18.54), recorded in `milestone5_results/mlups_report.txt` and plotted
in `milestone5_results/mlups_benchmark.png`. This number is the
single-process baseline carried into Milestone 6.

### Milestone 6: distributed-memory (MPI) solver and scaling

`executables/milestone6.cpp` decomposes the nonperiodic x-dimension into
contiguous strips (one MPI rank per strip, every rank owning the full y
extent), with a 1-column ghost region on each side. Only the 3 of 9
populations that actually cross a boundary are exchanged, once per
timestep, via a deadlock-safe pair of `MPI_Sendrecv` calls.

```bash
cmake --build build --target milestone6
mpiexec -n 3 ./build/executables/milestone6 128 128 --steps 1000
mpiexec -n 3 ./build/executables/milestone6 17 13 --steps 200 --check-solver
ctest --test-dir build -R Milestone6 --output-on-failure
```

**Correctness.** `--check-solver` independently runs the reference serial
kernels on every rank and compares owned populations plus global
mass/energy against the distributed run. All nine configurations checked
(1/2/3/4 ranks x several grid shapes, including single-column strips) pass,
with maximum population error 8.88e-16. Reproduce with
`python3 validate_milestone6.py`; see
[`milestone6_results/validation.md`](milestone6_results/validation.md).

**Strong-scaling performance**, measured on bwUniCluster (full procedure and
raw data in [`cluster/README.md`](cluster/README.md)):

| Backend | Grid | Ranks | MLUPS (median) | Efficiency |
|---|---|---|---|---|
| CPU (Kokkos Serial) | 256x256 | 1 | 20.98 | - |
| CPU (Kokkos Serial) | 256x256 | 32 | 616.97 | 91.9% |
| GPU (Kokkos CUDA, A100) | 4096x4096 | 1 | 2998.01 | - |
| GPU (Kokkos CUDA, A100) | 4096x4096 | 4 | 10432.21 | 87.0% |

At matched throughput comparison, a single A100 GPU (2998 MLUPS) reaches
about 143x the MLUPS of a single CPU core (20.98 MLUPS) for this solver.
Plots: `plot_scaling.py` (speedup/efficiency per run directory),
`plot_scaling_multisize.py` (communication-to-computation ratio across five
grid sizes), `plot_mlups_scaling.py` and `plot_milestone6_gpu_mlups.py`
(MLUPS vs. rank/GPU count) — see `milestone6_results/` for generated
figures.

## License

MIT — see [LICENSE](LICENSE).
