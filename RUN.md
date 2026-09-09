# Running YALB — one command chain per milestone

One CMake project, one build, four things you can run independently
afterward. All commands assume you're in the project root.

## 0. Build once

Local machine (needs a C++17 compiler and an MPI implementation; Kokkos
and Eigen are fetched automatically if not already installed):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

On the cluster, use the module-loading wrapper instead (produces
`build-cluster/` instead of `build/`):

```bash
bash cluster/build.sh
```

Everything below uses `build/`; swap in `build-cluster/` if you built on
the cluster.

---

## Milestone 3 — streaming & boundary primitives

No standalone executable — validated as unit tests against hand-computed
reference states.

```bash
ctest --test-dir build -R "StreamingTest|MacroscopicFieldsTest|WallStreamingTest" --output-on-failure
```

Or run the test binary directly for verbose gtest output:

```bash
./build/tests/tests --gtest_filter="StreamingTest.*:MacroscopicFieldsTest.*:WallStreamingTest.*"
```

---

## Milestone 4 — shear-wave viscosity validation

```bash
./build/executables/milestone4 milestone4_results
python3 plot_milestone4.py milestone4_results
```

Output: `milestone4_results/viscosity_vs_omega.csv` (+ `.svg` plot),
`log_amplitude_fit.svg`, `velocity_profiles.csv`. Console prints measured
vs. theoretical viscosity per omega.

---

## Milestone 5 — lid-driven cavity (single core)

```bash
./build/executables/milestone5 milestone5_results
python3 plot_milestone5.py milestone5_results
```

Output: `milestone5_results/summary.txt`, `mlups_report.txt`,
`cavity_fields.csv`, `centerline_ux.csv`, plus the plotted `.svg` files
(velocity field, vectors, centerline vs. Ghia et al. 1982) and
`benchmark_summary.txt` (RMSE against the reference profile).

---

## Milestone 6 — distributed MPI solver

**Correctness check** (small grid, compares every rank's slice against a
full serial reference — the fastest way to confirm the build is correct):

```bash
mpirun -np 4 build/executables/milestone6 64 64 --steps 100 --check-solver
```

Expect `Serial comparison passed: ...` in the output.

**Halo-exchange communication test** (packing/unpacking correctness,
no physics):

```bash
mpirun -np 3 build/executables/milestone6 7 5 --check-halos
```

**A real run to steady state** (prints mass/energy diagnostics every 100
steps):

```bash
mpirun -np 4 build/executables/milestone6 128 128 --steps 5000
```

**MLUPS benchmark** (throughput only, no diagnostics/validation —
what the strong-scaling sweeps use under the hood):

```bash
mpirun -np 4 build/executables/milestone6 256 256 --steps 20000 --benchmark
```

**Full milestone 6 regression suite** (all rank/grid/step combinations
defined in `executables/CMakeLists.txt`):

```bash
ctest --test-dir build -R "Milestone6" --output-on-failure
```

**Cluster strong-scaling sweep** (CPU, 1-32 ranks, five grid sizes —
see `cluster/README.md` for the array-job details and time-limit
workarounds):

```bash
bash cluster/build.sh
sbatch cluster/strong_scaling.sbatch
squeue -u "$USER"
```

**Cluster GPU strong-scaling sweep** (A100/H100, 1/2/4 GPUs — see
`cluster/README.md` for CUDA-aware MPI caveats before running):

```bash
bash cluster/build_gpu.sh a100
sbatch cluster/strong_scaling_gpu_a100.sbatch
```

After either sweep completes, copy `scaling-results*/` back locally and
plot with:

```bash
python3 plot_scaling.py milestone6_results/scaling/<JOB_ID>/size_256x256
python3 plot_scaling_multisize.py milestone6_results/scaling/<JOB_ID>
```
