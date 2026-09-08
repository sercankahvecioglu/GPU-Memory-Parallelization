# Strong scaling on bwUniCluster

The fixed problem is 256x256 cells, 20,000 timesteps, omega=1/0.596,
lid velocity=0.1 (Re=800). Counts: 1,2,4,8,16,32 MPI ranks, three repeats
per count. Kokkos Serial uses one CPU core per rank. All counts run on one exclusive dev_cpu node, with up to 64 bound cores. Core bindings
are retained in the raw logs. The Slurm allocation is exclusive to reduce
interference from other jobs.

64 ranks was tried and dropped: it ran cleanly through 1-32 ranks (each
finishing in seconds) then hung and hit `run_scaling.py`'s 300s subprocess
timeout, with the job's overall CPU efficiency (18.57% over 07:32
wall-clock) consistent with most ranks sitting idle rather than computing.
The jump from 2.1s at 32 ranks to a full timeout at 64 rules out gradual
scaling loss -- it is a hang, most likely in the `mpirun --map-by
ppr:64:node` launch/binding path at that scale rather than in
`HaloExchange::exchange` itself, which uses deadlock-safe `MPI_Sendrecv`
pairs with a buffer size independent of rank count. Not investigated
further since 1-32 already gives a complete, clean strong-scaling curve.

Build on the cluster after transferring the project sources:

```bash
bash cluster/build.sh
sbatch cluster/strong_scaling.sbatch
squeue -u "$USER"
```

Never launch benchmark simulations on a login node. The Slurm job launches
all simulations within its allocated CPU node. Compilation uses GNU 14.2 and
OpenMPI 5.0.8 through mpicc/mpicxx, Release mode, and the Kokkos Serial backend.

Each executable run performs 100 untimed warmup timesteps, resets the fluid,
synchronizes ranks, and times 20,000 solver timesteps using MPI_Wtime. Kokkos
fences ensure completion. MPI_MAX chooses the slowest rank's elapsed time.
Timing includes collision, halo communication, pull streaming and field updates,
but excludes initialization, diagnostics, validation, process startup and output.
Finite-state diagnostics run after timing. This is solver-only throughput,
not full application wall time or a steady-state convergence measurement.

Each run starts from the same equilibrium state. The process-count order is
reversed in the second repetition. Speedup uses the median one-rank time from
the same executable divided by each process count's median time. Efficiency is
speedup/process count. Plot error bars show the min-to-max speedup range from
runtime repeats with the median baseline held fixed (not confidence intervals).

Results are written under scaling-results/JOBID on the cluster. Copy that
folder to milestone6_results/scaling/JOBID locally and plot it with:

```bash
.venv/bin/python plot_scaling.py milestone6_results/scaling/JOBID
```

The plotting script requires matplotlib. It produces strong_scaling.png,
strong_scaling.svg, strong_scaling.pdf, and summary.csv. It refuses incomplete
runs or mixed grid/step configurations. Raw timing CSV, per-run logs, node/CPU
metadata, modules and CMake cache are retained for reproducibility.

The original milestone 5 MLUPS is not used as the speedup denominator: its
kernel organization differs. The one-rank milestone6 executable is the matching
baseline for this strong-scaling comparison.

## GPU strong scaling (A100 / H100)

The Kokkos CUDA backend needs no source changes: `HaloExchange::exchange`
already stages packed halo data through host mirrors before every MPI call
(`executables/milestone6.cpp`), so it runs correctly under ordinary MPI on
GPU without requiring CUDA-aware MPI. It is a portability choice, not a
performance-tuned one -- the host round trip costs an extra device-to-host
and host-to-device copy per exchange that a CUDA-aware MPI build could skip.

Build separately per GPU generation (compiling for the wrong architecture
flag works via forward compatibility but is not optimal):

```bash
bash cluster/build_gpu.sh a100
bash cluster/build_gpu.sh h100
sbatch cluster/strong_scaling_gpu_a100.sbatch
sbatch cluster/strong_scaling_gpu_h100.sbatch
```

Each job requests one exclusive node with 4 GPUs and runs 1, 2, 4 ranks
(one MPI rank per GPU, via `srun --gpus-per-task=1` so Slurm -- not
Kokkos -- assigns each rank its own device), 3 repeats, same 256x256 grid
and 20,000-step problem as the CPU run, so CPU and GPU speedup curves are
directly comparable once both are plotted. Extending past 4 GPUs needs a
second node and `--gpus-per-task`-aware launch across nodes, which these
scripts do not attempt.

**The partition names in the two `.sbatch` files (`gpu_a100_il`,
`gpu_h100`) and the `devel/cuda/12.8` module were assembled from
documentation, not confirmed in a live bwUniCluster session.** Before
submitting, check what actually exists:

```bash
sinfo -o "%20P %.5D %.4c %G"
module avail cuda
module avail mpi
```

and adjust `#SBATCH --partition=` and the `module load` line accordingly.

Results land under `scaling-results-gpu/<arch>-<JOBID>/`. Copy each to
e.g. `milestone6_results/scaling/gpu-a100-<JOBID>` locally and plot with
the same `plot_scaling.py` used for the CPU run -- the title and x-axis
label pick up the GPU generation automatically from `metadata.json`.

References:
- https://pastewka.github.io/Accelerators/notes/bwUniCluster.html
- https://pastewka.github.io/Accelerators/notes/scaling.html
- https://pastewka.github.io/Accelerators/notes/kokkos_gpu_compilation.html
