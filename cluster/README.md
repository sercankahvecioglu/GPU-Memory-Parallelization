# Strong scaling on bwUniCluster

`cluster/run_scaling.py` sweeps five fixed problem sizes -- 64x64, 128x128,
256x256, 512x512, 1024x1024 -- each with its own fixed step count (chosen so
every size's one-rank run takes roughly the same wall time), omega=1/0.596,
lid velocity=0.1 (Re=800). Counts: 1,2,4,8,16,32 MPI ranks, three repeats
per count, per size. Kokkos Serial uses one CPU core per rank. All counts run on one exclusive dev_cpu node, with up to 64 bound cores. Core bindings
are retained in the raw logs. The Slurm allocation is exclusive to reduce
interference from other jobs.

The sweep across sizes (not just 256x256) exists specifically to show the
communication-to-computation ratio effect described in
https://pastewka.github.io/Accelerators/notes/scaling.html : at a fixed rank
count, a small grid gives each rank fewer owned columns relative to the two
halo columns it exchanges every step, so its measured speedup should sit
further below the ideal line than a large grid's does, and should improve as
the grid grows. Plot all five together with `plot_scaling_multisize.py`
(below) to see this directly, rather than picking one problem size and
assuming it generalizes.

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

`dev_cpu`'s `MaxTime` is 00:30:00 (confirmed via `scontrol show partition
dev_cpu`), too tight to trust for all five sizes' rank sweeps in one job. So
`strong_scaling.sbatch` submits a Slurm array instead -- one task per problem
size (`cluster/run_scaling.py --size-index $SLURM_ARRAY_TASK_ID`), each
requesting the full 00:30:00 (a shorter request, e.g. 00:15:00, was rejected
outright with "Requested time limit is invalid" -- `dev_cpu` uses `QoS=dev`,
and while `sacctmgr show qos dev` shows no separate `MaxWall`, in practice
only the full partition MaxTime was accepted). All tasks in an array share one
results directory keyed by `$SLURM_ARRAY_JOB_ID` (constant across the array),
not `$SLURM_JOB_ID` (which differs per task); per-task hardware/module/CMake
snapshots are suffixed `_task0`, `_task1`, ... so concurrently-scheduled tasks
never race on the same file.

QOS `dev` also caps `MaxSubmitJobsPerUser` at 4 (`sacctmgr show qos dev`), and
each array task counts as a separate submitted job, so a single `--array=0-4`
(5 tasks, one per `PROBLEM_SIZES` entry) is rejected. The script's `#SBATCH
--array` is therefore `0-3` (sizes 64/128/256/512); submit the remaining size
(index 4, 1024x1024) separately once that array finishes, overriding
`--array` on the command line:

```bash
sbatch --array=4-4 cluster/strong_scaling.sbatch
```

This lands under a different `$SLURM_ARRAY_JOB_ID`, so copy both result
directories locally (see below) before plotting with
`plot_scaling_multisize.py`, which reads all `size_NxN` subdirectories under
whatever path you point it at regardless of which job produced them.

When transferring changes to the cluster, sync all of `cluster/*.py` together
with the `.sbatch` files -- syncing only the `.sbatch` once left a stale
`cluster/run_scaling.py` on the cluster (missing `--size-index`), which failed
every array task with `unrecognized arguments: --size-index N` before the
mismatch was caught.

Build on the cluster after transferring the project sources:

```bash
bash cluster/build.sh
sbatch cluster/strong_scaling.sbatch
squeue -u "$USER"
# after it completes:
sbatch --array=4-4 cluster/strong_scaling.sbatch
squeue -u "$USER"
```

Never launch benchmark simulations on a login node. The Slurm job launches
all simulations within its allocated CPU node. Compilation uses GNU 14.2 and
OpenMPI 5.0.8 through mpicc/mpicxx, Release mode, and the Kokkos Serial backend.

Each executable run performs 100 untimed warmup timesteps, resets the fluid,
synchronizes ranks, and times that size's fixed step count using MPI_Wtime. Kokkos
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

Results are written under scaling-results/ARRAY_JOB_ID/size_NxN on the
cluster, one subdirectory per problem size (ARRAY_JOB_ID is the number
`sbatch` prints, shared by all tasks in that array). Because the two
`sbatch` calls above (`--array=0-3` then `--array=4-4`) produce two different
ARRAY_JOB_IDs, copy each `size_NxN` subdirectory into one shared local folder
rather than two separate ARRAY_JOB_ID folders, e.g.:

```bash
mkdir -p milestone6_results/scaling/strong_scaling_all_sizes
scp -r cluster:.../scaling-results/FIRST_ARRAY_JOB_ID/size_* milestone6_results/scaling/strong_scaling_all_sizes/
scp -r cluster:.../scaling-results/SECOND_ARRAY_JOB_ID/size_1024x1024 milestone6_results/scaling/strong_scaling_all_sizes/
```

`plot_scaling_multisize.py` only reads `size_NxN` subdirectories, so it does
not care which job produced each one. Plot one size the usual way:

```bash
.venv/bin/python plot_scaling.py milestone6_results/scaling/ARRAY_JOB_ID/size_256x256
```

or overlay all five sizes on one log-log speedup plot to see the
communication-to-computation effect:

```bash
.venv/bin/python plot_scaling_multisize.py milestone6_results/scaling/ARRAY_JOB_ID
```

The plotting script requires matplotlib. It produces strong_scaling.png,
strong_scaling.svg, strong_scaling.pdf, and summary.csv. It refuses incomplete
runs or mixed grid/step configurations. Raw timing CSV, per-run logs, node/CPU
metadata, modules and CMake cache are retained for reproducibility.

The original milestone 5 MLUPS is not used as the speedup denominator: its
kernel organization differs. The one-rank milestone6 executable is the matching
baseline for this strong-scaling comparison.

## Multisize results (bwUniCluster, job 6823862 + 6823895)

The five-size sweep described above has been run to completion: sizes
64/128/256/512 came from array job `6823862` (`--array=0-3`), and 1024x1024
from the follow-up `6823895` (`--array=4-4`), merged locally under
`milestone6_results/scaling/strong_scaling_all_sizes/`. Measured efficiency
at 32 ranks (`plot_scaling.py` per size):

| Grid size | Efficiency @ 32 ranks |
|---|---|
| 64x64     | 61.2% |
| 128x128   | 81.9% |
| 256x256   | 92.4% |
| 512x512   | 96.0% |
| 1024x1024 | 94.8% |

This confirms the communication-to-computation prediction: 64x64 falls
clearly furthest below ideal, and 256x256/512x512 track the ideal line most
closely. 1024x1024 is slightly noisier than 512x512 at intermediate rank
counts (16 ranks: 83.8% vs. 512x512's 97.8%) -- plausibly because its step
count (600, vs. 20 000-150 000 for the smaller sizes) leaves less averaging
of per-run overhead, rather than a real regression in the comm/comp trend.
See `milestone6_results/scaling/strong_scaling_all_sizes/strong_scaling_multisize.{png,svg,pdf}`
for the overlay plot.

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
