#!/bin/bash -l
#SBATCH --job-name=yalb-scaling
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=6G
# dev_cpu caps MaxTime at 00:30:00. Too tight to trust for all 5 problem sizes
# in one job, so run one array task per size (see cluster/run_scaling.py
# --size-index) so each task's own 1..32-rank x 3-repeat sweep -- a few
# minutes at most -- fits comfortably. Requesting the full 00:30:00 rather
# than a shorter value since this partition rejected --time=00:15:00 with
# "Requested time limit is invalid (missing or exceeds some limit)",
# suggesting a QOS/partition MinTime pinned at 30 minutes on this cluster
# generation (bwUniCluster 3.0, uc3n* login nodes).
# QOS `dev` on this partition caps MaxSubmitJobsPerUser at 4 (confirmed via
# `sacctmgr show qos dev`), and each array task counts as a separate submitted
# job -- so --array=0-4 (5 tasks) is rejected. This covers PROBLEM_SIZES[0:4]
# (64/128/256/512); submit the remaining size (index 4, 1024x1024) once this
# array finishes, overriding --array on the command line (which takes
# precedence over the #SBATCH directive below):
#   sbatch --array=4-4 cluster/strong_scaling.sbatch
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=strong-scaling-%A_%a.out
set -euo pipefail
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0.8-gnu-14.2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' build-cluster/CMakeCache.txt
# SLURM_ARRAY_JOB_ID is shared by every task in the array; SLURM_JOB_ID differs
# per task. Using the array job id keeps all five sizes under one results dir.
result_dir="scaling-results/$SLURM_ARRAY_JOB_ID"
mkdir -p "$result_dir"
# Per-task filenames avoid concurrent array tasks racing on the same file.
module list > "$result_dir/modules_task${SLURM_ARRAY_TASK_ID}.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job_task${SLURM_ARRAY_TASK_ID}.txt"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c 'hostname; lscpu' > "$result_dir/hardware_task${SLURM_ARRAY_TASK_ID}.txt"
cp build-cluster/CMakeCache.txt "$result_dir/CMakeCache_task${SLURM_ARRAY_TASK_ID}.txt"
python3 cluster/run_scaling.py --output "$result_dir" --size-index "$SLURM_ARRAY_TASK_ID"
