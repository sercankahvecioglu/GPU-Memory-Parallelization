#!/bin/bash -l
#SBATCH --job-name=yalb-scaling
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=strong-scaling-%j.out
set -euo pipefail
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0.8-gnu-14.2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' build-cluster/CMakeCache.txt
result_dir="scaling-results/$SLURM_JOB_ID"
mkdir -p "$result_dir"
module list > "$result_dir/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c 'hostname; lscpu' > "$result_dir/hardware.txt"
cp build-cluster/CMakeCache.txt "$result_dir/CMakeCache.txt"
python3 cluster/run_scaling.py --output "$result_dir"
