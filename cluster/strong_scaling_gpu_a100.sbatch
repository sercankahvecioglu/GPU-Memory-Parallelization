#!/bin/bash -l
#SBATCH --job-name=yalb-scaling-a100
# NOTE: verify this partition name (`sinfo -o "%20P %.5D %.4c %G"`) before
# submitting -- gpu_a100_il / dev_gpu_a100_il were not confirmed live.
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=strong-scaling-gpu-a100-%j.out
set -euo pipefail
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0.8-gnu-14.2 devel/cuda/12.8
cd "$SLURM_SUBMIT_DIR"
build_dir="build-cluster-gpu-a100"
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' "$build_dir/CMakeCache.txt"
result_dir="scaling-results-gpu/a100-$SLURM_JOB_ID"
mkdir -p "$result_dir"
module list > "$result_dir/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm-job.txt"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 bash -c 'hostname; nvidia-smi -L' > "$result_dir/hardware.txt"
cp "$build_dir/CMakeCache.txt" "$result_dir/CMakeCache.txt"
python3 cluster/run_scaling_gpu.py --arch a100 --executable "$build_dir/executables/milestone6" \
    --gpus 1 2 4 --output "$result_dir"
