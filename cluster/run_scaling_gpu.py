#!/usr/bin/env python3
"""Run GPU strong-scaling measurements inside a Slurm compute allocation.

One MPI rank per GPU. Uses srun (not mpirun) so Slurm's own GPU binding
assigns each rank a distinct, exclusive device via --gpus-per-task; Kokkos
then sees exactly one visible device per rank and needs no extra selection
logic. Same timing protocol as cluster/run_scaling.py; the problem size is
deliberately much larger than the CPU run's 256x256, because per-step halo
overhead scales with ny alone while per-rank compute scales with
local_nx*ny -- at 256 columns split across 4 ranks (64 columns/rank) a
single A100 was so fast (~1.65s total) that communication overhead
dominated and speedup fell below 1. The default 4096x4096, 2000-step
problem targets ~40s on one A100 at the observed ~794 MLUPS, leaving each
rank enough owned columns for compute to dominate.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True,
                        help="e.g. build-cluster-gpu-a100/executables/milestone6")
    parser.add_argument("--arch", required=True, choices=["a100", "h100"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 4],
                        help="GPU/rank counts to test; must fit in the node's --gpus-per-node")
    parser.add_argument("--nx", type=int, default=4096)
    parser.add_argument("--ny", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit a cluster/strong_scaling_gpu_*.sbatch job; do not run simulations on a login node")
    executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    ranks_list = sorted(args.gpus)
    nx, ny, steps = args.nx, args.ny, args.steps
    metadata = dict(job_id=os.environ["SLURM_JOB_ID"], nodes=os.environ.get("SLURM_JOB_NODELIST"),
                    backend="cuda", gpu_arch=args.arch, nx=nx, ny=ny, steps=steps, repeats=3,
                    ranks=ranks_list, executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    launcher="srun --gpus-per-task=1", threads_per_rank=1)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "raw.csv").open("w", newline="") as file:
        fields = ["repeat", "ranks", "nx", "ny", "steps", "warmup", "seconds", "mlups"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for repeat in range(1, 4):
            # Alternate order to reduce correlation between process count and run order.
            counts = ranks_list if repeat % 2 else list(reversed(ranks_list))
            for ranks in counts:
                # --mpi=pmix_v5: srun's bare "--mpi=pmix" default alias fails its
                # handshake with this OpenMPI build and silently falls back to
                # launching N independent single-rank MPI singletons instead of
                # one N-rank job. Confirmed working via interactive salloc test
                # with pmix_v4 and pmix_v5 (srun --mpi=list shows both present);
                # pmix_v5 is the newer of the two.
                command = ["srun", "--ntasks", str(ranks), "--gpus-per-task=1", "--mpi=pmix_v5",
                           str(executable), str(nx), str(ny), "--steps", str(steps), "--benchmark"]
                run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=600)
                log = args.output / f"p{ranks}_repeat{repeat}.log"
                log.write_text("Command: " + " ".join(command) + "\n" + run.stdout)
                if run.returncode:
                    raise RuntimeError(f"Benchmark failed: {log}")
                match = re.search(r"^BENCHMARK (.+)$", run.stdout, re.MULTILINE)
                if not match:
                    raise RuntimeError(f"Missing benchmark record: {log}")
                values = dict(item.split("=", 1) for item in match[1].split())
                if (int(values["ranks"]), int(values["nx"]), int(values["ny"]), int(values["steps"])) != (ranks, nx, ny, steps):
                    raise RuntimeError("Benchmark configuration mismatch")
                writer.writerow(dict(repeat=repeat, **values))
                file.flush()
                print(f"repeat={repeat} ranks={ranks} seconds={values['seconds']}", flush=True)


if __name__ == "__main__":
    main()
