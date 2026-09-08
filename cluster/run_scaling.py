#!/usr/bin/env python3
"""Run strong-scaling measurements inside a Slurm compute allocation.

Sweeps several fixed problem sizes (not just one) so the communication-to-
computation ratio effect is visible in the data, per
https://pastewka.github.io/Accelerators/notes/scaling.html : at a fixed rank
count, a small grid gives each rank fewer owned columns relative to the two
halo columns it must exchange, so speedup should fall further below ideal
than it does for a large grid. Steps per size are chosen so the one-rank
run takes roughly the same wall time (~30-60s) across sizes, using the
milestone5 baseline of ~21 MLUPS as a rough single-core reference.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

# (nx=ny, steps): steps scaled so nx*ny*steps is roughly constant across sizes.
PROBLEM_SIZES = [(64, 150000), (128, 40000), (256, 20000), (512, 3000), (1024, 600)]


def run_one_size(executable, output, ranks_list, nx, ny, steps):
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(job_id=os.environ["SLURM_JOB_ID"], nodes=os.environ.get("SLURM_JOB_NODELIST"),
                    nx=nx, ny=ny, steps=steps, repeats=3, ranks=ranks_list,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    mapping="ppr:64:node", binding="core", threads_per_rank=1)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "raw.csv").open("w", newline="") as file:
        fields = ["repeat", "ranks", "nx", "ny", "steps", "warmup", "seconds", "mlups"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for repeat in range(1, 4):
            # Alternate order to reduce correlation between process count and run order.
            counts = ranks_list if repeat % 2 else list(reversed(ranks_list))
            for ranks in counts:
                command = ["mpirun", "-np", str(ranks), "--map-by", "ppr:64:node",
                           "--bind-to", "core", "--report-bindings", str(executable),
                           str(nx), str(ny), "--steps", str(steps), "--benchmark"]
                run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=300)
                log = output / f"p{ranks}_repeat{repeat}.log"
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
                print(f"nx=ny={nx} repeat={repeat} ranks={ranks} seconds={values['seconds']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-cluster/executables/milestone6"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size-index", type=int, default=None,
                        help="Run only PROBLEM_SIZES[index] (one Slurm array task = one problem size, "
                             "to fit under a partition's per-job time limit). Runs all sizes if omitted.")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit cluster/strong_scaling.sbatch; do not run simulations on a login node")
    executable = args.executable.resolve()
    # 64 ranks dropped: it hung (>300s timeout) on the nx=ny=256 problem while
    # 1-32 completed in seconds -- see cluster/README.md for the investigation.
    ranks_list = [1, 2, 4, 8, 16, 32]
    sizes = PROBLEM_SIZES if args.size_index is None else [PROBLEM_SIZES[args.size_index]]
    for nx, steps in sizes:
        run_one_size(executable, args.output / f"size_{nx}x{nx}", ranks_list, nx, nx, steps)


if __name__ == "__main__":
    main()
