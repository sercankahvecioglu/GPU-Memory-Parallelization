#!/usr/bin/env python3
"""Run strong-scaling measurements inside a Slurm compute allocation."""
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
    parser.add_argument("--executable", type=Path, default=Path("build-cluster/executables/milestone6"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit cluster/strong_scaling.sbatch; do not run simulations on a login node")
    executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # 64 dropped: it hung (>300s timeout) on this problem size while 1-32
    # completed in seconds -- see cluster/README.md for the investigation.
    ranks_list = [1, 2, 4, 8, 16, 32]
    nx = ny = 256
    steps = 20000
    metadata = dict(job_id=os.environ["SLURM_JOB_ID"], nodes=os.environ.get("SLURM_JOB_NODELIST"),
                    nx=nx, ny=ny, steps=steps, repeats=3, ranks=ranks_list,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    mapping="ppr:64:node", binding="core", threads_per_rank=1)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "raw.csv").open("w", newline="") as file:
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
