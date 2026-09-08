#!/usr/bin/env python3
"""Run MPI/serial correctness comparisons and retain reproducible evidence."""
import argparse
import csv
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/executables/milestone6"))
    parser.add_argument("--output", type=Path, default=Path("milestone6_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Same physical problem at each process count; also exercise uneven strips.
    cases = [(nx, ny, steps, ranks)
             for nx, ny, steps in [(128, 128, 1000), (17, 13, 200)]
             for ranks in [1, 2, 3, 4]]
    cases.append((3, 5, 25, 3))  # One owned column per rank.
    for nx, ny, steps, ranks in cases:
        command = ["mpiexec", "-n", str(ranks), str(args.executable.resolve()),
                   str(nx), str(ny), "--steps", str(steps), "--check-solver"]
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=120)
        log_name = f"{nx}x{ny}_{steps}steps_{ranks}ranks.log"
        (args.output / log_name).write_text("Command: " + " ".join(command) + "\n" + run.stdout)
        if run.returncode:
            raise RuntimeError(f"MPI validation failed; see {args.output / log_name}")
        match = re.search(
            r"Serial comparison passed: maximum population error=(\S+) "
            r"mass_relative_error=(\S+) energy_absolute_error=(\S+) "
            r"serial_mass=(\S+) serial_energy=(\S+)", run.stdout)
        if not match:
            raise RuntimeError(f"Missing serial comparison result in {log_name}")
        row = dict(nx=nx, ny=ny, steps=steps, ranks=ranks,
                   maximum_population_error=float(match[1]),
                   mass_relative_error=float(match[2]),
                   energy_absolute_error=float(match[3]),
                   serial_mass=float(match[4]), serial_energy=float(match[5]),
                   status="PASS", log=log_name)
        rows.append(row)
        print(f"{nx}x{ny}, {steps} steps, {ranks} ranks: PASS; "
              f"population error {float(match[1]):.3e}", flush=True)
    with (args.output / "validation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"All {len(rows)} cases passed. Results: {args.output / 'validation.csv'}")


if __name__ == "__main__":
    main()
