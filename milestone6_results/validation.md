# MPI versus serial validation

All nine cases passed using a Release build with the Kokkos Serial backend under
WSL. Each distributed run was compared with the existing full-domain collision
and push-streaming-with-walls kernels from src/lbm.cpp, using the same grid,
initial equilibrium, omega, lid velocity, corner convention, and timestep count.
All nine populations at every owned cell were compared; MPI_MAX combines the
largest local difference. Global mass and kinetic energy were also checked.
Ghosts are excluded from the comparison and from diagnostic sums.

| Grid | Steps | MPI processes | Maximum absolute population difference |
|---|---:|---|---:|
| 128 x 128 | 1000 | 1, 2, 3, 4 | 8.8818e-16 |
| 17 x 13 | 200 | 1, 2, 3, 4 | 4.9960e-16 |
| 3 x 5 | 25 | 3 | 4.9960e-16 |

Across all cases, the largest relative mass difference against the serial
reference was 4.3299e-15; the largest absolute kinetic-energy difference was
9.3259e-15. These are consistent with floating-point rounding, including
summation-order changes in collective reductions. Results agree to numerical
precision, without requiring bitwise equality.

The 128-column case includes an uneven 3-process split. The 17-column case has
uneven splits for 2, 3, and 4 processes. The 3-column case gives each rank only
one owned column, exercising both neighboring halos on the middle rank.

Acceptance criteria in check_against_serial:
- Maximum absolute population difference <= 1e-12.
- Relative mass difference against serial <= 1e-11.
- Absolute energy difference <= 1e-11 * max(1, serial energy).
- Mass drift relative to the initial NX*NY <= 1e-11.

The full existing CTest suite also passed: 24/24 tests.
This validates the tested configurations; it is not a GPU validation,
steady-state study, or scaling measurement.

Reproduce from the repository root:

```bash
cmake --build build --target milestone6 -j 2
python3 validate_milestone6.py
ctest --test-dir build --output-on-failure
```

See validation.csv for exact errors per run and the corresponding .log files
for commands and complete output. The optional --check-solver mode runs a
full-domain serial reference on every rank for checking; production runs do not.
