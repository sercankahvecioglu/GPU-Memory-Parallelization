# CMake skeleton code

## Milestone 5: lid-driven cavity

Build and run the separate Milestone 5 executable with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone5
./build/executables/milestone5 milestone5_results
```

The simulation uses stationary bounce-back on the left, right, and bottom
walls, and moving-wall bounce-back on the interior of the top lid. It stops
when the maximum velocity change falls below the convergence limit and writes
the full field, vertical centerline profile, Reynolds number, runtime, and
MLUPS performance to `milestone5_results`.

Create the velocity-vector, speed, and centerline plots with:

```bash
python3 plot_milestone5.py milestone5_results
```

The convergence limit is `1e-10` for the maximum velocity change over one
step, checked every 100 steps. The stricter limit avoids stopping while slow
transients still affect the benchmark profile. Runs that reach the step limit
without converging fail rather than reporting a steady state. The vertical
centerline averages the two central columns to sample the physical cavity center.

The saved Release/Kokkos Serial run at Re = 400 converged after 83,200 steps.
Against the 17 Ghia reference points, the normalized RMSE is 0.003645 and the
maximum absolute error is 0.008269 (0.827% of lid speed). The maximum relative
error over nonzero reference velocities is 4.16%; relative error at zero is
undefined. These metrics are recorded in `benchmark_summary.txt`, with the
pointwise comparison in `benchmark_comparison.csv`. The plotting script checks
that the simulation Reynolds number matches the bundled Re = 400 reference.

The solver-only performance baseline is 19.44 MLUPS over 2,000 steps on this
machine (Release, Kokkos Serial, one process). Timing includes synchronization
and excludes convergence checks and file output; performance varies by hardware.
The fields are saved after these additional benchmark steps.

## Milestone 4: shear-wave viscosity validation

Build and run the validation with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/executables/milestone4 milestone4_results
```

The optional argument is the output directory. The run verifies the initialized
density and velocity fields, simulates `omega = 0.6, 0.8, 1.0, 1.2, 1.4`, and
writes CSV data for plotting. The main outputs are:

- `velocity_profiles.csv`: x-averaged shear-wave profiles at selected times;
- `amplitude_omega_*.csv`: amplitude decay and `ln|A(t)|`;
- `viscosity_vs_omega.csv`: measured viscosity, theoretical viscosity, fit
  coefficients, and relative errors.

Amplitude is measured by projecting the velocity profile onto its initial sine
mode. The measured viscosity is `-slope / (2*pi/Ny)^2`, while the theoretical
value is `(1/omega - 1/2)/3`.

This repository contains a [CMake](https://cmake.org/) skeleton for C++ projects. It has provision
for dependencies on the numerical libraries

* [Eigen3](https://eigen.tuxfamily.org/)
* [Kokkos](https://kokkos.org/)
* [MPI](https://www.mpi-forum.org/)

## Getting started

Click the `Use this template` button above, then clone the newly created
repository.

### Compiling using CLion

> Note: for Windows users, please follow [these
> instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-product.html)
> in addition to the text below.

If you are using CLion, you can open your freshly cloned project by clicking on
the "Open" button in the CLion welcome window. If prompted, trust the project.

You can change CMake build option in "**File > Settings > Build, Execution, Deployment > CMake**".
This windows allows to set the `CMAKE_BUILD_TYPE` option, which controls the level of optimization applied to
the code. `Debug` disables optimizations and turns on useful debugging features.
This mode should be used when developing and testing code.
`Release` turns on aggressive optimization. This mode should be used when
running production simulations. Add `-DCMAKE_BUILD_TYPE=Release` or `-DCMAKE_BUILD_TYPE=Debug` to "CMake options"
to switch between the two.

To run the executable, click on the dialog directly right of the
green hammer in the upper right toolbar, select "main", and click
the green arrow right of that dialog. You should see the output in the "Run"
tab, in the lower main window.

To run the tests, select "tests" in the same dialog, then run. In the lower
window, on the right, appears a panel that enumerates all the tests that were
run and their results.

Try compiling and running for both `Debug` and `Release` configurations. Don't
forget to switch between them when testing code or running production simulations.

### Compiling from the command line

The command line (terminal) may look daunting at first, but it has the advantage
of being the same across all UNIX platforms, and does not depend on a specific
IDE. The standard CMake workflow is to create a `build/` directory which will
contain all the build files. To do that, and compile your project, run:

```bash
cd <your repository>

# Configure and create build directory
cmake -B build

# Compile
cmake --build build

# Run executable and tests
./build/executables/main
cd build && ctest
```

Note that CLion is by default configured to create a `cmake-build-debug/` directory.

If there are no errors then you are all set! Note that the flag
`-DCMAKE_BUILD_TYPE=Debug` should be changed to
`-DCMAKE_BUILD_TYPE=Release` when you run a production simulation, i.e. a
simulation with more than a few hundred atoms. This turns on aggressive compiler
optimizations, which results in speedup. However, when writing the code and
looking for bugs, `Debug` should be used instead.

Try compiling and running tests with both compilation configurations.

### Compiling on bwUniCluster, with MPI

The above steps should be done *after* loading the appropriate packages:

```bash
module load compiler/gnu mpi/openmpi

# configure
cmake -B build -DCMAKE_BUILD_TYPE=Release
# compile
cmake --build build
```

## How to add code to the repository

There are three places where you are asked to add code:

- `src/` is the core of the code. Code common to all executables and tests
  you will run should be added here. The `CMakeLists.txt` file in `src/` creates a
  [static library](https://en.wikipedia.org/wiki/Static_library) which is linked
  to all the other targets in the repository, and which propagates its dependency,
  so that there is no need to explicitly link against Eigen or MPI.
- `tests/` contains tests for the library code code. It uses
  [GoogleTest](https://google.github.io/googletest/) to define short, simple
  test cases.
- `executables/` contains the final executable codes, i.e. it needs a `main()`
  function.

### Adding to `src/`

Adding files to `src/` is straightforward: create your files, e.g. `lj.h` and
`lj.cpp`, then update the `add_library` command in
`src/CMakeLists.txt`:

```cmake
add_library(lib STATIC 
    hello.cpp
    lj.cpp
)
```

### Adding to `tests/`

Create your test file, e.g. `test_verlet.cpp` in `tests/`, then modify the
`TEST_SOURCES` variable in `tests/CMakeLists.txt`. Test that your test was
correctly added by running `cmake --build build` in the root directory: your test should
show up in the output of `ctest`.

### Adding to `executables/`

Create a new directory, e.g. with `mkdir executables/04`, then add `add_subdirectory(04)`
to `executables/CMakeLists.txt`, then create & edit
`executables/04/CMakeLists.txt`:

```cmake
add_executable(milestone04 main.cpp)
target_link_libraries(milestone04 PRIVATE lib)
```

You can now create & edit `executables/04/main.cpp`, which should include a
`main()` function as follows:

```c++
int main(int argc, char* argv[]) {
    return 0;
}
```

The code of your simulation goes into the `main()` function.

#### Input files

We often provide input files (`.xyz` files) for your simulations, for example in
milestone 4. You should place these in e.g. `executables/04/`, and add the
following to `executables/04/CMakeLists.txt`:

```cmake
configure_file(lj54.xyz lj54.xyz COPYONLY)
```

This will copy the file `executables/04/lj54.xyz` to
`<build>/executables/04/lj54.xyz`, but **only** when CMake is reconfigured.

*Note:* `.xyz` files are ignored by Git. That's on purpose to avoid you staging
very large files in the git tree.

## Pushing code to GitHub

If you have added files to your local repositories, you should commit and push them to
GitHub. To create a new commit (i.e. put your files in the repository's
history), simply run:

```bash
git status
# Look at the files that need to be added
git add <files> ...
git commit -m '<a meaningful commit message!>'
git push
```

This repository is setup with continuous integration (CI), so all your tests
will run automatically when you push. This is very handy to test if you have
breaking changes.

### Git in CLion

If you are using CLion, you can use Git directly from its interface. To add
files, right click the file you wish to add, then "Git > Add". Once you are
ready to commit, "Git > Commit" from the main menu bar. Add a message in the
lower left window where it reads "Commit message", then click "Commit" or
"Commit and Push...".

## Milestone 6, steps 1 through 4: distributed cavity solver

The MPI implementation is in `executables/milestone6.cpp`. It splits the
nonperiodic x dimension into contiguous strips, with remainder columns assigned
to the first ranks. Each rank owns all y rows and at least one x column.
`f(local_nx+2, NY, 9)` contains owned columns 1..local_nx and two ghost columns.
The global x index is `x_begin + local_x - 1`; end ranks have `MPI_PROC_NULL`
neighbors at the physical side walls.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone6
mpiexec -n 3 ./build/executables/milestone6 128 128 --steps 1000
mpiexec -n 3 ./build/executables/milestone6 17 13 --steps 200 --check-solver
ctest --test-dir build -R Milestone6 --output-on-failure
```

### Communication pattern and ghost cell width

**Decomposition.** 1-D strip decomposition along the nonperiodic x
dimension only; every rank owns the full y extent (`NY` rows) and a
contiguous span of x columns (`local_nx = NX/ranks`, with one extra column
given to each of the first `NX % ranks` ranks). Rank `r` has neighbors
`r-1` (left) and `r+1` (right); the first and last ranks get
`MPI_PROC_NULL` on their outward side, so `HaloExchange::exchange` (below)
naturally skips sending/receiving there and the physical side walls are
handled by ordinary bounce-back instead.

**Ghost cell width: 1 column on each side.** `f` is allocated as
`(local_nx + 2, NY, 9)`: local x-index 0 is the left ghost column, local
x-index `local_nx + 1` is the right ghost column, and 1..local_nx are the
owned columns. One ghost column is exactly enough because the D2Q9
stencil only ever reads one lattice site away in x (`direction_x(i)` in
`{-1, 0, 1}`), and only one exchange happens per timestep (see below), so
no rank ever needs to read two columns into a neighbor's territory.

**What gets exchanged, and when.** Only the 3 of 9 populations whose
`direction_x` actually crosses each boundary are sent, not the full ghost
column: rightward, channels 1, 5, 8 (`direction_x = +1`); leftward,
channels 3, 6, 7 (`direction_x = -1`). Diagonal channels 5/6/7/8 that
would leave through the top or bottom physical wall are also excluded
from the corresponding row of the buffer (`decode_slot`), since those
never come from a neighbor -- they get resolved locally by bounce-back
instead. That leaves `3*NY - 2` doubles per direction per exchange (the
`-2` is the two omitted diagonal rows). Exchange happens exactly once per
timestep, immediately after `collide_local` and before `pull_stream_local`:
collision only reads a cell's own 9 populations, so nothing needs to be
current before it; streaming reads across the x boundary, so the ghost
columns must be filled before it runs. No second exchange is needed after
streaming, because collision (the next step's first operation) never
reads ghost cells.

**How it's exchanged.** Two `MPI_Sendrecv` calls per timestep (one per
direction) -- deadlock-safe by construction, since each call pairs its own
send with its own receive rather than relying on ordering across ranks.
Data is packed by a Kokkos `parallel_for` into device buffers, copied to
host mirror buffers (`Kokkos::deep_copy`), sent via ordinary (non
GPU-aware) MPI, then copied back to device and unpacked by a second
`parallel_for`. This host round-trip is deliberate: it makes the exchange
correct on any Kokkos backend (Serial, and CUDA on the A100/H100 runs
below) without depending on the cluster's MPI having CUDA-aware support --
at the cost of an extra device-to-host and host-to-device copy per
exchange that a CUDA-aware MPI build could skip.

Grid dimensions default to 128 x 128 and the step count defaults to 1000.
`--steps 0` reports the initialized state. This is a fixed-step simulation,
not a steady-state convergence claim. Omega, lid velocity, density, and the
stationary top-corner convention match milestone 5; changing grid size with
these fixed parameters changes the Reynolds number.

Each timestep performs:

1. `collide_local`: Kokkos parallel_for updates only owned populations using
   local density/velocity and the D2Q9 BGK equilibrium.
2. `HaloExchange::exchange`: send post-collision channels 1,5,8 rightward and
   3,6,7 leftward. Diagonals hitting top/bottom walls are omitted, so each
   neighbor receives `3*NY-2` doubles, indexed by source row. Two MPI_Sendrecv
   calls and reusable host staging buffers support ordinary MPI. Kokkos fences
   ensure the arrays are ready before communication.
3. `pull_stream_local`: Kokkos parallel_for reads from owned cells or ghosts
   into a separate destination array. Global coordinates distinguish physical
   walls from MPI interfaces. Physical walls use halfway bounce-back, with the
   moving-lid correction on interior top-wall nodes.
4. Swap array handles and recompute density and velocity on owned cells.

No extra halo exchange is needed before collision or after streaming. Ghost
cells are never collided or counted in diagnostics. `global_diagnostics` first
uses Kokkos parallel_reduce for local mass and kinetic energy, then MPI_Allreduce
with MPI_SUM to give every rank the global values. Rank 0 prints them initially,
every 100 steps, and at the final step, together with relative mass drift.
Kinetic energy is sum(0.5*rho*(ux*ux+uy*uy)) in lattice units. Nonfinite values
or nonpositive densities are detected collectively at diagnostic checkpoints.

`--check-halos` tests two rounds of identifiable populations and, when used
alone, keeps the original zero-timestep communication test behavior.
`--check-solver` independently runs the existing serial kernels on each rank
and compares its owned populations plus global mass/energy with the reference.
Only this optional test allocates full-domain arrays on each rank; it is limited
to 65,536 cells. Population tolerance is 1e-12; diagnostic/mass-conservation
checks use 1e-11 scaled tolerances. CTest covers initial state, one step, 200
steps, 1/2/3 ranks, uneven strips, and one-column strips. The 17x13, 3-rank,
200-step run produced maximum population error 5.0e-16 and relative mass drift
about -1.8e-14 against the serial reference.

Verified with the Kokkos Serial backend under WSL. Threaded backends have
not been exercised here. GPU backends (CUDA, Serial host space) were built
and benchmarked on bwUniCluster; see the performance analysis below.

### Performance analysis: strong scaling

Measured on bwUniCluster (`cluster/README.md` has the full procedure and
raw data). Strong scaling holds the problem fixed and increases process
count; plotted as speedup `T(1)/T(p)` against an ideal `S(p) = p` line via
`plot_scaling.py`.

**CPU** (Kokkos Serial, one MPI rank per core, `dev_cpu` node): 256x256
grid, 20,000 steps, 1-32 ranks, 3 repeats each. Near-ideal: 91.9%
efficiency at 32 ranks (median 62.48s at 1 rank down to 2.12s at 32).
64 ranks was tried and dropped after it hung (see `cluster/README.md`);
1-32 already gives a complete curve.
[Plot](milestone6_results/scaling/6822999/strong_scaling.png) |
[raw data](milestone6_results/scaling/6822999/raw.csv)

**GPU** (Kokkos CUDA, one MPI rank per GPU, `gpu_a100_il`): the same
256x256 problem finished in 1.65s on a single A100 -- too little work per
GPU for a meaningful multi-GPU split, since the host-staged halo exchange
(see "Communication pattern" above) costs roughly the same per step
regardless of grid size while compute per rank shrinks as ranks increase;
2-4 ranks were consistently *slower* than 1. Scaling up to a 4096x4096
grid, 2,000 steps fixed this: 95.7% efficiency at 2 GPUs, 87.0% at 4
GPUs (median 11.19s at 1 GPU down to 3.22s at 4).
[Plot](milestone6_results/scaling/a100-6822867/strong_scaling.png) |
[raw data](milestone6_results/scaling/a100-6822867/raw.csv)

Raw times aren't comparable across the two problem sizes, but
`--benchmark`'s MLUPS figure is exactly the throughput metric that is:
a single CPU core reaches 20.98 MLUPS (1 rank, 256x256) versus a single
A100 at 2998 MLUPS (1 rank, 4096x4096) -- about 143x higher throughput
on one GPU than one CPU core for this solver.

MPI/serial validation has also been run for 1, 2, 3, and 4 processes on 128x128
(1000 steps) and 17x13 (200 steps), plus a one-column-per-rank case. All nine
runs passed, with maximum population error 8.88e-16. Reproduce with
`python3 validate_milestone6.py`; see [the validation report](milestone6_results/validation.md)
and [per-run errors](milestone6_results/validation.csv).
