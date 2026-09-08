#!/bin/bash -l
# Usage: bash cluster/build_gpu.sh <a100|h100>
#
# Kokkos_ENABLE_CUDA requires nvcc_wrapper (shipped inside the Kokkos source
# that CMakeLists.txt fetches via FetchContent) as CMAKE_CXX_COMPILER. That
# source does not exist on disk until a configure step has already run, so
# this is a two-pass bootstrap: pass 1 configures with a plain MPI compiler
# just to populate build-cluster-gpu-<arch>/_deps/kokkos-src, then pass 2
# reconfigures with nvcc_wrapper and CUDA enabled. This mirrors the two-pass
# pattern already used for the CPU build in cluster/build.sh.
set -euo pipefail
arch="${1:?Usage: build_gpu.sh <a100|h100>}"
case "$arch" in
    a100) kokkos_arch_flag=Kokkos_ARCH_AMPERE80 ;;
    h100) kokkos_arch_flag=Kokkos_ARCH_HOPPER90 ;;
    *) echo "Unknown architecture '$arch' (expected a100 or h100)" >&2; exit 1 ;;
esac

# NOTE: verify these module names/versions with `module avail cuda` and
# `module avail mpi` on the cluster before running -- they were assembled
# from documentation, not confirmed against a live bwUniCluster GPU session.
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0.8-gnu-14.2 devel/cuda/12.8

build_dir="build-cluster-gpu-${arch}"

cmake -S . -B "$build_dir" -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=OFF

nvcc_wrapper="$(pwd)/$build_dir/_deps/kokkos-src/bin/nvcc_wrapper"
test -x "$nvcc_wrapper" || { echo "nvcc_wrapper not found at $nvcc_wrapper after bootstrap configure" >&2; exit 1; }

# Drop the cache (keep _deps, which already holds the fetched sources) so
# CMake re-detects the compiler instead of refusing the change in-place.
rm -f "$build_dir/CMakeCache.txt"
# Kokkos_ENABLE_SERIAL stays ON: Kokkos requires a host execution space even
# with CUDA enabled, for host-side kernels and the HaloExchange host mirrors.
cmake -S . -B "$build_dir" -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER="$nvcc_wrapper" \
    -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON \
    -D"${kokkos_arch_flag}"=ON

cmake --build "$build_dir" --target milestone6 -j 4
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' "$build_dir/CMakeCache.txt"
echo "Built $build_dir for $arch ($kokkos_arch_flag)"
