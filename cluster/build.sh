#!/bin/bash -l
set -euo pipefail
module purge
module load compiler/gnu/14.2 mpi/openmpi/5.0.8-gnu-14.2
# Repeating configuration restores options if a changed compiler resets cache.
for pass in 1 2; do
    cmake -S . -B build-cluster -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON \
        -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=OFF
done
cmake --build build-cluster --target milestone6 -j 2
grep -qx 'CMAKE_BUILD_TYPE:STRING=Release' build-cluster/CMakeCache.txt
