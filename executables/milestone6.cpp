#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "lbm.hpp"
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int Q = 9;
constexpr double UNSET = -1.0;
using Populations = Kokkos::View<double***>;
using Buffer = Kokkos::View<double*>;

// Nonperiodic vertical strips. Every rank owns all y rows and at least one x
// column. The first remainder ranks receive one additional column.
struct Domain {
    int rank, ranks;
    int global_nx, ny, local_nx, x_begin;
    int left, right;

    Domain(int nx, int height) : global_nx(nx), ny(height) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        if (nx < ranks || nx > std::numeric_limits<int>::max() - 2 || height < 1 || height > (std::numeric_limits<int>::max() - 2) / 3) {
            throw std::invalid_argument("Require NX >= MPI process count and 1 <= NY <= (INT_MAX-2)/3");
        }
        const int base = nx / ranks;
        const int remainder = nx % ranks;
        local_nx = base + (rank < remainder ? 1 : 0);
        x_begin = rank * base + std::min(rank, remainder);
        left = rank == 0 ? MPI_PROC_NULL : rank - 1;
        right = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
    }
};

// Local x: 0 = left ghost, 1..local_nx = owned, local_nx+1 = right ghost.
// Global x for an owned cell is x_begin + local_x - 1.
void initialize_equilibrium(const Populations& f, const Domain& domain) {
    Kokkos::deep_copy(f, UNSET);
    const int nx = domain.local_nx;
    const int ny = domain.ny;
    Kokkos::parallel_for("initialize_local_cavity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {nx + 1, ny}),
        KOKKOS_LAMBDA(int x, int y) {
            for (int i = 0; i < Q; ++i) {
                // D2Q9 equilibrium for rho = 1 and u = 0.
                f(x, y, i) = i == 0 ? 4.0 / 9.0 : (i < 5 ? 1.0 / 9.0 : 1.0 / 36.0);
            }
        });
    Kokkos::fence();
}

// Buffer layout, indexed by the SOURCE row (diagonal y shifts occur later
// during pull streaming): horizontal: NY entries; up/down: NY-1 each.
// Diagonals leaving the physical top/bottom wall bounce back locally and
// therefore must not be read from a neighbor's ghost column.
KOKKOS_INLINE_FUNCTION
void decode_slot(int slot, int ny, bool rightward, int& y, int& channel) {
    if (slot < ny) {
        y = slot;
        channel = rightward ? 1 : 3;
    } else if (slot < 2 * ny - 1) {
        y = slot - ny;
        channel = rightward ? 5 : 6;  // cy = +1; omit source y = NY-1
    } else {
        y = slot - (2 * ny - 1) + 1;
        channel = rightward ? 8 : 7;  // cy = -1; omit source y = 0
    }
}

struct HaloExchange {
    Buffer send_left, send_right, receive_left, receive_right;
    Buffer::HostMirror host_send_left, host_send_right, host_receive_left, host_receive_right;
    int count;

    explicit HaloExchange(int ny)
        : send_left("send_left", 3 * ny - 2), send_right("send_right", 3 * ny - 2),
          receive_left("receive_left", 3 * ny - 2), receive_right("receive_right", 3 * ny - 2),
          host_send_left(Kokkos::create_mirror_view(send_left)),
          host_send_right(Kokkos::create_mirror_view(send_right)),
          host_receive_left(Kokkos::create_mirror_view(receive_left)),
          host_receive_right(Kokkos::create_mirror_view(receive_right)), count(3 * ny - 2) {}

    // Timestep order:
    //   collide owned cells -> exchange post-collision f -> pull-stream owned cells.
    // Collision only needs the nine populations at its own cell, so no second
    // exchange before collision or after streaming is necessary. Never collide
    // ghost cells. Physical-wall bounce-back takes precedence over halo reads.
    void exchange(const Populations& f, const Domain& domain) {
        const int nx = domain.local_nx;
        const int ny = domain.ny;
        const bool has_left = domain.left != MPI_PROC_NULL;
        const bool has_right = domain.right != MPI_PROC_NULL;
        const auto sl = send_left;
        const auto sr = send_right;
        const auto rl = receive_left;
        const auto rr = receive_right;
        Kokkos::parallel_for("pack_x_halos", count, KOKKOS_LAMBDA(int slot) {
            int y, channel;
            decode_slot(slot, ny, false, y, channel);
            sl(slot) = f(1, y, channel);
            decode_slot(slot, ny, true, y, channel);
            sr(slot) = f(nx, y, channel);
        });
        // Host staging works with ordinary MPI, including a device-backed View.
        // Fence ensures packing has completed before MPI touches host buffers.
        Kokkos::fence();
        Kokkos::deep_copy(host_send_left, send_left);
        Kokkos::deep_copy(host_send_right, send_right);
        MPI_Sendrecv(host_send_right.data(), count, MPI_DOUBLE, domain.right, 100,
                     host_receive_left.data(), count, MPI_DOUBLE, domain.left, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(host_send_left.data(), count, MPI_DOUBLE, domain.left, 101,
                     host_receive_right.data(), count, MPI_DOUBLE, domain.right, 101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_PROC_NULL leaves a receive buffer unchanged; do not unpack it.
        if (has_left) Kokkos::deep_copy(receive_left, host_receive_left);
        if (has_right) Kokkos::deep_copy(receive_right, host_receive_right);
        Kokkos::parallel_for("unpack_x_halos", count, KOKKOS_LAMBDA(int slot) {
            int y, channel;
            if (has_left) {
                decode_slot(slot, ny, true, y, channel);
                f(0, y, channel) = rl(slot);
            }
            if (has_right) {
                decode_slot(slot, ny, false, y, channel);
                f(nx + 1, y, channel) = rr(slot);
            }
        });
        //Wait until all previously started Kokkos work has finished before continuing
        Kokkos::fence();
    }
};

// Distinct values expose wrong neighbors, channels, row shifts, and stale data.
// This is a communication test, not yet a fluid simulation or scaling study.
double test_value(int round, int x, int y, int channel, const Domain& domain) {
    return 1.0 + (((static_cast<double>(round) * domain.global_nx + x) * domain.ny + y) * Q + channel);
}

void check_halos(const Populations& f, const Domain& domain, HaloExchange& halo) {
    //CPU accessible view
    auto host = Kokkos::create_mirror_view(f);
    for (int round = 0; round < 2; ++round) {
        for (int x = 0; x < domain.local_nx + 2; ++x) {
            for (int y = 0; y < domain.ny; ++y) {
                for (int i = 0; i < Q; ++i) {
                    host(x, y, i) = (x == 0 || x == domain.local_nx + 1)
                        ? UNSET : test_value(round, domain.x_begin + x - 1, y, i, domain);
                }
            }
        }
        Kokkos::deep_copy(f, host);
        halo.exchange(f, domain);
        Kokkos::deep_copy(host, f);
        for (int x = 0; x < domain.local_nx + 2; ++x) {
            for (int y = 0; y < domain.ny; ++y) {
                for (int i = 0; i < Q; ++i) {
                    const bool interior = x > 0 && x <= domain.local_nx;
                    // Independent expected crossing conditions; no decode_slot reuse.
                    const bool from_left = x == 0 && domain.left != MPI_PROC_NULL &&
                        (i == 1 || (i == 5 && y + 1 < domain.ny) || (i == 8 && y > 0));
                    const bool from_right = x == domain.local_nx + 1 && domain.right != MPI_PROC_NULL &&
                        (i == 3 || (i == 6 && y + 1 < domain.ny) || (i == 7 && y > 0));
                    const double expected = interior || from_left || from_right
                        ? test_value(round, domain.x_begin + x - 1, y, i, domain) : UNSET;
                    if (host(x, y, i) != expected) {
                        throw std::runtime_error("halo check failed at local (" + std::to_string(x) +
                            "," + std::to_string(y) + "), channel " + std::to_string(i));
                    }
                }
            }
        }
    }
}

// Same D2Q9 convention and physical parameters as milestone 5.
constexpr double OMEGA = 1.0 / 0.596;
constexpr double LID_VELOCITY = 0.1;
constexpr double WALL_DENSITY = 1.0;
using Field = Kokkos::View<double**>;
using OwnedCells = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

KOKKOS_INLINE_FUNCTION int direction_x(int i) {
    const int values[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    return values[i];
}
KOKKOS_INLINE_FUNCTION int direction_y(int i) {
    const int values[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    return values[i];
}
KOKKOS_INLINE_FUNCTION int opposite_direction(int i) {
    const int values[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return values[i];
}
KOKKOS_INLINE_FUNCTION double weight(int i) {
    return i == 0 ? 4.0 / 9.0 : (i < 5 ? 1.0 / 9.0 : 1.0 / 36.0);
}

// Fields include padding for convenient shared indexing, but only owned cells
// are computed or reduced. Ghost populations must never enter global sums.
void update_local_fields(const Populations& f, const Domain& d,
                         const Field& rho, const Field& ux, const Field& uy) {
    Kokkos::parallel_for("local_macroscopic_fields",
        OwnedCells({1, 0}, {d.local_nx + 1, d.ny}), KOKKOS_LAMBDA(int x, int y) {
            double density = 0.0, momentum_x = 0.0, momentum_y = 0.0;
            for (int i = 0; i < Q; ++i) {
                density += f(x, y, i);
                momentum_x += direction_x(i) * f(x, y, i);
                momentum_y += direction_y(i) * f(x, y, i);
            }
            rho(x, y) = density;
            ux(x, y) = density > 0.0 ? momentum_x / density : 0.0;
            uy(x, y) = density > 0.0 ? momentum_y / density : 0.0;
        });
    Kokkos::fence();
}

void collide_local(const Populations& f, const Domain& d,
                   const Field& rho, const Field& ux, const Field& uy) {
    Kokkos::parallel_for("local_collision",
        OwnedCells({1, 0}, {d.local_nx + 1, d.ny}), KOKKOS_LAMBDA(int x, int y) {
            const double u2 = ux(x, y) * ux(x, y) + uy(x, y) * uy(x, y);
            for (int i = 0; i < Q; ++i) {
                const double cu = direction_x(i) * ux(x, y) + direction_y(i) * uy(x, y);
                const double equilibrium = weight(i) * rho(x, y) *
                    (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                f(x, y, i) += OMEGA * (equilibrium - f(x, y, i));
            }
        });
    Kokkos::fence();
}

void pull_stream_local(const Populations& post_collision, const Populations& next,
                       const Domain& d) {
    const int global_nx = d.global_nx, ny = d.ny, x_begin = d.x_begin;
    Kokkos::parallel_for("local_pull_streaming",
        OwnedCells({1, 0}, {d.local_nx + 1, ny}), KOKKOS_LAMBDA(int x, int y) {
            const int global_x = x_begin + x - 1;
            for (int i = 0; i < Q; ++i) {
                const int source_x = x - direction_x(i);
                const int source_y = y - direction_y(i);
                const int global_source_x = global_x - direction_x(i);
                if (global_source_x >= 0 && global_source_x < global_nx &&
                    source_y >= 0 && source_y < ny) {
                    // source_x can be a ghost column, already filled by MPI.
                    next(x, y, i) = post_collision(source_x, source_y, i);
                } else {
                    // A physical wall, not an MPI interface: halfway bounce-back.
                    double reflected = post_collision(x, y, opposite_direction(i));
                    if (source_y >= ny && global_x > 0 && global_x < global_nx - 1) {
                        // Incoming direction has the opposite sign to the outgoing
                        // direction used by milestone 5's push bounce-back formula.
                        reflected += 6.0 * weight(i) * WALL_DENSITY *
                                     direction_x(i) * LID_VELOCITY;
                    }
                    next(x, y, i) = reflected;
                }
            }
        });
    Kokkos::fence();
}

struct Diagnostics {
    double mass, kinetic_energy;
};

Diagnostics global_diagnostics(const Populations& f, const Domain& d,
                               const Field& rho, const Field& ux, const Field& uy) {
    double local_mass = 0.0, local_energy = 0.0;
    int local_invalid = 0;
    Kokkos::parallel_reduce("local_diagnostics",
        OwnedCells({1, 0}, {d.local_nx + 1, d.ny}),
        KOKKOS_LAMBDA(int x, int y, double& mass, double& energy, int& invalid) {
            const double density = rho(x, y);
            const double vx = ux(x, y), vy = uy(x, y);
            bool valid = Kokkos::isfinite(density) && density > 0.0 &&
                         Kokkos::isfinite(vx) && Kokkos::isfinite(vy);
            for (int i = 0; i < Q; ++i) valid = valid && Kokkos::isfinite(f(x, y, i));
            if (!valid) { ++invalid; return; }
            mass += density;
            energy += 0.5 * density * (vx * vx + vy * vy);
        }, Kokkos::Sum<double>(local_mass), Kokkos::Sum<double>(local_energy),
           Kokkos::Sum<int>(local_invalid));
    Kokkos::fence();
    int any_invalid = 0;
    MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_invalid != 0) throw std::runtime_error("Nonfinite populations/fields or nonpositive density");
    const double local[2] = {local_mass, local_energy};
    double global[2] = {};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return {global[0], global[1]};
}

void perform_timestep(Populations& f, Populations& next, HaloExchange& halo,
                      const Domain& d, const Field& rho, const Field& ux, const Field& uy) {
    collide_local(f, d, rho, ux, uy);
    halo.exchange(f, d);
    pull_stream_local(f, next, d);
    std::swap(f, next);  // Swap array handles; no full-array copy.
    update_local_fields(f, d, rho, ux, uy);
}

// Optional small-grid regression test. Each rank independently runs the existing
// full-domain serial kernels, then checks only its owned slice. Full-domain
// allocations occur ONLY in this test mode, never in the distributed solver.
void check_against_serial(const Populations& distributed, const Domain& d,
                          int steps, const Diagnostics& actual) {
    const int nx = d.global_nx, ny = d.ny;
    Populations f("serial_f", nx, ny, Q), next("serial_next", nx, ny, Q), eq("serial_eq", nx, ny, Q);
    Field rho("serial_rho", nx, ny), ux("serial_ux", nx, ny), uy("serial_uy", nx, ny);
    Kokkos::View<int*> cx("serial_cx", Q), cy("serial_cy", Q), opposite("serial_opposite", Q);
    Buffer weights("serial_weights", Q);
    auto hcx = Kokkos::create_mirror_view(cx), hcy = Kokkos::create_mirror_view(cy);
    auto hop = Kokkos::create_mirror_view(opposite);
    auto hw = Kokkos::create_mirror_view(weights);
    for (int i = 0; i < Q; ++i) {
        hcx(i) = direction_x(i); hcy(i) = direction_y(i);
        hop(i) = opposite_direction(i); hw(i) = weight(i);
    }
    Kokkos::deep_copy(cx, hcx); Kokkos::deep_copy(cy, hcy);
    Kokkos::deep_copy(opposite, hop); Kokkos::deep_copy(weights, hw);
    auto hf = Kokkos::create_mirror_view(f);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int i = 0; i < Q; ++i) hf(x, y, i) = weight(i);
    Kokkos::deep_copy(f, hf);
    for (int step = 0; step < steps; ++step) {
        compute_density(rho, f, nx, ny);
        compute_velocity(ux, uy, rho, f, cx, cy, nx, ny);
        compute_f_eq(eq, rho, ux, uy, cx, cy, nx, ny, weights);
        collision(f, eq, OMEGA, nx, ny);
        streaming_with_walls(f, next, cx, cy, opposite, weights, nx, ny,
                             LID_VELOCITY, WALL_DENSITY);
        Kokkos::fence();
        std::swap(f, next);
    }
    Kokkos::deep_copy(hf, f);
    auto local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), distributed);
    double error = 0.0, serial_mass = 0.0, serial_energy = 0.0;
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            double density = 0.0, mx = 0.0, my = 0.0;
            for (int i = 0; i < Q; ++i) {
                if (!std::isfinite(hf(x, y, i))) throw std::runtime_error("Serial reference is nonfinite");
                density += hf(x, y, i);
                mx += direction_x(i) * hf(x, y, i);
                my += direction_y(i) * hf(x, y, i);
                if (x >= d.x_begin && x < d.x_begin + d.local_nx)
                    error = std::max(error, std::abs(hf(x, y, i) - local(x - d.x_begin + 1, y, i)));
            }
            serial_mass += density;
            serial_energy += 0.5 * (mx * mx + my * my) / density;
        }
    }
    double maximum_error = 0.0;
    MPI_Allreduce(&error, &maximum_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (maximum_error > 1e-12 || std::abs(actual.mass - serial_mass) > 1e-11 * serial_mass ||
        std::abs(actual.kinetic_energy - serial_energy) > 1e-11 * std::max(1.0, serial_energy) ||
        std::abs(actual.mass - static_cast<double>(nx) * ny) > 1e-11 * nx * ny)
        throw std::runtime_error("Distributed populations, mass, or energy disagree with serial reference");
    if (d.rank == 0) std::cout << "Serial comparison passed: maximum population error=" << maximum_error << '\n';
}
int positive_integer(const std::string& value) {
    std::size_t end = 0;
    const int result = std::stoi(value, &end);
    if (end != value.size() || result < 1) throw std::invalid_argument("Grid dimensions must be positive integers");
    return result;
}

}  // namespace

int main(int argc, char* argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        Kokkos::initialize(argc, argv);
        {
            bool check = false, check_solver = false;
            int steps = -1;
            std::vector<std::string> dimensions;
            for (int i = 1; i < argc; ++i) {
                if (std::string(argv[i]) == "--check-halos") check = true;
                else if (std::string(argv[i]) == "--check-solver") check_solver = true;
                else if (std::string(argv[i]) == "--steps") {
                    if (++i >= argc) throw std::invalid_argument("--steps requires a nonnegative integer");
                    steps = std::string(argv[i]) == "0" ? 0 : positive_integer(argv[i]);
                } else dimensions.emplace_back(argv[i]);
            }
            if (!dimensions.empty() && dimensions.size() != 2) {
                throw std::invalid_argument("Usage: milestone6 [NX NY] [--steps N] [--check-halos] [--check-solver]");
            }
            const int nx = dimensions.empty() ? 128 : positive_integer(dimensions[0]);
            const int ny = dimensions.empty() ? 128 : positive_integer(dimensions[1]);
            if (steps < 0) steps = check && !check_solver ? 0 : 1000;
            if (check_solver && static_cast<double>(nx) * ny > 65536)
                throw std::invalid_argument("--check-solver is limited to 65536 cells");
            const Domain domain(nx, ny);
            Populations f("local_populations_with_x_ghosts", domain.local_nx + 2, ny, Q);
            HaloExchange halo(ny);
            if (check) check_halos(f, domain, halo);
            Populations next("next_populations", domain.local_nx + 2, ny, Q);
            Field rho("local_rho", domain.local_nx + 2, ny);
            Field ux("local_ux", domain.local_nx + 2, ny), uy("local_uy", domain.local_nx + 2, ny);
            initialize_equilibrium(f, domain);
            halo.exchange(f, domain);
            std::cout << "rank " << rank << '/' << domain.ranks
                      << ": global x=[" << domain.x_begin << ',' << domain.x_begin + domain.local_nx
                      << "), owned=" << domain.local_nx << 'x' << ny
                      << ", neighbors=(" << domain.left << ',' << domain.right << ')'
                      << ", values/neighbor=" << halo.count
                      << (check ? ", halo check passed (2 rounds)" : ", equilibrium halo exchange complete")
                      << std::endl;
            update_local_fields(f, domain, rho, ux, uy);
            const Diagnostics initial = global_diagnostics(f, domain, rho, ux, uy);
            auto report = [&](int step, const Diagnostics& diagnostics) {
                if (rank == 0) std::cout << std::setprecision(16)
                    << "step=" << step << " mass=" << diagnostics.mass
                    << " kinetic_energy=" << diagnostics.kinetic_energy
                    << " relative_mass_drift=" << (diagnostics.mass - initial.mass) / initial.mass << '\n';
            };
            report(0, initial);
            Diagnostics final = initial;
            for (int step = 0; step < steps; ++step) {
                perform_timestep(f, next, halo, domain, rho, ux, uy);
                if ((step + 1) % 100 == 0 || step + 1 == steps) {
                    final = global_diagnostics(f, domain, rho, ux, uy);
                    report(step + 1, final);
                }
            }
            if (check_solver) check_against_serial(f, domain, steps, final);
        }  // Destroy Kokkos Views before finalizing Kokkos.
        Kokkos::finalize();
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << std::endl;
        // A rank-local failure must not leave its neighbors waiting in MPI.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
