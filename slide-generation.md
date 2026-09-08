# Prompt for a slide-generation agent

You are building presentation slides for a 10-minute academic talk on a
university course project (Freiburg "Accelerators" course, taught by Prof.
Pastewka). The speaker will read the speech below almost verbatim; your job
is to build slides that support each spoken section without duplicating it
word-for-word. Follow standard technical-talk slide design: minimal text per
slide (headline + 3-6 bullet fragments, not full sentences), one idea per
slide, generous use of the plots and data already produced by the project
(paths given below), and a consistent, clean visual style (e.g. simple sans
font, one accent color, dark-on-light or light-on-dark consistently, no
gratuitous clip art).

Build 10-12 slides total, mapped to the speech sections 1:1 or 1:2 as noted.
Output as a slide deck (PowerPoint .pptx, or Google Slides/Reveal.js HTML —
pick whichever tool you have available; if using an HTML/Reveal deck, treat
this as an Artifact and follow that design workflow).

## Source material to pull from

Project repo root: the Lattice Boltzmann (LBM) D2Q9 solver "YALB".

- `README.md` — describes milestones 4, 5, 6 in detail (viscosity
  validation, lid-driven cavity, MPI distributed solver).
- `include/lbm.hpp`, `src/lbm.cpp` — the Kokkos kernel implementations
  (collision, streaming, streaming_with_walls, equilibrium).
- `executables/milestone4.cpp`, `milestone5.cpp`, `milestone6.cpp` — the
  three milestone drivers.
- `milestone4_results/viscosity_vs_omega.svg`,
  `milestone4_results/log_amplitude_fit.svg`,
  `milestone4_results/velocity_profiles.svg` — viscosity validation plots.
- `milestone5_results/cavity_speed.svg`,
  `milestone5_results/cavity_velocity_vectors.svg`,
  `milestone5_results/centerline_ux.svg` — lid-driven cavity result plots
  (steady-state flow field and Ghia et al. comparison).
- `milestone5_results/summary.txt` and `benchmark_summary.txt` — the
  numbers: Re=400, 83,200 steps to converge, RMSE 0.36%, max error 0.83%,
  19.44 MLUPS baseline.
- `milestone6_results/validation.md` and `validation.csv` — MPI-vs-serial
  correctness table (max population error 8.9e-16 across 1-4 ranks).
- A recorded video/GIF of the sliding-lid simulation running (the speaker
  will supply this file separately — leave an explicit placeholder slide
  titled "Sliding lid — live/recorded run" with an embedded video
  placeholder, do not fabricate a video).
- If available by build time: `milestone6_results/scaling/<JOBID>/summary.csv`
  and `strong_scaling.png`/`.svg` from the bwUniCluster strong-scaling run
  (256x256 grid, 20000 steps, 1-64 MPI ranks). If this file does not exist
  yet, build the performance slide around the milestone5 baseline number
  only (19.44 MLUPS, i.e. 1.944e-5 GLUPS) and add a small note "cluster
  strong-scaling run pending" instead of inventing scaling numbers.

Do not invent numbers, plots, or results that are not in the files above or
in the speech text provided below. If a figure referenced in the speech
doesn't exist as a file, build a simple placeholder (labeled clearly as a
placeholder) rather than fabricating data.

## Slide-by-slide structure

1. **Title slide** — project name "YALB: A Hybrid Kokkos + MPI Lattice
   Boltzmann Solver", course name, speaker name, date.
2. **Introduction** — one-line problem statement (simulate fluid flow with
   LBM), one-line roadmap of the talk (what/how, methods, results, video,
   performance, discussion).
3. **What I built / programming model** — diagram: Kokkos (parallel_for
   over Views) for on-node kernels + hand-written MPI (Sendrecv halo
   exchange) for cross-rank, composed into one hybrid solver. Emphasize
   "single kernel source, multiple backends."
4. **Methods overview** — D2Q9 BGK scheme in one visual (9 velocity
   directions diagram) + the milestone progression as a small timeline:
   M3 boundaries -> M4 viscosity validation -> M5 cavity steady state ->
   M6 MPI distributed.
5. **Milestone 4 validation** — embed `viscosity_vs_omega.svg` and/or
   `log_amplitude_fit.svg`; one-line takeaway: measured viscosity matches
   `nu = (1/omega - 1/2)/3` theory.
6. **Programming model detail — vs. the lecture** — two-column comparison:
   "lecture path" (separate serial/OpenMP/CUDA code paths) vs. "this
   project" (one Kokkos kernel source, backend chosen at compile time) +
   a small note on the hand-rolled halo exchange only moving 3*NY-2
   doubles per neighbor (not full ghost columns).
7. **Advantages vs. pitfalls** — two-column list, pulled directly from
   speech section 5 (advantages: single codebase, thin auditable MPI
   layer, --check-solver cross-validation; pitfalls: Kokkos build
   complexity/lambda errors, global/local coordinate bugs at boundaries,
   D2Q9-specific halo optimization not easily reusable).
8. **Results — lid-driven cavity** — embed `cavity_speed.svg` or
   `cavity_velocity_vectors.svg`, plus the key numbers: Re=400, 128x128,
   converged at 83,200 steps, RMSE 0.36% vs. Ghia et al., max error 0.83%.
9. **Show it works — sliding lid video** — placeholder video embed slide
   (title only + "insert recorded run here" placeholder box); minimal
   other text.
10. **MPI correctness** — small table from `validation.md`: max population
    difference 8.9e-16 across 1-4 ranks, mass/energy conserved.
11. **Performance (GLUPS)** — headline number: 19.44 MLUPS baseline
    (1.944e-5 GLUPS), 128x128 grid, single rank, Kokkos Serial. If cluster
    scaling data exists, add the strong-scaling plot and speedup/efficiency
    numbers at max rank count; otherwise note "strong-scaling run on
    bwUniCluster: pending."
12. **Discussion — lessons and next time** — two bullets pulled from
    speech section 8: (1) parallelization bookkeeping (ghost cells,
    global/local coordinates) was harder than the physics; build the
    serial-vs-parallel correctness harness before the parallel kernel, not
    after; (2) untested assumption going forward: whether vertical-strip
    decomposition still scales at 64 ranks / 4 columns per rank
    (communication-bound regime).

## Style notes

- Keep slide text to fragments; the speaker is talking from the full
  speech separately, slides are visual support, not a teleprompter.
- Use the project's own SVG plots directly rather than redrawing them.
- Keep a consistent color for "theory/reference" vs. "measured/simulation"
  data if you recreate any chart (do not restyle the existing SVGs).
- Total should comfortably support ~10 minutes of spoken delivery: no
  slide should require more than ~
    bwUniCluster: pending."
12. **Discussion — lessons and next time** — two bullets pulled from
    speech section 8: (1) parallelization bookkeeping (ghost cells,
    global/local coordinates) was harder than the physics; build the
    serial-vs-parallel correctness harness before the parallel kernel, not
    after; (2) untested assumption going forward: whether vertical-strip
    decomposition still scales at 64 ranks / 4 columns per rank
    (communication-bound regime).

## Style notes

- Keep slide text to fragments; the speaker is talking from the full
  speech separately, slides are visual support, not a teleprompter.
- Use the project's own SVG plots directly rather than redrawing them.
- Keep a consistent color for "theory/reference" vs. "measured/simulation"
  data if you recreate any chart (do not restyle the existing SVGs).
- Total should comfortably support ~10 minutes of spoken delivery: no
  slide should require more than ~45-60 seconds of narration.