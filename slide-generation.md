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

`speech.md` is organized into exactly four top-level sections — Introduction,
Methods, Results, Discussion — per the course's standard talk structure
(Introduction: what and how; Methods: programming model, how it works,
advantages/pitfalls; Results: show the code working, then performance;
Discussion: lessons, what's next). Build the deck in that same four-section
order and group slides visually/logically under those four headings (a
section can span multiple slides, but slides should never jump between
sections out of order). Per general talk-craft guidance (avoid "data dump"
slides; give the audience a moment to "come up for air" at section
boundaries rather than blurring sections together): the section boundaries
in the speech carry a one-line spoken transition — do not put that
transition sentence on a slide verbatim, but the slide that starts a new
section should read as a clean, obvious new beginning (new headline
register, not a continuation of the previous slide's bullet list).

Build exactly 14 slides total, mapped to the speech sections as detailed
below. No separate references slide — this talk's only citation (Ghia,
Ghia & Shin, 1982) is folded inline as small caption text on the results
slide that uses it, per the "citation lives on the same slide as the
figure it supports" rule; do not add a standalone references/thank-you
slide, and do not add an agenda/table-of-contents slide (the audience
forgets it within seconds — the four spoken section transitions do that
job instead). Output as a slide deck (PowerPoint .pptx, or Google
Slides/Reveal.js HTML — pick whichever tool you have available; if using
an HTML/Reveal deck, treat this as an Artifact and follow that design
workflow).

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
- `milestone5_results/cavity_evolution.gif` — an actual recorded animation
  of the lid-driven cavity converging to steady state (this file now
  exists; embed it directly on the "show it works" slide, it is not a
  placeholder).
- `milestone5_results/summary.txt`, `mlups_report.txt`, and
  `benchmark_summary.txt` — the numbers: Re=400, 83,200 steps to converge,
  RMSE 0.36%, max error 0.83%, 18.38 MLUPS median single-rank baseline on
  this solver build (7 repeats, range 18.08-18.54).
- `milestone6_results/validation.md` and `validation.csv` — MPI-vs-serial
  correctness table (max population error 8.9e-16 across 1-4 ranks).
- `milestone6_results/mlups_report.txt` and `mlups_report.csv` — the
  headline CPU and GPU throughput numbers (single-core, 32-core, single-
  A100, 4-A100, and the CPU-vs-GPU comparison). This is the primary source
  for the performance slide(s).
- `milestone6_results/scaling/6822999/strong_scaling.{svg,png}` and
  `efficiency.{svg,png}` — single-size (256x256) CPU speedup and
  efficiency plots, 1-32 ranks.
- `milestone6_results/scaling/a100-6822867/strong_scaling.{svg,png}` and
  `efficiency.{svg,png}` — GPU (A100) speedup and efficiency plots, 1-4
  ranks.
- `milestone6_results/scaling/strong_scaling_all_sizes/strong_scaling_multisize.{svg,png}`
  and `efficiency_multisize.{svg,png}` — the five-size (64/128/256/512/1024)
  overlay showing the communication-to-computation ratio effect; this is
  the strongest single figure in the deck for demonstrating scaling
  understanding, not just a scaling result. Per-size efficiency-at-32-ranks
  numbers are in `cluster/README.md`'s "Multisize results" table and in
  each `size_NxN/summary.csv`.

This scaling sweep has completed (it was still pending in an earlier draft
of this document and of `speech.md` — both are now up to date). Do not
invent numbers, plots, or results beyond what is in the files above or in
the current `speech.md`. If a figure referenced in the speech doesn't exist
as a file, build a simple placeholder (labeled clearly as a placeholder)
rather than fabricating data — but check first, since most of what used to
require a placeholder now has a real file.

## Slide-by-slide structure

1. **Title slide** — project name "YALB: A Hybrid Kokkos + MPI Lattice
   Boltzmann Solver", course name, speaker name, date.

### SECTION 1 — Introduction (slides 2-3, speech §1)

2. **Goal** — one-line problem statement (simulate fluid flow with LBM) as
   short fragments: implement / validate / accelerate / scale. No
   agenda/table-of-contents list — this is the goal statement, not an
   outline of the talk.
3. **Programming model** — diagram: Kokkos (parallel_for over Views) for
   on-node kernels + hand-written MPI (Sendrecv halo exchange) for
   cross-rank, composed into one hybrid solver. Emphasize "single kernel
   source, multiple backends: Serial, OpenMP, CUDA."

### SECTION 2 — Methods (slides 4-7, speech §2)

4. **Methods overview** — D2Q9 BGK scheme in one visual (9 velocity
   directions diagram) + the milestone progression as a small timeline:
   M3 boundaries -> M4 viscosity validation -> M5 cavity steady state ->
   M6 MPI distributed.
5. **Milestone 4 validation** — embed `viscosity_vs_omega.svg` and/or
   `log_amplitude_fit.svg`; one-line takeaway: measured viscosity matches
   `nu = (1/omega - 1/2)/3` theory.
6. **How this differs from the lecture** — two-column comparison:
   "lecture path" (separate serial/OpenMP/CUDA code paths) vs. "this
   project" (one Kokkos kernel source, backend chosen at compile time) +
   a small note on the hand-rolled halo exchange only moving 3*NY-2
   doubles per neighbor (not full ghost columns). Third design-choice
   callout: the halo exchange stages through a host mirror before every
   MPI call even on GPU — a deliberate personal tradeoff (one debugged
   code path + portability, since the cluster's CUDA-aware MPI support
   couldn't be confirmed ahead of time) over CUDA-aware MPI's extra
   speed. Flag explicitly as a choice, not an oversight — this is the
   slide to make that judgment call legible to the audience.
7. **Advantages vs. pitfalls** — two-column list, pulled directly from
   speech section 2's advantages/pitfalls paragraphs (advantages: single
   codebase, thin auditable MPI layer, --check-solver cross-validation;
   pitfalls: Kokkos build complexity/lambda errors, global/local
   coordinate bugs at boundaries, D2Q9-specific halo optimization not
   easily reusable). This is the natural close of the Methods section —
   the next slide should read as an obvious new beginning (Results).

### SECTION 3 — Results (slides 8-13, speech §3): code works, then performance

8. **Results — lid-driven cavity** — embed `cavity_speed.svg` or
   `cavity_velocity_vectors.svg`, plus the key numbers: Re=400, 128x128,
   converged at 83,200 steps, RMSE 0.36% vs. Ghia et al., max error 0.83%.
   Small caption citation on this slide: "Ghia, Ghia & Shin, J. Comput.
   Phys. 48, 387 (1982)" — this is the deck's only citation and it lives
   here, not on a separate references slide.
9. **Show it works — sliding lid animation** — embed the real
   `milestone5_results/cavity_evolution.gif` (this file exists; it is not
   a placeholder), title only, minimal other text — let the animation
   carry the slide.
10. **MPI correctness — my `--check-solver` mode** — do not lead with a bare
    number table; lead with what was actually built: a `--check-solver`
    flag on the milestone 6 executable that reruns the existing full-domain
    serial kernels on every rank and diffs the distributed result against
    it, cell-by-cell, every population. That framing is the point of this
    slide (a self-checking harness the speaker built, not an abstract
    correctness claim) — the table from `validation.md` is the evidence
    underneath it: max population difference 8.9e-16 across 1-4 ranks
    (128x128), 5.0e-16 (17x13 and the 3x5 one-column-per-rank case), mass
    and kinetic energy conserved to the same tolerance. If this slide reads
    as just numbers with no sense of whose code produced them, it has
    failed — say "I built a mode that..." not "the results show...".
11. **Performance — GLUPS and CPU strong scaling** — headline numbers from
    `milestone6_results/mlups_report.txt`: 20.98 MLUPS single-core baseline
    (256x256), 616.97 MLUPS at 32 cores, 29.4x speedup / 92% efficiency.
    Embed `milestone6_results/scaling/6822999/strong_scaling.svg` (speedup
    vs. ideal) and/or `efficiency.svg` (1-32 ranks). One display item
    preferred; use two only if speedup and efficiency both fit cleanly.
12. **Performance — does problem size matter?** — the five-size overlay,
    `strong_scaling_all_sizes/efficiency_multisize.svg` (or
    `strong_scaling_multisize.svg`). One-line takeaway: efficiency at 32
    ranks rises from 61% (64x64) to 96% (512x512) as the halo-to-interior
    ratio shrinks — small grids are communication-bound, large ones
    aren't. This is the slide that shows scaling *understanding*, not just
    a scaling *result* — worth its own slide rather than folding into #11.
13. **Performance — GPU** — single A100: 2998 MLUPS on 4096x4096 (~143x a
    single CPU core); 4 A100s: 10,432 MLUPS, 87% efficiency (3.48x
    speedup). Embed `scaling/a100-6822867/strong_scaling.svg` or
    `efficiency.svg`. Emphasize: same Kokkos source, no code changes,
    recompiled for CUDA. One-line callback to slide 6's host-staged-halo
    design choice: it's why the GPU problem size (4096x4096) is much
    larger than the CPU one (256x256) — at the small size the extra
    device-host-device copy per exchange dominates and speedup falls
    below 1. Don't re-explain the tradeoff here, just point back to it.

### SECTION 4 — Discussion (slide 14, speech §4)

14. **Discussion — lessons and next time** — bullets pulled from speech
    section 4: (1) parallelization bookkeeping (ghost cells, global/local
    coordinates) was harder than the physics; build the serial-vs-parallel
    correctness harness before the parallel kernel, not after; (2) the
    multi-size sweep confirmed the communication-bound-at-small-grids
    hypothesis, but only after actually running it; (3) 64-rank runs hung
    outright (~18.6% CPU efficiency over a 7.5-min timeout) rather than
    degrading gracefully — likely an MPI launch/binding issue at that
    scale, not isolated further this round. This is the talk's closing
    slide — no references/thank-you/contact slide after it.

14 slides total, matching the four-section structure exactly (Introduction
2-3, Methods 4-7, Results 8-13, Discussion 14, plus the title slide). If
trimming is ever needed, #11 and #12 are the pair most easily merged
(headline CPU numbers + multisize efficiency plot on one slide).

## Style notes

- Keep slide text to fragments; the speaker is talking from the full
  speech separately, slides are visual support, not a teleprompter.
- Use the project's own SVG/PNG plots directly rather than redrawing them.
- Keep a consistent color for "theory/reference" vs. "measured/simulation"
  data if you recreate any chart (do not restyle the existing plots).
- Total should comfortably support ~11.5 minutes of spoken delivery
  (matches the updated `speech.md` section timings: Introduction 2.5 min +
  Methods 4.5 min + Results 3.5 min + Discussion 1 min): no slide should
  require more than ~45-60 seconds of narration, per "How to give a talk"'s
  one-minute-per-slide minimum.
- Say each section's one-line spoken transition (marked in `speech.md`)
  out loud between sections, but do not print it on a slide — it is a
  verbal signpost so the audience doesn't experience the four sections as
  one undifferentiated block, not slide content.