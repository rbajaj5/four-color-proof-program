# Four Color Proof Program

> **Status:** This repository does not claim a new proof of the Four Color
> Theorem. It contains an exact finite verification engine and a documented
> program for searching for a more conceptual knot-theoretic proof.

## Purpose

The established Four Color proof is computer-assisted. Robertson, Sanders,
Seymour, and Thomas reduce it to an unavoidable set of 633 configurations and
32 discharging rules.

This project explores a different interface:

```text
plane maps
  <-> bridgeless cubic plane graphs
  <-> Tait edge 3-colorings
  <-> Penrose tensor contractions
  <-> smoothing states of alternating link diagrams
```

The final equivalence uses the 2026 Penrose-Kauffman formulation of Kauffman,
Silver, and Williams: Four Color is equivalent to every reduced, prime,
alternating, bigon-free plane link diagram having a 3-colorable smoothing
state.

## What Is Implemented

The exact Python engine:

- enumerates proper Tait edge 3-colorings of cubic rotation systems;
- contracts the signed Levi-Civita tensors;
- verifies the Penrose identity on small plane fixtures;
- records bridge and nonplanar controls; and
- uses integer arithmetic throughout.

Current exact fixtures:

| Fixture | Tait colorings | Penrose contraction | Role |
| --- | ---: | ---: | --- |
| Theta plane multigraph | 6 | 6 | Planar calibration |
| `K4` | 6 | 6 | Simple planar calibration |
| Cube graph | 24 | 24 | Bridgeless planar fixture |
| Triangular prism | 6 | 6 | Bridgeless planar fixture |
| Planar bridge control | 0 | 0 | Bridge obstruction |
| `K3,3` | 12 | 0 | Naive Penrose planarity boundary |
| Petersen graph | 0 | 0 | Nonplanar uncolorable control |

The `K3,3` result is intentional: its genuine Tait count is 12, while the
unmodified plane Penrose contraction cancels to zero. This prevents the
implementation from silently treating a plane identity as embedding-free.

The second exact experiment resolves both local smoothings of every edge in a
small cubic-graph catalog and reconstructs every parent coloring that actually
lifts. It tests the precise partition

```text
Col_3(G) = Lift_0(G,e) disjoint-union Lift_1(G,e)
```

and records colorable smoothing children whose colorings do not lift. Those
rows are counterexamples to the tempting reduction rule "colorable child
implies colorable parent." The exact replacement-edge criterion is recorded:
a child coloring lifts precisely when its two replacement edges are assigned
different colors.

The third experiment uses exact Bateman potentials to sample electromagnetic
torus-knot core lines on CUDA. It:

- verifies free-space Maxwell and null-field residuals;
- checks field/tangent alignment for known Hopf, trefoil, cinquefoil, and
  `T(3,4)` fixtures;
- records a finite catalog of projection degeneracies;
- renders the full 3D magnetic core sets; and
- verifies an exact `58` versus `60` conflict-core example showing why planar
  four-colorability does not make fractional whole-route selection integral.

This is a physically constrained geometry fixture, not a new Maxwell
solution or new knot discovery. See
[`MAXWELL_KNOT_FIELD_PROGRAM.md`](MAXWELL_KNOT_FIELD_PROGRAM.md).

The fourth experiment treats a closed curve as a tower of Fourier modes and
applies a curvature-inspired parameter-space heat flow

```text
partial_tau gamma = partial_s^2 gamma.
```

Mode `k` decays as `exp(-tau k^2)`, so fine, high-curvature blocks disappear
first. The CUDA run follows projection crossings, nonlocal clearance, bending
energy, and effective mode count for four Bateman core fixtures. It also
records the elementary exact result that the centered, rescaled flow converges
to the first surviving harmonic. When that harmonic is a rank-two first mode,
a nontrivial knot must encounter the non-embedding discriminant at finite RG
time. A higher first mode instead gives a nonembedded multiple-cover limit;
that convergence alone does not imply finite-time contact. This is an abstract
RG/coarse-graining model, not a mechanical string simulation or a new knot
theorem. See the generated
[`CURVATURE_RG_KNOT_COLLAPSE_REPORT.md`](results/curvature_rg_knot_collapse/CURVATURE_RG_KNOT_COLLAPSE_REPORT.md).

## The Missing Lemma

For a link diagram, smoothing choices form a Boolean cube. The
Penrose-Kauffman paper identifies colorable states with strict local maxima of
the number of state components.

Every finite cube has a local maximum, but that observation does not prove
Four Color. The missing statement is:

```text
At least one strict local maximum has a 3-colorable component graph.
```

Three proof programs are tracked in
[`FOUR_COLOR_KNOT_PROOF_PROGRAM.md`](FOUR_COLOR_KNOT_PROOF_PROGRAM.md):

1. positivity-preserving smoothing reductions;
2. a discrete Morse certificate on the smoothing cube; and
3. a manifestly positive state-sum decomposition.

Each remains open.

## Run

Python 3.11 or newer is recommended.

```bash
python -m pip install -e ".[test,experiments]"
python scripts/check_four_color_penrose_fixtures.py
python scripts/check_cubic_edge_smoothing_lifts.py
python scripts/check_maxwell_knot_fields_gpu.py
python scripts/check_curvature_rg_knot_collapse_gpu.py
python -m pytest tests -q
```

Generated artifacts are written to:

```text
results/four_color_penrose_program/
results/cubic_edge_smoothing/
results/maxwell_knot_fields/
results/curvature_rg_knot_collapse/
```

## Repository Layout

```text
src/four_color_penrose.py
    Exact cubic-rotation-system and Penrose/Tait arithmetic.

scripts/check_four_color_penrose_fixtures.py
    Reproducible fixture and report generator.

tests/test_four_color_penrose.py
    Exact regression tests and scope controls.

src/cubic_edge_smoothing.py
    Exact cubic edge smoothings and labeled coloring reconstruction.

scripts/check_cubic_edge_smoothing_lifts.py
    Deterministic local-reduction obstruction experiment.

tests/test_cubic_edge_smoothing.py
    Exact partition and nonliftability tests.

src/maxwell_knot_fields.py
    Bateman potentials, magnetic core curves, and Maxwell residuals.

src/goemans_conflict_fixture.py
    Exact fractional-versus-integral conflict-core arithmetic.

scripts/check_maxwell_knot_fields_gpu.py
    CUDA fixture runner, projection diagnostics, plots, and reports.

tests/test_maxwell_knot_fields.py
    Closure, component, tangency, and Maxwell regression tests.

tests/test_goemans_conflict_fixture.py
    Exact 58-versus-60 arithmetic and feasibility tests.

src/curvature_rg_flow.py
    Fourier heat coarse-graining and scale-free geometric diagnostics.

scripts/check_curvature_rg_knot_collapse_gpu.py
    CUDA RG-flow runner, crossing diagnostics, plots, and report generator.

tests/test_curvature_rg_flow.py
    Semigroup, spectral monotonicity, and terminal-harmonic tests.

FOUR_COLOR_KNOT_PROOF_PROGRAM.md
    Mathematical proof-search program and claim boundaries.

MAXWELL_KNOT_FIELD_PROGRAM.md
    Maxwell equations, static obstruction, and analogy boundaries.

results/four_color_penrose_program/
    Committed generated CSV, JSON, and Markdown artifacts.

results/cubic_edge_smoothing/
    Committed edge-level, graph-level, audit, and report artifacts.

results/maxwell_knot_fields/
    CUDA diagnostics, CSVs, audit JSON, report, and 3D figures.

results/curvature_rg_knot_collapse/
    CUDA flow traces, summary, audit JSON, report, and 3D snapshots.
```

## Evidence Policy

- Passing finite fixtures is not a proof of Four Color.
- A reformulation equivalent to Four Color is not progress unless it yields a
  new proved reduction, positivity mechanism, or formal certificate.
- Nonplanar tensor contractions are not interpreted using the plane Penrose
  identity.
- Computational discoveries must be separated from unavoidability,
  termination, and lifting proofs.
- Known Bateman torus-knot fixtures are not reported as new electromagnetic
  solutions or new knot types.
- Four-coloring a diagnostic conflict graph does not certify weighted
  whole-trajectory selection.
- Heat-flow crossing changes are one-sided diagnostics, not complete knot
  classifications or topology-preserving simplifications.

## Primary Sources

- Kauffman, Silver, and Williams, *The Penrose-Kauffman Polynomial*:
  https://arxiv.org/abs/2604.16635
- Kauffman, *A State Calculus for Graph Coloring*:
  https://arxiv.org/abs/1511.06844
- Robertson, Sanders, Seymour, and Thomas, Four Color information:
  https://thomas.math.gatech.edu/FC/fourcolor.html
- Machine-readable configuration and discharging materials:
  https://thomas.math.gatech.edu/FC/ftpinfo.html
- Kedia et al., *Tying knots in light fields*:
  https://arxiv.org/abs/1302.0342
- McDonald, *Can the Field Lines of a Permanent Magnet Be Tied in Knots?*:
  https://kirkmcd.princeton.edu/examples/knot.pdf

## Provenance

This standalone repository contains only the Four Color/Penrose proof program
extracted from a broader research codebase. Unrelated experiments,
conjectures, and topology-validation outputs are not included.
