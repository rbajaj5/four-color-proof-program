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

The CUDA smoothing-landscape experiment exhaustively evaluates the mod-2
Laplacian nullity on Boolean smoothing cubes. It reconstructs only strict
local maxima on the CPU and tests their component graphs exactly. Odd wheels
through `W14` satisfy a closed-form pattern, while small planar-atlas controls
falsify the tempting universal rule that every nonzero maximum should be
3-colorable. This is a bounded structural diagnostic, not a new proof.

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

A source-gating companion isolates the classical complex Hopf-fibration
layer from a recent non-peer-reviewed gauge-gravity proposal. CUDA checks
rank-one projectors on `CP^1`, `CP^2`, and `CP^4`, common-`U(1)` phase
invariance, the projective distance identity, the first Chern number of
`S^3 -> CP^1`, and unit linking of distinct Hopf fibers. The proposal's
unification and particle-spectrum claims are recorded but not imported as
established results. See
[`HOPF_FIBRATION_SOURCE_GATE_REPORT.md`](results/hopf_fibration_gate_gpu/HOPF_FIBRATION_SOURCE_GATE_REPORT.md)
and [`NIETTU_HOPF_SOURCE_AUDIT.md`](notes/NIETTU_HOPF_SOURCE_AUDIT.md).

The follow-on Hopf-to-Hex experiment pulls those validated projective
coordinates back through the Bateman map, samples randomly oriented planar
windows, and thresholds each Hopf-map component into a completed Hex board.
Random common `U(1)` phase rotations provide a gauge control: both the
projective field and every thresholded board must remain unchanged. This
tests a finite planar connectivity signature of Hopf texture without
claiming that a slice recovers three-dimensional knot type. See
[`HOPF_HEX_SLICE_GPU_REPORT.md`](results/hopf_hex_slices_gpu/HOPF_HEX_SLICE_GPU_REPORT.md).
The associated
[`KAN model-selection audit`](notes/KAN_MODEL_SELECTION_AUDIT.md) explains
why no learned surrogate is used for these exact and one-dimensional
observables, and defines a compute-matched gate for any later
symbolic-regression use.

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

A second CUDA pass resolves that flow into declared wavelength bands:
macroscopic modes `1-2`, mesoscopic modes `3-8`, and microscopic modes `9+`.
It distinguishes absolute energy depletion from relative structural share.
The resulting cascade shows why microscopic roughness can disappear long
before the knot diagram changes, and why very small higher modes can remain
topologically decisive near a multiply covered limiting curve. See
[`CURVATURE_RG_MULTISCALE_JENGA_REPORT.md`](results/curvature_rg_knot_collapse/CURVATURE_RG_MULTISCALE_JENGA_REPORT.md).

A third coarse-graining pass replaces the single Jenga tower with a finite
product-Fibonacci environment. Golden-ratio interval tilings on three axes are
affinely sheared, and every resulting box is subdivided face-to-face into six
convex tetrahedra. The metric RG time `tau` smooths the knot, while an
independent inflation level coarsens the environment. A retained itinerary
ledger records occupied cells, cell transitions and entropy, curvature
concentration, projected winding about cell centers, and tangent holonomy. A
fourth, mechanics-aware channel records the exact Coulton-Galperin
constant-curvature force response as a normalized local-chart susceptibility.
This separates geometric simplification from loss of combinatorial history
and from scale-dependent response.
The construction is a reproducible quasiperiodic prototype, not a claim to be
the Ammann-Kramer-Neri tiling. Its Galperin-inspired defect calculations are
local cell diagnostics; the ambient knot does not yet satisfy the hypotheses
of Galperin's surface-geodesic identities. See
[`QUASIPERIODIC_POLYHEDRAL_RG_REPORT.md`](results/quasiperiodic_polyhedral_rg/QUASIPERIODIC_POLYHEDRAL_RG_REPORT.md).

The next CUDA branch moves from prescribed electromagnetic knot cores to the
analytic line-tied magnetic-braid family of Wilmot-Smith, Hornig, and Pontin.
A uniform guide field is perturbed by alternating Gaussian toroidal flux
rings. The experiment sweeps 108 braid/diffusion configurations and records
neighbor winding, footpoint-map squashing and stretching, current, Lorentz
force, divergence residuals, and tangent motion on the direction sphere.
An exact free-diffusion subflow supplies a controlled coarse-graining axis;
it is not a full resistive-MHD simulation. Targeted 25x25 and 49x49 audits
separate converged diffused mappings from undiffused states whose `Q` layers
remain finer than the sampling grid. See
[`MAGNETIC_BRAID_MHD_REPORT.md`](results/magnetic_braid_mhd_gpu/MAGNETIC_BRAID_MHD_REPORT.md).

The fractional-field CUDA branch turns the scaling idea into a finite planar
map experiment. It samples cutoff fractional Gaussian surfaces with Hurst
parameter `H`, uses the sign of the discrete mixed curvature in each square
to select a diagonal, and compactifies the boundary with one exterior
vertex. Every output is therefore a sphere triangulation. Such a
triangulation is 3-colorable exactly when every vertex has even degree;
otherwise its exact chromatic number is four. This gives a certified local
statistic, the odd-degree or **Four-color frustration density**, without
mistaking a rendered palette for evidence.

The 3,840-map CUDA sweep found that increasing `H` from `0.1` to `0.9`
increased fine-grid curvature-sign agreement from `0.4087` to `0.5040` and
decreased 17-to-65 scale-transport mismatch from `0.3303` to `0.2059`.
Every stochastic sample was exactly four-chromatic, while exact checkerboard
and one-flip fixtures exercise the three- and four-color cases. This is a
finite-cutoff construction inspired by Cao and Sheffield's fractional
Gaussian forms; it does not construct a canonical continuum coloring. See
[`FRACTIONAL_FIELD_FOUR_COLOR_REPORT.md`](results/fractional_field_four_color_gpu/FRACTIONAL_FIELD_FOUR_COLOR_REPORT.md).

The finite Fourier model also admits an analytic explanation. Neighboring
mixed-curvature values are jointly Gaussian, so their sign agreement is
exactly `1/2 + asin(rho)/pi`; `rho` is an explicit filtered spectral sum.
Across the full run, the largest absolute difference between the predicted
and observed mean agreement was below `0.003`. At interior vertices, the
degree parity is exactly the XOR of the four surrounding diagonal choices,
or equivalently a local `Z2` curvature defect.

The Hex/Y branch adds topology-preserving planar connectivity observables.
An exhaustive CUDA census verifies the Karlin-Peres triangular-majority
reduction through all 2,097,152 side-six Y boards. A separate rhombic Hex
census enumerates all 33,554,432 side-five boards and records the exact
crossing reliability polynomial. Its derivative at `p=1/2` is the expected
pivotal-cell count under independent coloring. Larger-board sampling then
compares i.i.d. cells with thresholded fractional-Gaussian fields. On the
`7 x 7` board, increasing local neighbor agreement from approximately `0.50`
to `0.73` broadens the 25-to-75 percent crossing window from `0.1601` to
`0.2513`. These are finite planar connectivity diagnostics, not new Hex
theorems or three-dimensional knot certificates. See
[`HEX_CROSSING_GPU_REPORT.md`](results/hex_crossing_gpu/HEX_CROSSING_GPU_REPORT.md)
and
[`HEX_CORRELATED_FIELD_GPU_REPORT.md`](results/hex_correlated_fields_gpu/HEX_CORRELATED_FIELD_GPU_REPORT.md).

The magnetic-knot bridge applies that parity construction to scalar
potentials deposited from the curvature of heat-coarse-grained Bateman
magnetic core knots. Its 432-map CUDA sweep covers the trefoil, cinquefoil,
and `T(3,4)` cores, two curvature channels, two bandwidths, nine RG scales,
and four resolutions. Four colors are encoded as `Z2 x Z2`, so edge
differences produce a nonzero conserved Klein-four flow on the dual cubic
graph. All 24 representative certificates passed.

Unlike the rough fractional fields, the knot potentials had odd-defect
counts scaling with mean exponent `1.188` rather than the area-filling
exponent two. This agrees with a conditional geometric bound: if the
finite-difference mixed-curvature zero set is regular with uniformly bounded
length, parity defects lie in its `O(h)` tube, giving `O(h^-1)` defects and
`O(h)` density. See
[`CURVATURE_KNOT_FOUR_COLOR_REPORT.md`](results/curvature_knot_four_color_gpu/CURVATURE_KNOT_FOUR_COLOR_REPORT.md).

A generated
[`Four Color special-case dictionary`](results/four_color_special_case_dictionary/FOUR_COLOR_SPECIAL_CASE_DICTIONARY.md)
separates bipartite, triangle-free, checkerboard knot-region, Eulerian
triangulation, Tait/Klein-flow, curvature-map, and arbitrary planar cases.
Its Goemans-style weighted control keeps the exact 58/60 integrality gap
visible: planar colorability can schedule conflict-free layers without
lifting a fractional weighted choice to an integral whole-route choice.
The dictionary now also records Zamir's fixed-palette theorem as a generic
algorithmic fallback. For a remaining arbitrary planar graph, the useful
query is 3-colorability: a YES gives `chi <= 3`, while a NO and Four Color
give `chi = 4`. Structured cases continue to use bipartite, checkerboard,
parity, or Klein-flow certificates before any exponential search. The
repository records Zamir's theorem and decision-to-search route but does not
reimplement the paper's randomized sub-`2^n` algorithm.

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
py -3.12 scripts/check_penrose_smoothing_landscape_gpu.py
python scripts/check_maxwell_knot_fields_gpu.py
python scripts/check_curvature_rg_knot_collapse_gpu.py
python scripts/check_curvature_rg_multiscale_gpu.py
python scripts/check_quasiperiodic_polyhedral_rg_gpu.py
python scripts/check_magnetic_braid_mhd_gpu.py --smoke
python scripts/check_magnetic_braid_mhd_gpu.py
python scripts/check_fractional_field_four_color_gpu.py --smoke
python scripts/check_fractional_field_four_color_gpu.py
python scripts/check_curvature_knot_four_color_gpu.py --smoke
python scripts/check_curvature_knot_four_color_gpu.py
python scripts/build_four_color_special_case_dictionary.py
python -m pytest tests -q
```

Generated artifacts are written to:

```text
results/four_color_penrose_program/
results/cubic_edge_smoothing/
results/penrose_smoothing_landscape_gpu/
results/maxwell_knot_fields/
results/curvature_rg_knot_collapse/
results/quasiperiodic_polyhedral_rg/
results/magnetic_braid_mhd_gpu_smoke/
results/magnetic_braid_mhd_gpu/
results/fractional_field_four_color_gpu_smoke/
results/fractional_field_four_color_gpu/
results/curvature_knot_four_color_gpu_smoke/
results/curvature_knot_four_color_gpu/
results/four_color_special_case_dictionary/
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

src/penrose_smoothing_landscape.py
    Exact plane-state reconstruction and bounded coloring checks.

scripts/check_penrose_smoothing_landscape_gpu.py
    Exhaustive CUDA nullity and strict-local-maximum census.

tests/test_penrose_smoothing_landscape.py
    Exact state/nullity agreement and counterexample controls.

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

scripts/check_curvature_rg_multiscale_gpu.py
    CUDA macro/meso/micro spectral ledger and cascade visualization.

tests/test_curvature_rg_flow.py
    Semigroup, spectral partition, monotonicity, and terminal-harmonic tests.

src/quasiperiodic_polyhedral_rg.py
    Product-Fibonacci tetrahedral hierarchy and retained itinerary invariants.

scripts/check_quasiperiodic_polyhedral_rg_gpu.py
    CUDA metric/environment RG runner, visualizations, and report generator.

tests/test_quasiperiodic_polyhedral_rg.py
    Axis, tetrahedral defect, inflation, turning, holonomy, and winding tests.

src/magnetic_braid_mhd.py
    Divergence-free line-tied braid fields, exact Gaussian diffusion,
    field-line integration, winding, mapping, and direction-sphere metrics.

scripts/check_magnetic_braid_mhd_gpu.py
    CUDA parameter sweep, coarse-graining ledger, refinement audit, figures,
    and report generator.

tests/test_magnetic_braid_mhd.py
    Divergence, diffusion, mapping, current, winding, and spherical-path tests.

src/fractional_field_four_color.py
    CUDA fractional Gaussian surfaces, curvature triangulations, exact
    parity classification, DSATUR certificates, and scale transport.

scripts/check_fractional_field_four_color_gpu.py
    CUDA Hurst-parameter sweep, exact fixtures, visualizations, and report.

tests/test_fractional_field_four_color.py
    Fourier, compactification, parity, coloring, transport, and scaling tests.

src/curvature_knot_four_color.py
    Curvature deposition, knot-map dissections, faces, and Klein-flow checks.

scripts/check_curvature_knot_four_color_gpu.py
    CUDA knot/RG/grid sweep, scaling fits, flow certificates, and figures.

tests/test_curvature_knot_four_color.py
    Curvature-channel, potential, face-count, and conserved-flow tests.

src/four_color_special_cases.py
    Claim-bounded special-case registry, planar decision hierarchy,
    fixed-palette solver diagnosis, and weighted-lift warning.

scripts/build_four_color_special_case_dictionary.py
    Generate the special-case, Zamir-solver, hierarchy, and weighted
    diagnostics with two matrices.

tests/test_four_color_special_cases.py
    Registry integrity, planar theorem routing, fixed-palette scope, and
    exact Goemans-style control tests.

FOUR_COLOR_KNOT_PROOF_PROGRAM.md
    Mathematical proof-search program and claim boundaries.

MAXWELL_KNOT_FIELD_PROGRAM.md
    Maxwell equations, static obstruction, and analogy boundaries.

results/four_color_penrose_program/
    Committed generated CSV, JSON, and Markdown artifacts.

results/cubic_edge_smoothing/
    Committed edge-level, graph-level, audit, and report artifacts.

results/penrose_smoothing_landscape_gpu/
    CUDA audit, exhaustive maxima, fixture summaries, timing, and report.

results/maxwell_knot_fields/
    CUDA diagnostics, CSVs, audit JSON, report, and 3D figures.

results/curvature_rg_knot_collapse/
    CUDA flow and scale traces, summaries, audits, reports, and 3D snapshots.

results/quasiperiodic_polyhedral_rg/
    CUDA hierarchy ledger, defect library, audit, report, and schematics.

results/magnetic_braid_mhd_gpu/
    CUDA field-line sweep, physical diagnostics, multiscale ledger,
    grid-refinement audit, report, and phase/trajectory figures.

results/fractional_field_four_color_gpu/
    CUDA finite-map sweep, exact fixture certificates, multiscale transport
    tables, device audit, report, and coloring figures.

results/curvature_knot_four_color_gpu/
    CUDA curvature/RG map sweep, defect scaling, Klein-flow certificates,
    reports, and explicit colorings.

results/four_color_special_case_dictionary/
    Special-case registry, fixed-palette solver hierarchy, weighted-lift
    diagnosis, report, and matrices.
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
- Projected winding around an ambient cell center is not identified with
  Galperin's vertex index unless the curve has first been placed on an
  appropriate polyhedral surface.
- A bounded integer-relation search among tetrahedral angle defects is a
  finite diagnostic, not a proof of rational independence.
- The space-form force susceptibility is a declared local-chart response; it
  is not a physical force produced by the ambient Euclidean tiling.
- Heat-flow crossing changes are one-sided diagnostics, not complete knot
  classifications or topology-preserving simplifications.
- Free diffusion of the analytic magnetic perturbation is not full
  resistive MHD, and large `Q` values are not reconnection rates.
- Pairwise winding of open line-tied strands is not a closed-knot invariant.
- Finite-difference `Q` estimates whose area-preservation residual has not
  converged are reported as unresolved fine structure, not point estimates.
- The direction-tube picture is Kakeya-like only as an incidence analogy;
  the finite collision-diagonal complement is not a Kakeya set.
- A cutoff fractional Gaussian surface is not literally an infinite
  coloring. Its Hurst parameter controls scaling and regularity, not
  chromatic or Hausdorff dimension by itself.
- Curvature-selected diagonals are a declared finite discretization. Four
  Color certifies each compactified triangulation but does not supply a
  canonical measurable coloring compatible across cutoff scales.
- The regular-nodal-set defect bound is conditional on transversality and a
  uniform nodal-length bound; fitted scaling exponents do not prove those
  assumptions.
- Klein-four edge labels are exact discrete graph flows, not physical
  magnetic fluxes.
- A planar coloring does not imply that a fractional weighted solution lifts
  to an integral route or resource allocation.
- Zamir's fixed-palette result is a theoretical colorability fallback, not a
  weighted solver or a reimplemented practical coloring engine.

## Primary Sources

- Kauffman, Silver, and Williams, *The Penrose-Kauffman Polynomial*:
  https://arxiv.org/abs/2604.16635
- Kauffman, *A State Calculus for Graph Coloring*:
  https://arxiv.org/abs/1511.06844
- Robertson, Sanders, Seymour, and Thomas, Four Color information:
  https://thomas.math.gatech.edu/FC/fourcolor.html
- Machine-readable configuration and discharging materials:
  https://thomas.math.gatech.edu/FC/ftpinfo.html
- Zamir, *k-Coloring is Faster than Computing the Chromatic Number*:
  https://arxiv.org/abs/2607.25973
- Hou, Ji, Zhang, and Stefanidis, *Kolmogorov-Arnold Networks: A Critical
  Assessment of Claims, Performance, and Practical Viability*:
  https://arxiv.org/abs/2407.11075
- Kedia et al., *Tying knots in light fields*:
  https://arxiv.org/abs/1302.0342
- McDonald, *Can the Field Lines of a Permanent Magnet Be Tied in Knots?*:
  https://kirkmcd.princeton.edu/examples/knot.pdf
- Wilmot-Smith, Hornig, and Pontin, *Magnetic Braiding and
  Quasi-Separatrix Layers*: https://arxiv.org/abs/0907.3820
- Wilmot-Smith, Hornig, and Pontin, *Magnetic Braiding and Parallel Electric
  Fields*: https://arxiv.org/abs/0810.1415

## Provenance

This standalone repository contains only the Four Color/Penrose proof program
extracted from a broader research codebase. Unrelated experiments,
conjectures, and topology-validation outputs are not included.
