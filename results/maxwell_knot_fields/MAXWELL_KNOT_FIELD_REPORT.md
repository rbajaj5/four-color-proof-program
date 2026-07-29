# Maxwell Knot Field Experiment

## Scope

This is a finite numerical audit of a published analytic family. It does not
claim a new Maxwell solution, a new knot type, or a new Four Color proof.

Kedia et al. start with Bateman potentials

```text
alpha = (r^2 - t^2 - 1 + 2 i z) / (r^2 - (t-i)^2)
beta  = 2(x - i y) / (r^2 - (t-i)^2)
F     = E + iB = grad(alpha^p) x grad(beta^q).
```

For positive integers `(p,q)`, the magnetic core lines are known `(p,q)`
torus knots or links. The script evaluates these equations directly and
checks them numerically on `cuda`.

## CUDA Audit

- Actual device: `cuda`
- CUDA used: `True`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- Torch: `2.12.0.dev20260408+cu128`
- Dtype: `float64 / complex128`
- Peak CUDA allocation: `2145792` bytes
- Total elapsed: `3.016` seconds

## Geometric Fixtures

| Fixture | `(p,q)` | Core components | Projected crossings | Min `|cos(B,tangent)|` | Projection diagnostic |
| --- | --- | ---: | ---: | ---: | --- |
| hopfion_1_1 | (1,1) | 2 | 0 | 0.99999999 | pass |
| linked_rings_1_2 | (1,2) | 2 | 0 | 0.99999996 | pass |
| trefoil_2_3 | (2,3) | 2 | 3 | 0.99999967 | pass |
| cinquefoil_2_5 | (2,5) | 2 | 5 | 0.99999879 | pass |
| four_component_2_2 | (2,2) | 4 | 0 | 0.99999999 | pass |
| torus_knot_3_4 | (3,4) | 2 | 8 | 0.99999875 | pass |

The fixed projection is used only to expose numerical degeneracies. Crossing
count in this table is not asserted to be minimal. The local crossing-conflict
graph colors overlapping diagnostic tasks; its color count is not a knot
invariant.

## Maxwell Residuals

| Fixture | `div B` relative mean | Faraday relative mean | Ampere relative mean | `E dot B` mean |
| --- | ---: | ---: | ---: | ---: |
| hopfion_1_1 | 2.797e-10 | 7.049e-10 | 7.064e-10 | 4.314e-17 |
| linked_rings_1_2 | 5.606e-10 | 1.440e-09 | 1.387e-09 | 1.032e-16 |
| trefoil_2_3 | 1.683e-09 | 5.638e-09 | 5.676e-09 | 1.741e-16 |
| cinquefoil_2_5 | 3.715e-09 | 1.161e-08 | 1.179e-08 | 1.752e-16 |
| four_component_2_2 | 1.119e-09 | 3.429e-09 | 3.337e-09 | 1.427e-16 |
| torus_knot_3_4 | 3.663e-09 | 1.386e-08 | 1.392e-08 | 1.282e-16 |

These are centered finite-difference residuals. The equations themselves are
analytic consequences of Bateman's construction.

## Static Permanent-Magnet Boundary

McDonald's note distinguishes a knotted material magnet from an arbitrary
knotted field in its exterior. In a simply connected, current-free, static
region, `curl B = 0` and `div B = 0`, hence `B = grad Phi` for harmonic
`Phi`. Along a nonstationary field line,

```text
d Phi / ds = |grad Phi|^2 > 0.
```

Such a gradient line cannot be a closed orbit. Closed knotted magnetic lines
therefore require a source/current region, nontrivial domain topology,
time-dependent null fields such as the Bateman family, or passage through
field zeros where reconnection can occur. This is why blindly varying a
static dipole parameter is the wrong search space.

## Sanders-Style Diagnostic Analogy

The Robertson-Sanders-Seymour-Thomas proof has two logically separate jobs:
its configurations are reducible, and the collection is unavoidable. Here we
borrow only that audit architecture. The finite bad-configuration catalog is:

1. near-zero field magnitude;
2. failure of core-line closure;
3. loss of field/tangent alignment;
4. near-tangent projected crossings;
5. nearly equal crossing depths; and
6. a nonplanar overlap graph for local diagnostic tasks.

Passing this catalog is a fixture check, not an unavoidability theorem for all
Maxwell fields. The 633 Four Color configurations are not being transferred
to electromagnetism.

## Goemans Conflict-Core Test

The exact abstraction uses a triangle of pairwise-incompatible detours. The
fractional point `(1/3, 2/5, 1/3)` satisfies every edge inequality and has
cost `58`. Every integral selection uses at most one detour and costs at least
`60`. The triangle is planar and 3-colorable.

This gives a clean negative lesson for the field-line analogy: coloring a
planar conflict graph can schedule whole-route checks, but does not make the
edge relaxation integral or preserve fractional cost. The missing inequality
is the odd-cycle constraint `z1 + z2 + z3 <= 1`, violated by `1/15`.

The repository verifies only this abstract conflict core. The recently
reported directed-path counterexample to the cost-strengthened Goemans
conjecture is not yet treated here as peer-reviewed mathematical fact.

## What Changed in the Proof Program

- **Upgraded to exact fixture:** Bateman `(p,q)` values provide a canonical
  Maxwell-constrained geometry generator, replacing arbitrary 3D knot
  drawings.
- **Falsified as a general bridge:** Four-colorability alone cannot certify a
  cost-preserving whole-field-line selection.
- **Still open:** whether a useful unavoidable family of local geometric
  pathologies can support a reduction theorem for a physically constrained
  class of knotted fields.

## Sources

- K. T. McDonald, *Can the Field Lines of a Permanent Magnet Be Tied in
  Knots?*: https://kirkmcd.princeton.edu/examples/knot.pdf
- H. Kedia et al., *Tying knots in light fields*:
  https://arxiv.org/abs/1302.0342
- H. Kedia, D. Peralta-Salas, and W. T. M. Irvine, *When do knots in light
  stay knotted?*: https://arxiv.org/abs/1706.06175
- N. Robertson, D. Sanders, P. Seymour, and R. Thomas, Four Color proof
  materials: https://thomas.math.gatech.edu/FC/fourcolor.html
- Y. Dinitz, N. Garg, and M. X. Goemans, *On the single-source unsplittable
  flow problem*: https://doi.org/10.1007/s004930050043
- Current planar SSUF formulation and Goemans Conjecture 1.3:
  https://doi.org/10.1007/s10107-026-02365-x
