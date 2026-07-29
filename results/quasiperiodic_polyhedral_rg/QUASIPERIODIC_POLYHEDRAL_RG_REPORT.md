# Quasiperiodic Polyhedral Knot RG Report

## Replacement for the Jenga Model

This experiment separates four coupled channels:

1. **metric flow:** Fourier heat time `tau` smooths the curve;
2. **spatial inflation:** a Fibonacci substitution tiling is inflated through
   five declared levels; and
3. **connection compression:** total turning and tangent holonomy are retained
   even when the detailed cell itinerary is contracted; and
4. **force susceptibility:** each occupied cell carries a normalized
   positive- and negative-curvature response envelope.

The 3D patch is the Cartesian product of three Fibonacci interval tilings,
globally transformed by a golden-ratio affine shear and subdivided by a
face-compatible Freudenthal triangulation. Every cell is a convex tetrahedron,
the patch is face-to-face, and inflation preserves quasiperiodic order.

This is an exact product-Fibonacci prototype, **not** an
Ammann-Kramer-Neri icosahedral tiling.

## Galperin Figure Interface

Figures 6-8 in Galperin's *Convex Polyhedra Without Simple Closed Geodesics*
provide the discrete differential-geometric interface:

- signed intersections define differences of vertex indices;
- stereographic projection turns those indices into planar winding numbers;
- successive faces are developed into a plane so a surface geodesic becomes
  straight; and
- the polyhedral Gauss-Bonnet ledger is
  `delta_L + sum_v Delta_v ind_L(v) = 2 pi k`.

The original schematic in this folder adapts those operations to the RG
pipeline. Source: https://www.ux1.eiu.edu/~ggalperin/papers/GeodesRCD.pdf

## Space-Form Force Channel

Coulton and Galperin compute the exact coupling force needed to keep two
constant-speed paths separated by distance `d` in a smooth two-dimensional
space form:

```text
K > 0:  F =  2 m v^2 sqrt(K) tan(sqrt(K) d / 2)
K < 0:  F = -2 m v^2 sqrt(-K) tanh(sqrt(-K) d / 2).
```

For small `d`, both have the signed approximation
`F = m v^2 K d + O(d^3)`. The implementation verifies this limit and records
a normalized local-chart susceptibility. Each sampled tetrahedron is assigned
a chart radius equal to its diameter and a virtual separation equal to one
half that diameter. Inflation therefore renormalizes the response scale along
with occupancy and curvature load. The `21` exact calibration
states are written to `space_form_force_response.csv`.

This channel is a geometric response model. The ambient tiling remains
Euclidean and does not physically exert the reported forces.

## CUDA Audit

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Curves: `4`
- Curve samples: `512`
- Metric times: `23`
- Inflation levels: `5`
- State rows: `115`
- Peak CUDA allocation: `46050816` bytes
- Elapsed: `8.816` seconds

## Results

| Fixture | State | Crossings | Fine cells | Coarse cells | Coarse/fine | Fine nonzero-winding fraction | Tangent holonomy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hopf_core_unknot | initial | 0 | 192 | 37 | 0.1927 | 0.3698 | 0 |
| trefoil_2_3 | initial | 3 | 351 | 84 | 0.2393 | 0.6695 | 2.825 |
| trefoil_2_3 | projection_fall_tau=0.602 | 0 | 205 | 39 | 0.1902 | 0.5024 | -3.101 |
| cinquefoil_2_5 | initial | 5 | 415 | 115 | 0.2771 | 0.7084 | 1.024 |
| cinquefoil_2_5 | projection_fall_tau=2.469 | 3 | 177 | 40 | 0.2260 | 0.4859 | -1.885e-26 |
| torus_knot_3_4 | initial | 8 | 410 | 115 | 0.2805 | 0.8073 | 2.917 |
| torus_knot_3_4 | projection_fall_tau=0.1468 | 4 | 355 | 61 | 0.1718 | 0.4507 | 3.011 |

The occupied-cell count and itinerary entropy contract under inflation, but
the tangent holonomy is retained as separate connection data. Curvature load
is also aggregated by cell rather than discarded. This is the central
improvement over the Fourier tower: coarse geometric appearance is no longer
the entire state. The space-form response decreases as cell diameter grows,
while the spherical response is slightly stronger than the equal-magnitude
hyperbolic response because `tan(x) > tanh(x)` for positive `x`.

## Defect Spectrum

The eight Fibonacci box types and six Freudenthal simplices produce 48 local
tetrahedral prototypes. Every defect vector sums to `4 pi`, as required by
Descartes' formula. A bounded integer search with coefficients in `[-3,3]`
found minimum normalized residual `0.0002104` and
`0` exact-within-tolerance prototype relations.

This finite search does not prove rational independence. In particular,
symmetry-induced equal defects can create exact small relations, so Galperin's
generic-tetrahedron obstruction cannot be assigned automatically to every
cell.

## Scale Semantics

- **Microscopic:** individual tetrahedra, face transitions, local curvature
  loads, and local defect spectra.
- **Mesoscopic:** recurrent cell words, transition entropy, projected winding
  proxies, and regions where curvature concentrates.
- **Macroscopic:** contracted dual itinerary, total turning, tangent holonomy,
  and the knot's independent projection diagnostics.

## Claim Boundary

The knot passes through cell interiors in this first prototype. Therefore the
reported winding values around cell centers are ambient projected winding
proxies, not Galperin vertex indices of a broken line constrained to one
polyhedron surface. Local tetrahedral defects are likewise not summed as if
interior tiling vertices carried ambient curvature; face-to-face Euclidean
cells cancel that curvature globally.

The force susceptibility is not inferred from those discrete defects. The
Coulton-Galperin theorem assumes smooth constant sectional curvature and
equidistant particle paths; neither is supplied by an ambient Euclidean
tetrahedral itinerary. The recorded force is a declared chart envelope only.

No knot classification, geodesic-existence theorem, or new quasiperiodic
tiling theorem is claimed. The next rigorous step is to route each
entry-to-exit segment along the corresponding convex cell boundary, unfold
the traversed faces, and then evaluate the actual defect-index identity.
