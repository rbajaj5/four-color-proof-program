# Magnetic Braid MHD GPU Report

## Scope

This branch sweeps the analytic line-tied magnetic braid family introduced by
Wilmot-Smith, Hornig, and Pontin for model solar coronal loops. Each field is
a uniform vertical guide field plus localized toroidal flux rings. The
construction is analytically divergence-free.

The `diffusion_time` parameter applies the exact free heat evolution
`partial_t B = eta Laplacian(B)` to each Gaussian ring perturbation, with the
product `eta t` represented by the parameter. It is a resistive coarse-
graining subflow, **not** a complete resistive-MHD solution: velocity,
pressure, density, energy transport, and reconnection feedback are absent.

## Bundle Geometry

The individual-line observable is pairwise winding of neighboring field
lines. The bundle observable is the differential of the lower-to-upper
footpoint map. Its squashing factor `Q`, largest singular value, and
finite-length Lyapunov exponent quantify the wavefront-like deformation of a
small line bundle. This is the magnetic analogue adopted from the billiard
wavefront intuition; magnetic field lines remain line-tied and are not
specularly reflected.

The normalized magnetic tangent is also projected to the upper direction
sphere. Its spherical path length is a smooth analogue of Galperin's finite
angular budget for reflected rays. Spatial decimation and direction-space
arc loss are reported separately.

At each axial slice, the collection of distinct line positions lies in a
configuration-space complement of pair-collision diagonals. Pairwise winding
records local motion around those forbidden sets. The many direction tubes
have a Kakeya-like visual appearance, but this finite, constrained family is
not a Kakeya set and no Kakeya dimension claim is made.

## CUDA Audit

- Actual device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Dtype: `torch.float32`
- Configurations: `4`
- Field lines per configuration: `25`
- Steps per elementary cycle: `48`
- Total integrated line steps: `4800`
- Peak CUDA allocation: `33904128` bytes
- Total elapsed: `0.711` seconds

## Results by Braid Complexity and Diffusion

| Cycles | Diffusion | Mean max Q | Mean |winding| | Mean direction-sphere arc | Mean |integrated J_parallel| | Mean |J x B| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 4.965 | 0.1169 | 0.9507 | 1.341 | 0.05216 |
| 1 | 0.5 | 2.671 | 0.05409 | 0.4067 | 0.4836 | 0.02006 |

The largest squashing factor occurred for `n1_k1_b1_s1_rho0`:
`Q_max=6.1231` with maximum
finite-length Lyapunov exponent
`0.046927`.

The largest mean neighbor winding occurred for
`n1_k1_b1_s1_rho0`:
`0.14657` turns.

Because the normal field is the same constant on both boundaries, the
primary `Q` uses the exact determinant ratio `|B_z(start)/B_z(end)|=1`.
The unconstrained finite-difference `Q` is retained in the runs CSV as a
resolution diagnostic. At the base grid,
`2/4` configurations had maximum local
area-preservation residual at most `0.5`.

## Targeted Resolution Audit

| Configuration | Grid | Mean |winding| | Max Q | Max stretch | Max area residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| not run | - | - | - | - | - |

At the finest audit grid, the area-preservation residual was at most `0.1`
for none. These estimates are numerically resolved within the
declared audit. The residual remained above `0.5` for none.
For those high-gradient states, winding persists under refinement but `Q`
and maximum stretch continue to expose finer scales; they are not converged
point estimates.

## Multiscale Direction/Trajectory Ledger

Trajectory decimation is independent of physical diffusion. Galperin-style
angular-budget compression is represented by lost spherical tangent arc,
while braid information is checked by pairwise-winding error.

| Decimation factor | Mean direction arc loss | Max direction arc loss | Max winding error |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 2 | 0.005571 | 0.05818 | 8.941e-08 |
| 4 | 0.02559 | 0.2754 | 1.192e-07 |
| 8 | 0.1616 | 0.893 | 8.941e-08 |

The smallest nontrivial mean winding error was
`3.26e-10` at
coarse factor `2`. Across every row,
the largest winding error was `1.192e-07` turns and
the largest lost direction-sphere arc was `0.893`
radians. Thus winding is exceptionally stable here, while aggressive
decimation can hide substantial local bending.

## Physical Interpretation

- `Q` and finite-length Lyapunov stretching measure sensitivity of the
  footpoint map, not magnetic reconnection itself.
- `integrated_parallel_current` is relevant to three-dimensional
  reconnection, but a reconnection rate would require an electric field and
  a specified resistivity.
- `J x B` measures how far these analytic initial fields are from force-free
  balance. No ideal or magnetofrictional relaxation was performed.
- Pairwise winding of open, line-tied strands is a geometric statistic, not a
  closed-knot invariant.
- Magnetic energy is reported in normalized units and cannot be converted to
  solar-flare energy without a dimensional calibration.

## Performance

Mean group time was `0.710` seconds. The expensive operation
is batched RK4 field-line integration followed by current evaluation along
the trajectories. Only summaries and one selected trajectory bundle are
transferred to CPU.

## Decision

This branch is suitable for locating parameter regimes with simultaneously
large winding, large bundle stretching, and tolerable force imbalance.
The robust finding is a hierarchy: braid cycles increase winding, current,
and tangent-sphere variation, while free diffusion suppresses all three.
The largest undiffused winding states require adaptive spatial derivatives
before quantitative `Q` claims. A subsequent physics branch should select a
small number of states for magnetofrictional or resistive-MHD evolution
rather than increasing this static parameter grid indefinitely.
