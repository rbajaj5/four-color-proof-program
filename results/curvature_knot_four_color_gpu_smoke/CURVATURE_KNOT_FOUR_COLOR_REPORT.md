# Curvature-Knot Four Color GPU Report

## Question

Can Four Color diagnostics be attached to the curvature coarse graining of
closed magnetic knot cores in a way that records geometry rather than merely
coloring the regions of an ordinary knot diagram?

## Construction

The input curves are exact Bateman magnetic-core torus knots. The periodic
heat semigroup suppresses Fourier mode `k` by `exp(-tau k^2)`. At each RG
scale, a fixed generic projection is deposited onto the plane with either
integrated three-dimensional bending mass or signed projected turning as the
Gaussian-kernel weight.

The mixed curvature of that scalar potential chooses one diagonal in every
grid square. Adding one exterior vertex makes a sphere triangulation. The
map has exact chromatic number three exactly when all vertex degrees are
even; otherwise it has exact chromatic number four.

This is deliberately not ordinary knot-diagram face coloring, which is
checkerboard colorable and would make Four Color uninformative.

## Exact Discrete Interpretation

For interior vertices, odd degree is exactly the XOR of the four surrounding
diagonal choices. It is therefore a local `Z2` curvature defect.

Four vertex colors are encoded as `Z2 x Z2`. The color difference across
each primal edge is one of the three nonzero group elements. On the dual
cubic graph, the three incident labels at every triangular face XOR to zero.
The exported certificates therefore interpret every explicit four-coloring
as a nowhere-zero conserved Klein-four flux.

## Regular-Nodal-Set Bound

Let `g_h` be the continuous finite-difference mixed-curvature field whose
sign chooses the diagonals at mesh width `h`. An interior parity defect
requires both signs around that vertex, so a zero of `g_h` lies within an
`O(h)` neighborhood. If zero is a regular value and the nodal set has
uniformly bounded total length, a tubular-neighborhood count gives

`number of parity defects = O(h^(-1))`

and hence defect density `O(h)` on a two-dimensional grid. This is a direct
geometric consequence of the local XOR identity. It is conditional on the
regularity and uniform nodal-length assumptions.

## CUDA Audit

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Samples per knot: `128`
- RG scales: `3`
- Maps analyzed: `12`
- Peak CUDA allocation: `35065344` bytes
- Total elapsed: `1.004` seconds

## Results

| Knot | Dissection | Bandwidth | Grid | Four-color scales | Initial odd fraction | Final odd fraction | Final mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trefoil_T_2_3 | curvature_density | 0.16 | 9 | 3/3 | 0.585 | 0.610 | 0.078 |
| trefoil_T_2_3 | curvature_density | 0.16 | 17 | 3/3 | 0.386 | 0.379 | 0.082 |
| trefoil_T_2_3 | signed_turning | 0.16 | 9 | 3/3 | 0.537 | 0.512 | 0.062 |
| trefoil_T_2_3 | signed_turning | 0.16 | 17 | 3/3 | 0.352 | 0.359 | 0.055 |

All `4` representative Klein-flux certificates were nonzero
and conserved: `True`.

Across `6` fixed knot/channel/bandwidth/RG configurations,
the fitted odd-defect count exponent had mean `1.245`,
range `1.138` to `1.309`.
Curve-supported defects predict exponent one; area-filling defects predict
two. The fit supports, but does not certify, the regular-nodal-set picture.

The largest final diagonal displacement was
`0.082` for
`trefoil_T_2_3`,
`curvature_density`, bandwidth
`0.16`. The smallest was
`0.055` for
`trefoil_T_2_3`,
`signed_turning`.

The coloring ceiling itself is not the differentiator: most generic
triangulations require four colors. The useful observables are defect
density, defect correlation, and transport of the curvature-selected
diagonals under coarse graining.

## What This Upgrades

1. Four colors become a `Z2 x Z2` potential whose differences define a
   conserved dual flux.
2. Odd-degree obstructions become local `Z2` curvature defects.
3. RG transport asks whether the same defect/flux organization survives
   when fine knot curvature is removed.

These are exact finite statements attached to declared discretizations.
They do not strengthen the Four Color Theorem itself.

## Claim Boundary

- The potential map is not an intrinsic invariant of a knot; it depends on
  projection, bandwidth, channel, grid, and compactification.
- Heat flow can leave the original knot type after a self-intersection.
- Four-colorability does not classify a knot or magnetic field.
- Klein-four labels are discrete graph flows, not physical magnetic flux.
- No new Four Color or Maxwell theorem is claimed.
