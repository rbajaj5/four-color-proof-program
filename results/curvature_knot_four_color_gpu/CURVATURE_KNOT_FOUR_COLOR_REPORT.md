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
- Samples per knot: `512`
- RG scales: `9`
- Maps analyzed: `432`
- Peak CUDA allocation: `85767168` bytes
- Total elapsed: `1.673` seconds

## Results

| Knot | Dissection | Bandwidth | Grid | Four-color scales | Initial odd fraction | Final odd fraction | Final mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cinquefoil_T_2_5 | curvature_density | 0.1 | 9 | 9/9 | 0.463 | 0.390 | 0.359 |
| cinquefoil_T_2_5 | curvature_density | 0.1 | 17 | 9/9 | 0.441 | 0.359 | 0.336 |
| cinquefoil_T_2_5 | curvature_density | 0.1 | 33 | 9/9 | 0.250 | 0.206 | 0.332 |
| cinquefoil_T_2_5 | curvature_density | 0.1 | 65 | 9/9 | 0.131 | 0.110 | 0.337 |
| cinquefoil_T_2_5 | curvature_density | 0.18 | 9 | 9/9 | 0.488 | 0.585 | 0.359 |
| cinquefoil_T_2_5 | curvature_density | 0.18 | 17 | 9/9 | 0.414 | 0.359 | 0.336 |
| cinquefoil_T_2_5 | curvature_density | 0.18 | 33 | 9/9 | 0.213 | 0.198 | 0.355 |
| cinquefoil_T_2_5 | curvature_density | 0.18 | 65 | 9/9 | 0.115 | 0.104 | 0.356 |
| cinquefoil_T_2_5 | signed_turning | 0.1 | 9 | 9/9 | 0.463 | 0.390 | 0.344 |
| cinquefoil_T_2_5 | signed_turning | 0.1 | 17 | 9/9 | 0.400 | 0.359 | 0.316 |
| cinquefoil_T_2_5 | signed_turning | 0.1 | 33 | 9/9 | 0.242 | 0.206 | 0.335 |
| cinquefoil_T_2_5 | signed_turning | 0.1 | 65 | 9/9 | 0.122 | 0.108 | 0.337 |
| cinquefoil_T_2_5 | signed_turning | 0.18 | 9 | 9/9 | 0.463 | 0.585 | 0.281 |
| cinquefoil_T_2_5 | signed_turning | 0.18 | 17 | 9/9 | 0.359 | 0.386 | 0.297 |
| cinquefoil_T_2_5 | signed_turning | 0.18 | 33 | 9/9 | 0.206 | 0.198 | 0.299 |
| cinquefoil_T_2_5 | signed_turning | 0.18 | 65 | 9/9 | 0.107 | 0.106 | 0.310 |
| torus_T_3_4 | curvature_density | 0.1 | 9 | 9/9 | 0.512 | 0.439 | 0.312 |
| torus_T_3_4 | curvature_density | 0.1 | 17 | 9/9 | 0.359 | 0.359 | 0.316 |
| torus_T_3_4 | curvature_density | 0.1 | 33 | 9/9 | 0.264 | 0.213 | 0.333 |
| torus_T_3_4 | curvature_density | 0.1 | 65 | 9/9 | 0.132 | 0.114 | 0.329 |
| torus_T_3_4 | curvature_density | 0.18 | 9 | 9/9 | 0.561 | 0.585 | 0.328 |
| torus_T_3_4 | curvature_density | 0.18 | 17 | 9/9 | 0.379 | 0.386 | 0.340 |
| torus_T_3_4 | curvature_density | 0.18 | 33 | 9/9 | 0.217 | 0.213 | 0.331 |
| torus_T_3_4 | curvature_density | 0.18 | 65 | 9/9 | 0.112 | 0.116 | 0.327 |
| torus_T_3_4 | signed_turning | 0.1 | 9 | 9/9 | 0.512 | 0.439 | 0.688 |
| torus_T_3_4 | signed_turning | 0.1 | 17 | 9/9 | 0.393 | 0.372 | 0.664 |
| torus_T_3_4 | signed_turning | 0.1 | 33 | 9/9 | 0.248 | 0.209 | 0.670 |
| torus_T_3_4 | signed_turning | 0.1 | 65 | 9/9 | 0.133 | 0.115 | 0.656 |
| torus_T_3_4 | signed_turning | 0.18 | 9 | 9/9 | 0.488 | 0.634 | 0.672 |
| torus_T_3_4 | signed_turning | 0.18 | 17 | 9/9 | 0.359 | 0.400 | 0.680 |
| torus_T_3_4 | signed_turning | 0.18 | 33 | 9/9 | 0.209 | 0.217 | 0.677 |
| torus_T_3_4 | signed_turning | 0.18 | 65 | 9/9 | 0.110 | 0.116 | 0.676 |
| trefoil_T_2_3 | curvature_density | 0.1 | 9 | 9/9 | 0.585 | 0.512 | 0.250 |
| trefoil_T_2_3 | curvature_density | 0.1 | 17 | 9/9 | 0.407 | 0.428 | 0.230 |
| trefoil_T_2_3 | curvature_density | 0.1 | 33 | 9/9 | 0.255 | 0.237 | 0.239 |
| trefoil_T_2_3 | curvature_density | 0.1 | 65 | 9/9 | 0.128 | 0.120 | 0.249 |
| trefoil_T_2_3 | curvature_density | 0.18 | 9 | 9/9 | 0.585 | 0.537 | 0.219 |
| trefoil_T_2_3 | curvature_density | 0.18 | 17 | 9/9 | 0.386 | 0.372 | 0.238 |
| trefoil_T_2_3 | curvature_density | 0.18 | 33 | 9/9 | 0.202 | 0.207 | 0.238 |
| trefoil_T_2_3 | curvature_density | 0.18 | 65 | 9/9 | 0.110 | 0.109 | 0.243 |
| trefoil_T_2_3 | signed_turning | 0.1 | 9 | 9/9 | 0.488 | 0.561 | 0.703 |
| trefoil_T_2_3 | signed_turning | 0.1 | 17 | 9/9 | 0.407 | 0.441 | 0.758 |
| trefoil_T_2_3 | signed_turning | 0.1 | 33 | 9/9 | 0.224 | 0.244 | 0.755 |
| trefoil_T_2_3 | signed_turning | 0.1 | 65 | 9/9 | 0.122 | 0.120 | 0.746 |
| trefoil_T_2_3 | signed_turning | 0.18 | 9 | 9/9 | 0.537 | 0.561 | 0.875 |
| trefoil_T_2_3 | signed_turning | 0.18 | 17 | 9/9 | 0.352 | 0.386 | 0.758 |
| trefoil_T_2_3 | signed_turning | 0.18 | 33 | 9/9 | 0.193 | 0.207 | 0.758 |
| trefoil_T_2_3 | signed_turning | 0.18 | 65 | 9/9 | 0.104 | 0.111 | 0.762 |

All `24` representative Klein-flux certificates were nonzero
and conserved: `True`.

Across `108` fixed knot/channel/bandwidth/RG configurations,
the fitted odd-defect count exponent had mean `1.188`,
range `1.061` to `1.363`.
Curve-supported defects predict exponent one; area-filling defects predict
two. The fit supports, but does not certify, the regular-nodal-set picture.

The largest final diagonal displacement was
`0.762` for
`trefoil_T_2_3`,
`signed_turning`, bandwidth
`0.18`. The smallest was
`0.243` for
`trefoil_T_2_3`,
`curvature_density`.

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
