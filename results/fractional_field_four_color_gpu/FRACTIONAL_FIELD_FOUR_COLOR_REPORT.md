# Fractional-Field Four Color GPU Report

## Question

Can the regularity parameter of a fractional Gaussian field generate a
natural family of finite planar maps on which Four Color has a measurable,
nontrivial role?

This experiment uses the Cao-Sheffield convention
`FGF_s(R^2) = (-Delta)^(-s/2) W`, with `H = s - 1`. A finite Fourier cutoff
produces a continuous lattice surface. It is not an infinite coloring and it
is not a sample of a pointwise field when the continuum object exists only
as a generalized function.

## Curvature-to-Map Construction

For every lattice square, the sign of

`h00 + h11 - h10 - h01`

selects one of its two diagonals. The square window is compactified by adding
one exterior vertex connected to its boundary cycle. The result is a sphere
triangulation with the exact identity `E = 3V - 6`.

A sphere triangulation is vertex-3-colorable exactly when every vertex has
even degree. Because every face is a triangle, any non-Eulerian instance
needs at least four colors; the Four Color Theorem supplies the matching
upper bound. Thus each finite instance has exact chromatic number three or
four without treating a rendered color palette as evidence.

## Exact Fixtures

| Fixture | Odd vertices | Exact chromatic number | Explicit coloring proper |
| --- | ---: | ---: | --- |
| eulerian_checkerboard | 0 | 3 | True |
| single_curvature_flip | 4 | 4 | True |

## CUDA Audit

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Dtype: `torch.float32`
- Samples per H: `256`
- Periodic Fourier resolution: `64`
- H values: `[0.1, 0.25, 0.5, 0.75, 0.9]`
- Peak CUDA allocation: `55700992` bytes
- Total elapsed: `0.554` seconds

## Results

| H | Grid | Estimated H | Odd-degree fraction | Neighbor sign agreement | Four-chromatic rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 17 | 0.306 | 0.4131 | 0.3573 | 1.0000 |
| 0.1 | 33 | 0.306 | 0.4299 | 0.3661 | 1.0000 |
| 0.1 | 65 | 0.306 | 0.4647 | 0.4087 | 1.0000 |
| 0.25 | 17 | 0.390 | 0.4230 | 0.3688 | 1.0000 |
| 0.25 | 33 | 0.390 | 0.4386 | 0.3776 | 1.0000 |
| 0.25 | 65 | 0.390 | 0.4733 | 0.4226 | 1.0000 |
| 0.5 | 17 | 0.541 | 0.4437 | 0.3930 | 1.0000 |
| 0.5 | 33 | 0.541 | 0.4566 | 0.4016 | 1.0000 |
| 0.5 | 65 | 0.541 | 0.4867 | 0.4494 | 1.0000 |
| 0.75 | 17 | 0.698 | 0.4634 | 0.4266 | 1.0000 |
| 0.75 | 33 | 0.698 | 0.4742 | 0.4338 | 1.0000 |
| 0.75 | 65 | 0.698 | 0.4945 | 0.4808 | 1.0000 |
| 0.9 | 17 | 0.776 | 0.4775 | 0.4523 | 1.0000 |
| 0.9 | 33 | 0.776 | 0.4834 | 0.4585 | 1.0000 |
| 0.9 | 65 | 0.776 | 0.4959 | 0.5040 | 1.0000 |

At the finest grid, changing `H` from `0.1` to
`0.9` changed curvature-sign agreement from
`0.4087` to
`0.5040` and the
odd-degree fraction from
`0.4647` to
`0.4959`.

The largest mean disagreement between a coarse diagonal and the majority of
its nested fine diagonals was
`0.3303` for
`H=0.1` between grids
`17` and
`65`.

The new finite statistic is the **odd-degree or Four-color frustration
density**. It measures how often local curvature choices obstruct an
Eulerian/three-color triangulation. Four Color gives a uniform ceiling, while
`H` and scale control the density and persistence of the obstructions.

## Relation to Fractional Forms and Flux Tubes

Cao and Sheffield define fractional Gaussian differential forms, their
curl-free/divergence-free projections, lattice versions, and restrictions.
Their Chern-Simons discussion also recovers the Gauss linking integral from
the quadratic form `(J, curl^(-1) J)` for divergence-free currents.

This run uses only scalar cutoff 0-forms to generate planar triangulations.
A future magnetic branch can add a divergence-free fractional 1-form to the
analytic flux-tube field, then apply the same finite coloring construction to
transverse slices. That would connect random roughness, flux-tube winding,
and dynamic four-color transport without identifying colors with gauge
states.

## Claim Boundary

- `H` is a scaling/regularity parameter, not a chromatic or fractal dimension
  by itself.
- The continuum field is not literally an infinite Four Color map.
- Four Color applies to each finite compactified triangulation. It does not
  provide a canonical, measurable, or scale-consistent limiting coloring.
- Curvature-selected diagonals are one declared discretization, not an
  intrinsic decomposition forced by the continuum field.
- This is not a new Four Color theorem, gauge-theory theorem, or result about
  the Hausdorff dimension of fractional-field graphs.

## Performance

Mean per-H GPU group time was
`0.080`
seconds. Only finite summaries and one visualization fixture were
transferred to CPU.

## Source

Sky Cao and Scott Sheffield, *Fractional Gaussian Forms and Gauge Theory: An
Overview*: https://arxiv.org/abs/2406.19321
