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

At every interior grid vertex, the parity of its degree is the XOR of the
four surrounding diagonal choices. Encoding a choice as a spin
`sigma in {-1,+1}` gives the exact local identity

`1[odd degree] = (1 - sigma_NW sigma_NE sigma_SW sigma_SE) / 2`.

This identifies Four-color frustration with a finite `Z2` curvature defect.

## Spectral Sign-Agreement Formula

The mixed differences are centered jointly Gaussian. If two neighboring
mixed differences have correlation `rho`, the Gaussian arcsine identity
gives

`P(equal signs) = 1/2 + asin(rho) / pi`.

For the finite Fourier cutoff, `rho` is computed directly by summing the
filtered spectral density

`|k|^(-2(H+1)) |1-exp(i k_x a)|^2 |1-exp(i k_y a)|^2`

against the one-cell translation phase. The report therefore compares an
analytic finite-spectral prediction with CUDA observations; it does not fit
the prediction to the samples.

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
- Samples per H: `16`
- Periodic Fourier resolution: `32`
- H values: `[0.25, 0.75]`
- Peak CUDA allocation: `891904` bytes
- Total elapsed: `0.524` seconds

## Results

| H | Grid | Estimated H | Odd fraction | Agreement | Spectral prediction | Residual | Four-chromatic rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 9 | 0.352 | 0.4131 | 0.3527 | 0.3684 | -0.0157 | 1.0000 |
| 0.25 | 17 | 0.352 | 0.4185 | 0.3734 | 0.3771 | -0.0037 | 1.0000 |
| 0.25 | 33 | 0.352 | 0.4665 | 0.4261 | 0.4230 | +0.0031 | 1.0000 |
| 0.75 | 9 | 0.630 | 0.4360 | 0.4129 | 0.4201 | -0.0071 | 1.0000 |
| 0.75 | 17 | 0.630 | 0.4728 | 0.4322 | 0.4322 | -0.0001 | 1.0000 |
| 0.75 | 33 | 0.630 | 0.4978 | 0.4843 | 0.4812 | +0.0031 | 1.0000 |

At the finest grid, changing `H` from `0.25` to
`0.75` changed curvature-sign agreement from
`0.4261` to
`0.4843` and the
odd-degree fraction from
`0.4665` to
`0.4978`.

The largest mean disagreement between a coarse diagonal and the majority of
its nested fine diagonals was
`0.3008` for
`H=0.25` between grids
`9` and
`33`.

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
`0.207`
seconds. Only finite summaries and one visualization fixture were
transferred to CPU.

## Source

Sky Cao and Scott Sheffield, *Fractional Gaussian Forms and Gauge Theory: An
Overview*: https://arxiv.org/abs/2406.19321
