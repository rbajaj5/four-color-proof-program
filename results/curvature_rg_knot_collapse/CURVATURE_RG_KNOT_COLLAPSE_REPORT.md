# Curvature RG Knot Collapse Report

## Model

The knot is treated as a periodic stack of Fourier blocks. Coarse-graining is
the heat semigroup

```text
partial_tau gamma = partial_s^2 gamma,
gamma_hat_k(tau) = exp(-tau k^2) gamma_hat_k(0).
```

For arclength parameter `s`, `partial_s^2 gamma` is the curvature vector.
The implementation retains the periodic parameter inherited from the Bateman
core sampler, so it is exactly a parameter-space Fourier heat flow rather
than an intrinsic curve-shortening solver. Its spectral bending energy weights
mode `k` by `k^4`, making the removal of fine, curvature-producing modes
explicit. Curves are centered and rescaled only for visualization and
scale-free diagnostics.

This is an abstract geometry flow, not a mechanical model of elastic string
or a literal Jenga tower.

## Exact Collapse Proposition

Let `m` be the first nonzero Fourier mode of a nonconstant smooth closed
curve. After subtracting the centroid and multiplying by `exp(m^2 tau)`, the
heat flow converges in every fixed `C^r` norm to its `m`-th harmonic.

The proof is termwise: every mode `k > m` is suppressed relative to mode `m`
by `exp(-(k^2-m^2)tau)`. When `m=1` and the leading sine/cosine vectors have
rank two, the limit is a planar ellipse. A nontrivial initial knot in this
case cannot remain embedded for every finite RG time: eventual `C^1`
closeness to the ellipse would place it in the ellipse's tubular neighborhood
and hence in the unknot isotopy class.

When `m>1`, the limit is an `m`-fold cover of an ellipse and is not embedded.
Convergence to that limiting multiple cover alone does **not** prove that
self-contact occurs at a finite RG time. The finite-time fall intervals below
are numerical projection diagnostics, not consequences of the exact
proposition. These Fourier statements are elementary and are not claimed as
new literature theorems.

## CUDA Run

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Samples per curve: `512`
- RG scales per curve: `48`
- Peak CUDA allocation: `42286080` bytes
- Elapsed: `57.099` seconds

## Results

| Fixture | First mode | Leading rank | Minimal crossing baseline | First below-baseline interval | Minimum clearance |
| --- | ---: | ---: | ---: | --- | ---: |
| hopf_core_unknot | 1 | 2 | 0 | not_applicable | 2.34 |
| trefoil_2_3 | 1 | 2 | 3 | (0.475833, 0.602014] | 0.272 |
| cinquefoil_2_5 | 2 | 2 | 5 | (1.95148, 2.46897] | 0 |
| torus_knot_3_4 | 1 | 2 | 8 | (0.116023, 0.14679] | 0.822 |

The fixed generic projection supplies a one-sided fall certificate: once its
crossing count is below the known minimal crossing number of the initial
torus knot, the curve cannot still be a generic diagram of that knot. A
crossing-count change by itself is not used as a complete knot classifier.
The nonlocal-clearance trace shows where the sampled tower approaches
self-contact.

The cinquefoil has first surviving mode `2`, so its terminal object is a
double-covered ellipse rather than an embedded unknot. This is why its
clearance becomes extremely small before the projection finally degenerates.

## Interpretation

The useful RG state variables are:

- effective Fourier-block count;
- spectral bending energy;
- geometric curvature;
- nonlocal clearance;
- projection transversality; and
- the minimal-crossing lower bound of the starting knot.

This upgrades the earlier visual analogy into a controlled statement:
curvature determines which scales disappear, while loss of embedding marks
the fall. It does not yet supply a topology-preserving simplifier. Adding
self-repulsion or a hard tube-thickness constraint is the next mathematically
meaningful branch.

## Claim Boundary

- New knot theorem: `false`
- New Maxwell solution: `false`
- Exact abstract collapse proposition recorded: `true`
- Numerical topology classification claimed: `false`
