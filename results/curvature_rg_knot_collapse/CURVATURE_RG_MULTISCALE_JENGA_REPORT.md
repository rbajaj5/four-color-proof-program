# Curvature RG Multiscale Jenga Report

## Discretization

The 512-vertex curve is resolved into Fourier blocks and then grouped by
wavelength relative to one traversal of the closed curve:

- **macroscopic:** modes 1-2, controlling the global loop and terminal
  harmonic;
- **mesoscopic:** modes 3-8, controlling lobes, crossing corridors, and the
  recognizable knot diagram; and
- **microscopic:** modes 9 and above, controlling fine bending and local
  geometric roughness.

The boundaries are declared diagnostics, not universal physical constants.
For each band this run records spectral power, Dirichlet energy, and bending
energy. A band is called depleted when it retains at most
`1%` of its initial bending energy.

## CUDA Audit

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA used: `True`
- Curves: `4`
- Samples per curve: `512`
- RG scales: `48`
- Peak CUDA allocation: `108032` bytes
- Elapsed: `1.240` seconds

## Scale Cascade

| Fixture | Scale | Modes | Initial bending share | 99% absolute-depletion interval | Absolute retention at projection fall | Relative share at projection fall | Depleted before fall |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| hopf_core_unknot | macroscopic | 1-2 | 0.4544 | (1.54245, 1.95148] | not_applicable | not_applicable | not_applicable |
| hopf_core_unknot | mesoscopic | 3-8 | 0.5448 | (0.185716, 0.234963] | not_applicable | not_applicable | not_applicable |
| hopf_core_unknot | microscopic | 9+ | 0.0008 | (0.0223607, 0.0282902] | not_applicable | not_applicable | not_applicable |
| trefoil_2_3 | macroscopic | 1-2 | 0.0500 | (0.602014, 0.761654] | 0.0104 | 0.9976 | False |
| trefoil_2_3 | mesoscopic | 3-8 | 0.6721 | (0.116023, 0.14679] | 1.866e-06 | 0.002409 | True |
| trefoil_2_3 | microscopic | 9+ | 0.2780 | (0.0223607, 0.0282902] | 6.682e-26 | 3.566e-23 | True |
| cinquefoil_2_5 | macroscopic | 1-2 | 0.0197 | (0.475833, 0.602014] | 2.642e-09 | 1 | True |
| cinquefoil_2_5 | mesoscopic | 3-8 | 0.4214 | (0.0724841, 0.0917052] | 9.945e-22 | 8.047e-12 | True |
| cinquefoil_2_5 | microscopic | 9+ | 0.5589 | (0.0176739, 0.0223607] | 4.38e-34 | 4.701e-24 | True |
| torus_knot_3_4 | macroscopic | 1-2 | 0.0001 | (1.95148, 2.46897] | 0.7456 | 0.01588 | False |
| torus_knot_3_4 | mesoscopic | 3-8 | 0.4821 | (0.116023, 0.14679] | 0.00979 | 0.9841 | True |
| torus_knot_3_4 | microscopic | 9+ | 0.5178 | (0.0139695, 0.0176739] | 1.176e-12 | 1.27e-10 | True |

`not_applicable` means the unknot fixture has no nonzero minimal-crossing
baseline. `not_seen` means the band did not cross the depletion threshold in
the sampled RG interval. Absolute retention follows the unrescaled heat flow;
relative share records which band controls the rescaled shape.

## Macroscopic Behavior

The low modes retain the gross occupied region while all shorter wavelengths
are suppressed. Eventually the first nonzero harmonic dominates: a rank-two
first mode gives an ellipse, while a higher first mode gives a multiply
covered ellipse. This level answers where the entire tower leans and what
shape survives after the fall. Its absolute energy still decays, but its
relative share approaches one.

## Mesoscopic Behavior

The middle modes encode the visible lobe and crossing architecture. Their
depletion is therefore compared directly with the first fixed-projection
crossing count below the knot's known minimum. This level is the load-bearing
arrangement of the tower: topology can persist after microscopic roughness is
gone, but it cannot be read from the macroscopic outline alone. Near a
multiple-cover limit, a band with tiny relative energy may still act as a
topologically decisive shim that keeps nearly coincident strands apart.

## Microscopic Behavior

High modes carry disproportionately large bending energy because mode `k`
is weighted by `k^4`. The heat multiplier removes them at rate `k^2`, so this
band is depleted first. In the analogy, individual high-curvature blocks fall
without necessarily changing the tower's global knot type.

## Interpretation Boundary

This is a spectral coarse-graining hierarchy. It does not model gravity,
friction, block contacts, or intrinsic elastic-rod dynamics. The scale bands
are useful observables, while the fixed-projection fall remains a one-sided
diagnostic rather than a complete knot classifier.
