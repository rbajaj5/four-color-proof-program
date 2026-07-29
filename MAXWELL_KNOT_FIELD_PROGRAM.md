# Maxwell Knot Field Program

## Research question

Can exact electromagnetic knot families supply geometrically meaningful
fixtures for the Four Color/knot proof program?

The answer is useful but limited:

- yes, they replace arbitrary 3D drawings with fields constrained by Maxwell's
  equations;
- yes, they expose projection pathologies that can be cataloged and tested;
- no, four-colorability by itself does not preserve weighted whole-trajectory
  choices; and
- no, sampling known torus-knot fields discovers neither a new knot type nor a
  new solution to Maxwell's equations.

## Static obstruction

In a simply connected, current-free, static region,

```text
curl B = 0
div B  = 0.
```

Consequently `B = grad Phi` with `Delta Phi = 0`. If a field line is
parametrized by `x'(s) = B(x(s))`, then

```text
d/ds Phi(x(s)) = |grad Phi(x(s))|^2.
```

Away from zeros of `B`, `Phi` increases strictly. Such a trajectory cannot be
periodic, so it cannot be a closed knot. This is consistent with McDonald's
analysis of a long permanent magnet: knotting the material can confine knotted
`B` lines inside it, but its exterior field is not freely tunable into an
arbitrary knot.

The experiment therefore uses a time-dependent null-field family rather than
a static dipole.

## Bateman/Kedia family

In units where `c = 1`, define

```text
D     = r^2 - (t-i)^2
alpha = (r^2 - t^2 - 1 + 2 i z) / D
beta  = 2(x - i y) / D
F     = E + iB = grad(alpha^p) x grad(beta^q).
```

For positive integers `p,q`, this is a null free-space Maxwell field:

```text
div E = div B = 0
partial_t B + curl E = 0
partial_t E - curl B = 0
E dot B = 0
|E|^2 - |B|^2 = 0.
```

At `t = 0`, the magnetic core is the extremal level set of
`Re(alpha^p beta^q)`. Let `g = gcd(p,q)`. Its two extremal sets contain
`2g` components in total. Each component is sampled once by

```text
arg(alpha) = (q/g) theta
arg(beta)  = -(p/g) theta + (c + 2 pi k)/q,
theta in [0, 2 pi],
k in {0, ..., g-1},
c in {0, -pi}.
```

The implementation evaluates the field, maps these `S^3` coordinates to
`R^3` by inverse stereographic projection, and verifies that each sampled
curve is tangent to `B`.

## Sanders-style finite diagnostics

Robertson, Sanders, Seymour, and Thomas separate two burdens:

1. every listed configuration is reducible; and
2. some listed configuration is unavoidable.

The Maxwell experiment borrows this separation, not their 633 graph
configurations. It records six local numerical failure modes:

1. a near-zero field;
2. an open numerical trajectory;
3. poor field/tangent alignment;
4. a near-tangent projected crossing;
5. ambiguous over/under depth; and
6. a nonplanar overlap graph among local diagnostic tasks.

The current fixtures pass this catalog. No completeness or unavoidability
claim is made for general Maxwell fields.

## Four Color boundary

When the local overlap graph is planar, its checks can be scheduled in at
most four conflict-free layers. This is a legitimate operational use of the
Four Color Theorem. It does not classify the knot, prove projection
equivalence, or solve a weighted path-selection problem.

The exact Goemans conflict-core fixture makes the last limitation concrete.
For a triangle of pairwise incompatible detours, the fractional point

```text
(z1,z2,z3) = (1/3, 2/5, 1/3)
```

satisfies all three edge inequalities and has direct-route cost `58`.
Every integral selection has at most one detour and therefore costs at least
`60`. The graph is planar and 3-colorable. The omitted odd-cycle inequality
`z1+z2+z3 <= 1` is violated by `1/15`.

This is an exact check of an abstract conflict core. It is not a verification
of the recently reported directed-path counterexample to the
cost-strengthened Goemans conjecture.

## Curvature RG extension

The sampled Bateman cores also seed an abstract coarse-graining experiment.
Write a centered periodic curve as

```text
gamma(s) = sum_k gamma_hat_k exp(i k s).
```

The heat semigroup evolves it by

```text
gamma_hat_k(tau) = exp(-tau k^2) gamma_hat_k(0).
```

For arclength `s`, its generator is the curvature vector
`partial_s^2 gamma`. This gives the "falling tower" analogy a precise rule:
blocks at frequency `k` are removed at rate `k^2`. The implementation retains
the periodic Bateman sampling parameter rather than continuously
reparametrizing by arclength, so it is a parameter-space heat semigroup, not
an intrinsic curve-shortening solver.

Let `m` be the first nonzero Fourier mode. After centering and rescaling by
`exp(m^2 tau)`, the flow converges in every fixed `C^r` norm to its `m`-th
harmonic. If `m=1` and its coefficient vectors span a plane, the limit is an
ellipse. If `m>1`, the limit is an `m`-fold covered ellipse and is not
embedded. In the `m=1`, rank-two case, a nontrivial initial knot cannot remain
embedded for all finite RG time: eventual `C^1` closeness to the ellipse
would force it into the unknot isotopy class. For `m>1`, convergence to a
nonembedded limit alone does not prove finite-time contact.

This last statement is an elementary Fourier consequence, not a claimed new
theorem. The finite projection crossings in the CUDA run provide only
one-sided fall certificates: a generic diagram with fewer crossings than the
known minimal crossing number cannot represent the original knot. See
`results/curvature_rg_knot_collapse/CURVATURE_RG_KNOT_COLLAPSE_REPORT.md`.

## Candidate directions

### MK-C1. Pathology-complete isotopy monitor

For a compact, physically constrained family of fields, identify sufficient
quantitative margins such that the knot diagram cannot change while all
margins remain positive. Zeros, projection tangencies, and crossing-depth
collisions are the natural boundary events.

Status: `open_program`; closely related to standard isotopy-stability
principles, with no novelty claim yet.

### MK-C2. Reducible local projection events

Find local event templates whose resolution provably preserves the ambient
isotopy class and decreases a well-founded geometric complexity.

Status: `open_program`; a finite catalog without an unavoidability theorem
would not suffice.

### MK-C3. Whole-field-line selection

Choose complete field lines under spatial capacity or inspection constraints
without splitting a line between choices.

Status: `four_color_bridge_falsified_in_general`. The exact `58/60` triangle
fixture shows that planar colorability does not imply cost-preserving
integral selection. Any positive theorem needs stronger polyhedral structure.

## Run

The committed run used CUDA:

```text
python scripts/check_maxwell_knot_fields_gpu.py
```

The script fails if CUDA is unavailable. A deliberate CPU calibration can be
requested with:

```text
python scripts/check_maxwell_knot_fields_gpu.py --allow-cpu
```

Outputs are written to `results/maxwell_knot_fields/`.

The curvature RG extension is run separately:

```text
python scripts/check_curvature_rg_knot_collapse_gpu.py
```

Its outputs are written to `results/curvature_rg_knot_collapse/`.

The declared macro/meso/micro scale decomposition is generated by:

```text
python scripts/check_curvature_rg_multiscale_gpu.py
```

The bands are diagnostics rather than universal physical divisions. The
output records both absolute bending-energy depletion and each band's
relative share after RG rescaling.

## Sources

- K. T. McDonald, *Can the Field Lines of a Permanent Magnet Be Tied in
  Knots?*: https://kirkmcd.princeton.edu/examples/knot.pdf
- H. Kedia et al., *Tying knots in light fields*:
  https://arxiv.org/abs/1302.0342
- H. Kedia, D. Peralta-Salas, and W. T. M. Irvine, *When do knots in light
  stay knotted?*: https://arxiv.org/abs/1706.06175
- N. Robertson, D. Sanders, P. Seymour, and R. Thomas:
  https://thomas.math.gatech.edu/FC/fourcolor.html
- Y. Dinitz, N. Garg, and M. X. Goemans:
  https://doi.org/10.1007/s004930050043
- Current planar SSUF formulation:
  https://doi.org/10.1007/s10107-026-02365-x
