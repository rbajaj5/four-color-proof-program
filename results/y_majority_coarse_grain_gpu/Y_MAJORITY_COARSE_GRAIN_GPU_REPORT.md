# Y-Majority Coarse-Graining CUDA Report

## Exact finite audit

The triangular board is

```text
{(q,r): q>=0, r>=0, q+r<n}.
```

One reduction step replaces the triangle

```text
(q+1,r), (q,r), (q,r+1)
```

by its majority color. The GPU exhausts every coloring through side length
**6** and compares the recursively reduced one-cell color with a
direct three-boundary connectivity test.

| n | cells | states | blue Y | yellow Y | both | neither | mismatch | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 1 | 1 | 0 | 0 | 0 | 0.1173 |
| 2 | 3 | 8 | 4 | 4 | 0 | 0 | 0 | 0.0096 |
| 3 | 6 | 64 | 32 | 32 | 0 | 0 | 0 | 0.0083 |
| 4 | 10 | 1024 | 512 | 512 | 0 | 0 | 0 | 0.0145 |
| 5 | 15 | 32768 | 16384 | 16384 | 0 | 0 | 0 | 0.0288 |
| 6 | 21 | 2097152 | 1048576 | 1048576 | 0 | 0 | 0 | 0.3682 |

All tested boards have exactly one monochromatic Y, and majority reduction
preserves its color.

## Coarse-graining interpretation

This supplies an exact planar renormalization primitive. Threshold a scalar
field on a triangular hexagonal slice into high/low cells. With the declared
three-side boundary geometry, recursive majority reduction preserves which
phase has a three-arm connection. For a two-terminal Hex geometry, the same
interface argument certifies a high-phase barrier or a low-phase dual
corridor.

That is useful for curvature or flux-tube slices: it preserves a declared
connectivity event while reducing resolution. It is substantially stronger
than replacing blocks by their average and hoping topology survives.

## Boundaries

- This is the Karlin-Peres Hex/Y theorem made executable, not a new theorem.
- The certificate is planar. In three dimensions, both phases may percolate,
  and there is no analogous exclusivity without additional hypotheses.
- It preserves the Y-connectivity predicate, not knot type, linking number,
  magnetic helicity, or the Four Color Theorem.
- A physical application must preregister the slice, threshold, boundary
  arcs, and refinement rule.

## Source

Anna R. Karlin and Yuval Peres, *Game Theory, Alive*, Sections 1.2.1-1.2.3,
American Mathematical Society. The supplied excerpt contains the oriented
Hex-interface proof and the majority-triangle reduction from Y boards of
side `n` to side `n-1`.
