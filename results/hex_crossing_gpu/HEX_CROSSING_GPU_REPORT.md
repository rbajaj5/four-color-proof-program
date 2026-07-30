# Hex Crossing CUDA Report

## Exact reliability census

Blue connects the two `q` sides of an `n x n` rhombic Hex board; yellow
connects the two `r` sides. Every coloring through side length
**5** was enumerated and tested directly.

| n | cells | states | blue wins | yellow wins | slope at 1/2 | p75-p25 | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 1 | 1 | 1.000000 | 0.500000 | 0.1098 |
| 2 | 4 | 16 | 8 | 8 | 1.500000 | 0.347296 | 0.0020 |
| 3 | 9 | 512 | 256 | 256 | 1.945312 | 0.271716 | 0.0031 |
| 4 | 16 | 65536 | 32768 | 32768 | 2.348633 | 0.226559 | 0.0065 |
| 5 | 25 | 33554432 | 16777216 | 16777216 | 2.725070 | 0.196000 | 0.7897 |

The coefficient file records the exact reliability polynomial

```text
P_n(p) = sum_k B[n,k] p^k (1-p)^(n^2-k),
```

where `B[n,k]` is the number of blue-winning boards with exactly `k` blue
cells. Color/axis duality gives `P_n(1/2)=1/2`. The derivative
`P_n'(1/2)` is the expected number of pivotal cells under unbiased
independent coloring by Russo's formula.

The log-log slope fit over the very small exact sizes `n=2,...,5`
is **0.651139**. This is a finite-size
diagnostic only, not an estimate of a critical exponent.

## Larger-board sampling at p=1/2

| n | samples | blue crossing estimate | 95% low | 95% high | seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 500000 | 0.499612 | 0.498226 | 0.500998 | 0.0679 |
| 7 | 500000 | 0.500240 | 0.498854 | 0.501626 | 0.0932 |

The full Monte Carlo file contains the crossing curve across the declared
probability grid. All random runs use seed `20260730`.

## Interpretation

This turns the qualitative Hex interface theorem into two quantitative
objects: an exact small-board reliability polynomial and a sampled
large-board transition curve. The slope and critical-window width measure
how sensitively global connectivity responds to local color perturbations.
That makes them suitable diagnostics for thresholded planar slices of
curvature, flux, or other fields.

## Boundaries

- The exactly-one-winner statement is the standard Hex theorem, not new.
- The exact census ends at `n=5`; larger rows are Monte Carlo.
- The finite-size exponent fit is not an asymptotic percolation claim.
- Crossing connectivity does not certify knot type, linking, helicity, or
  a three-dimensional Four Color analogue.
