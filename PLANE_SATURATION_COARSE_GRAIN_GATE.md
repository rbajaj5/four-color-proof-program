# Plane-Saturation Coarse-Graining Gate

Clifton and Salia's plane-saturation ratio measures how sparse a
plane-saturated partial embedding can be relative to its planar host:

```text
psr(G) = min e(H) / e(G).
```

Their twin-free `1/16` theorem does not improve the Four Color Theorem.
It adds a different structural diagnostic to this project:

> A coarse-grained planar graph should be audited for its retained-edge
> ratio, skeleton, isolates, and low-degree twin multiplicities before its
> coloring statistics are interpreted.

The exact checker:

- verifies the counts in Construction 1;
- checks the two compensating `r2/r3` inequalities used at the `1/16`
  boundary;
- computes degree-1 and degree-2 open-neighborhood twin profiles on finite
  fixtures; and
- blocks direct application to knot-projection multigraphs.

Reproduce with:

```bash
python scripts/check_plane_saturation_gate.py
python -m pytest tests/test_plane_saturation_gate.py -q
```

See `results/plane_saturation_gate/PLANE_SATURATION_GATE_REPORT.md`.

## Source

Alexander Clifton and Nika Salia, *Saturated Partial Embeddings of Planar
Graphs*, arXiv:2403.02458v1 (2024).
