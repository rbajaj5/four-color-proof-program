# Plane-Saturation Coarse-Graining Gate

## Verdict

The Clifton-Salia paper supplies a useful **embedding-efficiency diagnostic**
for this repository, not an improvement to the Four Color Theorem.

The exact Construction 1 count gate
**passes**. The two-case
`r2/r3` arithmetic gate behind the twin-free `1/16` lower bound
**passes** across all
**4224** nonzero integer pairs with coordinates at most
`64`.

This finite enumeration checks transcription and boundary arithmetic. The
paper's structural graph argument remains the proof.

## Published Results Recorded

For a plane-saturated subgraph `H` of a planar graph `G`, the paper defines

```text
psr(G) = min e(H) / e(G).
```

It proves:

- every planar graph with no degree-1 or degree-2 twins has
  `psr(G) > 1/16`;
- twin-free examples approach `1/16`;
- for the stated parameter regime,
  `psr(G) > 1 / (9 + k1 + 6 k2)`.

The extension to all nonnegative `(k1,k2)` and the minimum-degree-three
bound `psr(G) > 1/9` remain Conjectures 4.1 and 4.2.

## Construction 1 Audit

For `m >= 7`, the published construction has

```text
v(G) = 7m + 8       e(G) = 16m + 8
v(H) = 7m + 8       e(H) = m + 34.
```

The checker verifies exactly that

```text
(m + 34)/(16m + 8) < 1/16 + 3/m
```

for the recorded fixtures. The distance above `1/16` decreases to zero.

## Why The Lower Bound Needs Two Cases

One skeleton estimate leaves an `r2` term with ratio `1/17`. Grouping it
with `r3` repairs the estimate precisely when

```text
(r2 + r3)/(17r2 + 12r3) >= 1/16
iff 4r3 >= r2.
```

The alternate estimate repairs its deficient `r3` term when

```text
(r2 + r3)/(15r2 + 18r3) >= 1/16
iff r2 >= 2r3.
```

Every nonzero nonnegative pair satisfies at least one of these conditions.
The code checks both boundary equalities exactly with `Fraction` arithmetic.

## Upgrade To The Coarse-Graining Program

A planar coarse-graining record should now include:

1. host and retained edge counts;
2. the skeleton after deleting isolated vertices;
3. isolated-vertex count and face placement;
4. maximum degree-1 and degree-2 open-neighborhood twin multiplicities;
5. whether saturation was actually certified;
6. whether the object is a simple graph or a multigraph.

This catches a failure mode that color counts miss: repeated low-degree
neighborhoods can make the retained embedding ratio arbitrarily small.

## Knot-Projection Boundary

A simple four-regular planar projection graph automatically has no
degree-1 or degree-2 twins, so the paper's `1/16` hypothesis applies.
Generic knot projections, however, can be planar multigraphs with loops or
parallel edges. Passing to the underlying simple graph can change the host
subgraph relation and the saturation property. Therefore the theorem is
not applied to those diagrams until a preservation lemma or a multigraph
extension is supplied.

Plane saturation and Four Color also answer different questions:
saturation measures maximality of a partial embedding, while coloring
measures compatibility of region or vertex labels.

## Scope

- No Four Color bound is improved.
- Conjectures 4.1 and 4.2 remain conjectures.
- No knot equivalence or knot invariant follows.
- No simulation or GPU calculation was needed.
- The paper's full structural proof was not re-proved.

## Source

Alexander Clifton and Nika Salia, *Saturated Partial Embeddings of Planar
Graphs*, arXiv:2403.02458v1 (2024):
https://arxiv.org/abs/2403.02458
