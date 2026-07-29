# Cubic Edge-Smoothing Lift Report

## Question

For an edge of a loopless cubic graph, delete its endpoints and reconnect the
four dangling halfedges in the two possible cross pairings. Does a Tait
3-edge-coloring of a smoothed child imply a coloring of the parent?

## Exact result

- Parent fixtures: **8**
- Parent edges checked: **108**
- Smoothing branches recorded: **216**
- Exact coloring-partition identity passed on every edge: **True**
- Colorable children with no liftable coloring: **47**
- Such children from planar parents: **17**

For every tested edge, the parent's labeled Tait colorings partition exactly
between the two smoothing pairings:

```text
Col_3(G) = Lift_0(G,e) disjoint-union Lift_1(G,e).
```

This identity follows directly from the two noncentral colors at the endpoints
of a colored edge: exactly one pairing joins equal colors. The program checks
the stronger labeled-set equality, not just equality of cardinalities.

## Fixture summary

| Fixture | Planar parent | Parent colorings | Colorable children | Liftable children | Colorable but nonliftable |
| --- | --- | ---: | ---: | ---: | ---: |
| k4_tetrahedral | True | 6 | 6 | 6 | 0 |
| triangular_prism | True | 6 | 12 | 9 | 3 |
| cube_graph | True | 24 | 24 | 24 | 0 |
| k33_nonplanar | False | 12 | 18 | 18 | 0 |
| petersen_snark | False | 0 | 30 | 0 | 30 |
| frucht_graph | True | 6 | 26 | 18 | 8 |
| truncated_tetrahedron | True | 6 | 24 | 18 | 6 |
| heawood_graph | False | 48 | 42 | 42 | 0 |

## Obstruction to a naive reduction

- `triangular_prism`, edge `(0, 3)`, pairing 1: 6 child colorings, 0 lifts
- `triangular_prism`, edge `(1, 4)`, pairing 1: 6 child colorings, 0 lifts
- `triangular_prism`, edge `(2, 5)`, pairing 1: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 1)`, pairing 0: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 1)`, pairing 1: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 4)`, pairing 0: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 4)`, pairing 1: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 5)`, pairing 0: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(0, 5)`, pairing 1: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(1, 2)`, pairing 0: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(1, 2)`, pairing 1: 6 child colorings, 0 lifts
- `petersen_snark`, edge `(1, 6)`, pairing 0: 6 child colorings, 0 lifts

A child being 3-edge-colorable is therefore not, by itself, a
positivity-preserving reduction certificate. A valid FC-C1 move needs either a
liftable child coloring or a structural condition forcing positive lift rate.

The exact boundary criterion is especially simple. A proper child coloring
lifts if and only if the two replacement edges are bichromatic. Consequently,

```text
|Col_3(G)| = Bichrom_0(G,e) + Bichrom_1(G,e),
```

while the monochromatic sectors contribute no parent colorings. This
two-terminal signature is the useful quantity for subsequent reductions.

There is also a proved local sufficient condition: if the two replacement
edges meet at a child vertex, properness forces distinct colors, and every
child coloring lifts. All
**33** branches
with adjacent replacement edges passed this implication. What remains missing
is an
unavoidability argument that supplies a complexity-reducing, colorable branch
with a positive bichromatic sector in every bridgeless cubic plane graph.

## Scope

This is an exact finite experiment and an elementary local identity. It is not
a new proof of the Four Color Theorem and is not presented as a novel theorem.
The nonplanar controls diagnose the lifting logic; they are not evidence about
plane unavoidability.

## Decision

Retain the exact partition identity as infrastructure. Search next for
checkable local conditions that force a nonzero liftable sector, while testing
every proposed condition against snarks and nonplanar controls before any
planar proof claim.
