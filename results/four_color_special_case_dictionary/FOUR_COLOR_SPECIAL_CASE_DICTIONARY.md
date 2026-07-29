# Four Color Special-Case Dictionary

## Purpose

This registry separates mechanisms that are often collapsed into the phrase
"Four Color." It asks which smaller theorem or exact certificate actually
does the work, whether a `Z2` curvature or Klein-four flow interpretation is
available, and whether a weighted optimization claim needs an additional
integrality audit.

| Case | Color bound | Diagnosis | Weighted-lift status |
| --- | ---: | --- | --- |
| bipartite_planar_graph | 2 | bipartiteness | not_tested |
| triangle_free_planar_graph | 3 | Groetzsch theorem gives an upper bound | not_tested |
| generic_knot_diagram_regions | 2 | checkerboard face coloring | not_applicable |
| eulerian_sphere_triangulation | 3 | Heawood parity criterion | not_tested |
| non_eulerian_sphere_triangulation | 4 | parity obstruction plus Four Color | not_tested |
| dual_cubic_tait_instance | 4 | Tait/Klein-four reformulation | not_tested |
| curvature_generated_triangulation | 4 | local XOR defects and degree parity | not_tested |
| weighted_planar_conflict_triangle | 3 | planar coloring plus relaxation audit | fails_edge_relaxation_lift |
| arbitrary_planar_graph | 4 | Four Color theorem | not_tested |

## Goemans-Style Weighted Control

The planar conflict triangle is 3-colorable, and its fractional point
satisfies all pairwise edge inequalities. Nevertheless:

- fractional direct cost: `58`;
- minimum integral direct cost: `60`;
- exact gap: `2`;
- omitted triangle-inequality slack: `-1/15`.

Thus a planar coloring can schedule conflict-free layers while failing to
lift a fractional weighted choice to a whole-route integral choice. The
diagnosis is: `add the triangle odd-cycle inequality; do not infer weighted whole-route integrality from planar colorability`.

## Fixed-Palette Algorithmic Layer

Zamir's 2026 theorem proves that, for every fixed palette size `K`,
fixed-palette list coloring has a randomized
`O*((2-epsilon_K)^n)` algorithm with exponentially small one-sided error.
An explicit coloring can be recovered from a decision algorithm with only
polynomial overhead. The result is useful here as a generic fallback, not as
the primary solver for structured planar cases.

Source: [k-Coloring is Faster than Computing the Chromatic Number](https://arxiv.org/abs/2607.25973), Or Zamir,
arXiv:2607.25973.

| Case | Target k | Formulation | Preferred route | Zamir role | Weighted objective in scope |
| --- | ---: | --- | --- | --- | --- |
| bipartite_planar_graph | 2 | vertex coloring | bipartite BFS/DFS | polynomial base case | False |
| triangle_free_planar_graph | 3 | vertex coloring | bipartite test; otherwise Groetzsch certifies exact chi=3 | theoretical fixed-palette fallback | False |
| generic_knot_diagram_regions | 2 | vertex coloring of the region-dual graph | checkerboard face coloring | polynomial base case | False |
| eulerian_sphere_triangulation | 3 | vertex coloring | degree parity plus explicit 3-coloring | theoretical fixed-palette fallback | False |
| non_eulerian_sphere_triangulation | 3 | vertex coloring decision returning NO | odd-degree obstruction plus explicit 4-coloring | theoretical fixed-palette fallback | False |
| dual_cubic_tait_instance | 3 | vertex coloring of the line graph | nonzero conserved Klein-four dual flow | theoretical fixed-palette fallback | False |
| curvature_generated_triangulation | 3 | vertex coloring decision | local XOR parity plus explicit 3/4-coloring | theoretical fixed-palette fallback | False |
| weighted_planar_conflict_triangle | 3 | vertex coloring of the conflict graph | 3-color conflict graph, then run a separate weighted-integrality audit | theoretical fixed-palette fallback | False |
| arbitrary_planar_graph | 3 | vertex coloring decision | bipartite test, then 3-color decision, then Four Color fallback | theoretical fixed-palette fallback | False |

The implementation does **not** reproduce Zamir's research algorithm. It
records theorem applicability and keeps the existing exact structural
solvers in front. For `k=3`, specialized algorithms also have much stronger
explicit constants than the paper's general bootstrap. The paper's own
asymptotic quantitative estimate gives a reciprocal saving with tower height
`Theta(K)`, so this result should not be advertised as a practical solver
for these small fixtures.

## Planar Chromatic-Number Decision Hierarchy

| Stage | Certified condition | Exact outcome | Route |
| ---: | --- | --- | --- |
| 1 | no edges | chi=1 | direct inspection |
| 2 | has edges and is bipartite | chi=2 | BFS/DFS odd-cycle test |
| 3 | triangle-free and non-bipartite | chi=3 | Groetzsch upper bound plus odd-cycle lower bound |
| 4 | sphere triangulation | chi=3 iff all degrees are even; otherwise chi=4 | Heawood parity criterion plus Four Color |
| 5 | remaining planar graph | chi=3 if 3-colorable; otherwise chi=4 | fixed 3-color decision plus Four Color; Zamir gives a generic randomized sub-2^n route |

This hierarchy makes the headline distinction operational: one should not
compute the unrestricted chromatic number when a fixed `3`-color decision
plus the Four Color upper bound resolves the planar case.

## How To Use The Dictionary

1. Use bipartite/checkerboard structure before invoking Four Color.
2. Use the Heawood parity test for sphere triangulations.
3. Use `Z2 x Z2` differences when a conserved Tait/Klein-flow certificate is
   useful.
4. Use the weighted conflict audit whenever colors are interpreted as
   selectable routes, resources, or physical channels.
5. Keep curvature-generated maps labeled by projection, scale, kernel, and
   compactification.
6. Use fixed-palette search only after structural certificates fail.

## Claim Boundary

This is an organizational and diagnostic result. It neither strengthens the
Four Color Theorem nor proves a Goemans conjecture. The exact 58/60 row is an
abstract conflict-core control, not a certified directed-path realization.
Zamir's theorem concerns colorability, not weighted routing, and its full
algorithm has not been implemented in this repository.
