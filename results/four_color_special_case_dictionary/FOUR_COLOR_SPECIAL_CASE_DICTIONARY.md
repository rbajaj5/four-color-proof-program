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

## How To Use The Dictionary

1. Use bipartite/checkerboard structure before invoking Four Color.
2. Use the Heawood parity test for sphere triangulations.
3. Use `Z2 x Z2` differences when a conserved Tait/Klein-flow certificate is
   useful.
4. Use the weighted conflict audit whenever colors are interpreted as
   selectable routes, resources, or physical channels.
5. Keep curvature-generated maps labeled by projection, scale, kernel, and
   compactification.

## Claim Boundary

This is an organizational and diagnostic result. It neither strengthens the
Four Color Theorem nor proves a Goemans conjecture. The exact 58/60 row is an
abstract conflict-core control, not a certified directed-path realization.
