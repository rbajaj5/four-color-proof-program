# Fixed-Embedding Saturation Checker

## Result

All **3** exact fixtures
**pass**.

For a connected spanning simple subgraph `H` with a declared sphere rotation
system, every face boundary is enumerated exactly as a dart orbit. A missing
host edge can be inserted without crossing precisely when its endpoints are
cofacial. The checker therefore certifies whether `H` is edge-saturated for
that fixed labeled embedding.

## Separator Fixture

The nontrivial fixture places one leaf inside a separating triangle and a
second leaf outside it. The abstract host adds an edge between the leaves.
That edge is not cofacial in the declared embedding, so it is blocked. If the
host instead adds an edge from the interior triangle vertex to the exterior
leaf, the checker finds the endpoints on a common face and rejects
saturation.

This makes the embedding dependence concrete: abstract planarity of the host
does not mean every host edge can be inserted into an arbitrary partial
embedding.

## Strict Scope Boundary

This checker does **not** implement the full Clifton-Salia definition.
Their vertices are unlabeled under the subgraph embedding, and an added edge
may introduce new vertices. Their proof also handles disconnected skeletons
and the placement of isolated vertices among faces.

Accordingly, every output keeps
`clifton_salia_plane_saturation_certified = False`. The current certificate
is useful as a necessary implementation layer and debugging oracle, not as a
replacement for the paper's saturation test.

## Next Exact Step

The next extension should enumerate label-preserving graph isomorphisms for
small hosts and explicitly encode component nesting and isolated-vertex face
assignments. Only then can finite fixtures approach the paper's unlabeled
notion. Multigraph knot projections remain a separate extension.

## Scope

- No Four Color result is inferred.
- No knot invariant or knot-equivalence claim is made.
- No simulation or GPU computation was used.
