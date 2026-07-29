# Four Color Knot-Theoretic Proof Program

## Objective

The goal is not to announce another informal proof of the Four Color Theorem.
It is to search for a proof with a better audit profile:

- fewer opaque configurations;
- a conceptual local invariant;
- exact symbolic certificates;
- a clean distinction between equivalence, finite evidence, and proof; and
- a route to proof-assistant formalization.

The classical computer-assisted proof remains the baseline. Robertson,
Sanders, Seymour, and Thomas reduce the theorem to reducibility and
unavoidability for 633 configurations, with 32 discharging rules:

- https://thomas.math.gatech.edu/FC/fourcolor.html
- https://thomas.math.gatech.edu/FC/ftpinfo.html

## Knot-theoretic reformulation

Tait's classical equivalence says that the Four Color Theorem holds if and
only if every bridgeless cubic plane graph has a proper edge 3-coloring.

Kauffman, Silver, and Williams sharpen a link-diagram version in *The
Penrose-Kauffman Polynomial*:

- https://arxiv.org/abs/2604.16635

Their Theorem 6.6 states that the Four Color Theorem is equivalent to:

```text
Every reduced, prime alternating link diagram in the plane
without any bigon regions has a 3-colorable state.
```

A state is obtained by smoothing a subset of crossings. Their Theorem 5.3
identifies colorable states with strict local maxima of the number of state
components on the Boolean smoothing cube.

This turns the proof search into a discrete-Morse-style question:

```text
Can one force a strict local maximum whose component graph is 3-colorable?
```

Existence of a strict local maximum alone is trivial on a finite cube.
The 3-colorability clause is the entire unresolved burden. Omitting it would
be a circular "proof."

## Exact Penrose/Tait layer

For a cubic rotation system, assign the Levi-Civita tensor to each vertex and
contract over three edge colors. With the standard plane phase convention,
the Penrose contraction equals the number of proper Tait edge 3-colorings.

The implementation in `src/four_color_penrose.py`:

1. enumerates proper edge 3-colorings by exact backtracking;
2. computes the signed Levi-Civita contraction;
3. checks the plane Penrose identity on small fixtures; and
4. keeps nonplanar and bridge controls separate.

This verifies the algebraic interface. It does not prove positivity for all
bridgeless cubic plane graphs.

## Candidate proof directions

### FC-C1. Positivity-preserving smoothing reduction

Find finitely many local moves on reduced prime alternating bigon-free
diagrams such that:

1. every non-base diagram admits a move;
2. the move lowers a well-founded complexity;
3. existence of a 3-colorable state for the smaller diagram lifts to the
   original diagram; and
4. every terminal diagram has an explicit 3-colorable state.

These four clauses would give a conceptual inductive proof. Clause 1 is an
unavoidability theorem in new clothing and must not be assumed.

The edge-smoothing experiment adds a necessary warning. For each edge, the
parent Tait colorings partition between the two local pairings, but a smoothed
child may be colorable while none of its colorings lift to the parent. Thus
"the smaller graph is colorable" is too weak for clause 3. The reduction must
control the liftable coloring sector explicitly.

The replacement-edge boundary signature gives one small positive lemma: if
the replacement edges are adjacent in the child, every proper child coloring
lifts because adjacent edges receive distinct colors. This is a valid local
lifting condition, but no unavoidability or global termination statement is
claimed.

### FC-C2. Discrete Morse certificate on the smoothing cube

Construct a matching or flow on non-3-colorable states so that at least one
unmatched strict local maximum has a 3-colorable component graph.

The component-count function supplies a natural height, but component count
alone cannot distinguish a 3-colorable component graph from one requiring
four colors.

### FC-C3. Positive state-sum decomposition

Rewrite `P(3)`, the Penrose-Kauffman evaluation, as a sum of manifestly
nonnegative local contributions and prove at least one contribution is
strictly positive for every bridgeless cubic plane graph.

The existing state sum counts colorings exactly, but merely rewriting the
count does not establish nonvanishing. The strict-positivity mechanism is the
missing lemma.

## What would count as progress

- A new local reduction proved to preserve 3-colorable states.
- A smaller unavoidable family with independently checkable certificates.
- A structural subclass theorem beyond already-known Hamiltonian,
  low-treewidth, or reducible cases.
- A formal proof of one nontrivial reduction lemma.
- A counterexample that eliminates an attractive but false monotonicity
  principle.

More successful finite enumerations alone do not count as a better proof.

## Run

```text
python scripts/check_four_color_penrose_fixtures.py
python scripts/check_cubic_edge_smoothing_lifts.py
python -m pytest tests -q
```

Outputs are written to `results/four_color_penrose_program/` and
`results/cubic_edge_smoothing/`.

## Status

- Four Color Theorem: `standard_theorem`.
- Tait and Penrose identities: `standard_theorems`.
- Kauffman-Silver-Williams link reformulation: `published_2026_equivalence`.
- FC-C1 through FC-C3: `open_proof_program`.
- New proof obtained: `false`.
