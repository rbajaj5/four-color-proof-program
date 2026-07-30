# Jaffe-Liu Diagram Analysis

This note analyzes operations depicted in Arthur Jaffe and Zhengwei Liu,
*A Mathematical Picture Language Program*, rather than reproducing its
figures.

## Page 6: rotation and two products

A square picture admits vertical and horizontal gluing. A 90-degree string
Fourier rotation interchanges the two products, modeled algebraically as
multiplication and convolution. On the Boolean smoothing cube
`Z_2^E`, the repository uses the Walsh-Hadamard transform as the corresponding
finite Fourier model:

```text
W(f *_xor g) = W(f) W(g).
```

This identity is checked exactly with integer arithmetic.

## Page 7: reflection positivity

The diagram maps a reflected-gluing positivity statement to ordinary
`C*`-positivity after rotation. The repository separates two questions:

1. Is a declared translation kernel positive semidefinite?
2. Does the actual colorable-smoothing predicate define such a kernel?

The first can be answered by the finite Bochner criterion: a translation
kernel on `Z_2^E` is positive semidefinite exactly when its Walsh spectrum is
nonnegative. The second fails already on the `K4` smoothing cube.

## Pages 8-10: Max/GHZ rotation

The Max and GHZ pictures are related by a quarter-turn and hence by Fourier
duality. For this project, the relevant lesson is structural: a simple
observable in one product can become a nonlocal observable in the dual
product. The smoothing nullity weight has a positive spectral description,
while the colorability indicator does not.

## Page 11: graph duality

The paper connects pictorial Fourier duality with graph duality on a sphere
and then with modular-tensor-category identities. This motivates checking
dual graph observables, but the repository does not import the displayed
Verlinde or `6j` identities into ordinary three-color Penrose arithmetic.
Such an import would require a declared simulation functor and normalization.

## Page 12: the simulation clock

The progression between mathematical problems, picture languages, and
simulations warns against treating a visual analogy as a theorem. Every
repository picture gate therefore records:

- the picture objects and legal moves;
- the target mathematical object;
- the simulation map;
- exact identities preserved by the map; and
- virtual pictures or operations that have no target interpretation.

The page's discrete-to-continuum question remains open here. Finite smoothing
cubes and finite planar slices are not presented as continuum limits.

## Page 13: quon CNOT

The quon CNOT illustrates how a topological picture can encode a quantum
operation. It does not imply that the current smoothing language is a quantum
algorithm. A computational-complexity or topological-quantum-computing claim
would require a gate representation, approximation theorem, and resource
analysis absent from the present finite coloring experiments.

## New exact smoothing consequence

Let `L_s` be the mod-2 Laplacian of the edge subset `s` and let
`nu(s) = dim ker L_s`. For every nonnegative integer `r`,

```text
f_r(s) = 2 ^ (r nu(s))
```

is positive definite on `Z_2^E`.

Indeed, `f_r(s)` counts `r`-tuples `(x_1,...,x_r)` satisfying
`L_s x_j = 0`. For each fixed tuple, the admissible edge subsets `s` form a
linear subspace because the constraints are linear in `s`. The indicator of a
subspace has a nonnegative Walsh transform, supported on its orthogonal
subspace. Summing these indicators proves the claim.

This positivity is special. The weight `3^nu` has a negative Walsh
coefficient on planar graph-atlas fixture 48, and the 3-colorable-smoothing
indicator has negative Walsh coefficients already for `K4`.

## Game-theory boundary

If each edge is a binary player and every player receives common payoff
`nu(s)`, strict Nash equilibria are exactly the strict local maxima already
studied in the smoothing landscape. This is an identical-interest potential
game, not a Prisoner's Dilemma. The graph-atlas counterexamples show that a
nullity equilibrium need not satisfy the global 3-colorability objective.

The user supplied the Veritasium overview
[*What The Prisoner's Dilemma Reveals About Life, The Universe, and
Everything*](https://www.veritasium.com/videos/2024/1/15/what-the-prisoners-dilemma-reveals-about-life-the-universe-and-everything)
as a conceptual prompt. Its Axelrod-style cooperation setting motivates
asking for player-level incentives, but it is not used as mathematical
evidence for the smoothing claims.
