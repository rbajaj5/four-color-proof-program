# NIETTU Hopf Source Audit

## Citation

Jennifer Lorraine Nielsen, *The Complex Hopf Fibration as the Canonical
Space for Gauge-Gravity Unification: The Field, Universal Action, and
Particle Spectrum*, non-peer-reviewed preprint,
DOI [10.20944/preprints202604.0315.v1](https://doi.org/10.20944/preprints202604.0315.v1).
The user supplied the evolving
[PhilArchive copy](https://philpapers.org/archive/NIETTU.pdf).

## Repository use

The paper is used as an exploratory prompt for a narrow, independently
checked Hopf-bundle fixture. The repository adopts only standard identities:

1. `S^(2n+1) -> CP^n` is invariant under a common `U(1)` phase.
2. A projective point can be represented by the rank-one projector `zz*`.
3. The `S^3 -> CP^1 ~= S^2` Hopf map has first Chern number one.
4. Distinct Hopf fibers form a link of linking number one.

These statements are tested in `src/hopf_fibration_gate.py` and the generated
source-gate report.

## Evidential boundary

The preprint's gauge-gravity unification, uniqueness, particle-spectrum, and
phenomenology claims are not used as premises. The shell identifications
`S^3 ~= SU(2)` and `S^5 ~= SU(3)/SU(2)` are classical homogeneous-space
facts, but turning them into uniquely forced physical sectors requires
arguments beyond those identities.

This distinction keeps the useful topology while preventing a speculative
physical proposal from being cited as validation of the Four Color, Hex, or
magnetic-knot experiments.
