# Fixed-Embedding Saturation Checker

This exact checker is the first executable layer beneath the Clifton-Salia
plane-saturation gate.

Given:

- a finite simple planar host graph `G`;
- a connected spanning subgraph `H`;
- fixed vertex labels; and
- a sphere rotation system for `H`,

it enumerates face boundaries and checks every missing host edge for
cofacial endpoints. No missing edge is drawable without a crossing exactly
when the declared fixed embedding is edge-saturated.

This is narrower than plane saturation in Clifton and Salia:

- their subgraph embedding is unlabeled;
- edge addition can introduce vertices;
- disconnected skeletons and isolated-vertex face placement matter; and
- generic knot projections may be multigraphs.

The generated artifacts therefore never label a fixture as a full
Clifton-Salia certificate.

```bash
python scripts/check_fixed_embedding_saturation.py
python -m pytest tests/test_fixed_embedding_saturation.py -q
```

See `results/fixed_embedding_saturation/`.
