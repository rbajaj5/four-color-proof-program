"""Generate exact reflection-positivity fixtures for planar pictures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.picture_language_reflection import (  # noqa: E402
    amplitude_inner_product,
    boundary_amplitudes,
    feature_gram_matrix,
    loop_gram_matrix,
    noncrossing_pairings,
    reflect_pairing,
)


OUTPUT_DIR = ROOT / "results" / "picture_language_reflection"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gram_rows: list[dict[str, object]] = []
    matrices: dict[tuple[int, int], np.ndarray] = {}
    for pair_count in range(1, 6):
        pairings = noncrossing_pairings(pair_count)
        boundary_count = 2 * pair_count
        reflection_passed = all(
            reflect_pairing(
                reflect_pairing(pairing, boundary_count),
                boundary_count,
            )
            == pairing
            for pairing in pairings
        )
        for loop_weight in (2, 3):
            matrix = loop_gram_matrix(pairings, loop_weight)
            feature_matrix = feature_gram_matrix(pairings, loop_weight)
            symbolic = sympy.Matrix(matrix)
            eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
            matrices[(pair_count, loop_weight)] = np.asarray(
                matrix,
                dtype=float,
            )
            gram_rows.append(
                {
                    "pair_count": pair_count,
                    "catalan_dimension": len(pairings),
                    "loop_weight": loop_weight,
                    "reflection_involution": reflection_passed,
                    "feature_factorization_exact": matrix == feature_matrix,
                    "rank": symbolic.rank(),
                    "determinant": symbolic.det(),
                    "minimum_eigenvalue": float(eigenvalues[0]),
                    "positive_definite_numerically": bool(
                        eigenvalues[0] > 1e-10
                    ),
                }
            )

    tripod = boundary_amplitudes((("a", "b", "c"),), ("a", "b", "c"))
    reversed_tripod = boundary_amplitudes(
        (("b", "a", "c"),),
        ("a", "b", "c"),
    )
    channel = boundary_amplitudes(
        (
            ("a", "b", "x"),
            ("x", "c", "d"),
        ),
        ("a", "b", "c", "d"),
    )
    penrose_rows = [
        {
            "fixture": "tripod_reflection_double",
            "boundary_size": 3,
            "evaluation": amplitude_inner_product(tripod, tripod),
            "is_norm_square": True,
            "nonnegative": True,
            "role": "positive reflected double",
        },
        {
            "fixture": "reversed_tripod_reflection_double",
            "boundary_size": 3,
            "evaluation": amplitude_inner_product(
                reversed_tripod,
                reversed_tripod,
            ),
            "is_norm_square": True,
            "nonnegative": True,
            "role": "positive reflected double",
        },
        {
            "fixture": "tripod_mixed_orientation",
            "boundary_size": 3,
            "evaluation": amplitude_inner_product(
                tripod,
                reversed_tripod,
            ),
            "is_norm_square": False,
            "nonnegative": False,
            "role": "mixed-gluing obstruction",
        },
        {
            "fixture": "two_vertex_channel_reflection_double",
            "boundary_size": 4,
            "evaluation": amplitude_inner_product(channel, channel),
            "is_norm_square": True,
            "nonnegative": True,
            "role": "positive reflected double",
        },
    ]

    write_csv(OUTPUT_DIR / "loop_gram_fixtures.csv", gram_rows)
    write_csv(OUTPUT_DIR / "penrose_half_picture_fixtures.csv", penrose_rows)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for axis, loop_weight in zip(axes, (2, 3), strict=True):
        matrix = matrices[(5, loop_weight)]
        image = axis.imshow(np.log10(matrix), cmap="viridis")
        axis.set_title(
            f"n=5 pairing kernel, loop weight {loop_weight}"
        )
        axis.set_xlabel("noncrossing pairing")
        axis.set_ylabel("noncrossing pairing")
        figure.colorbar(image, ax=axis, label="log10 evaluation")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "loop_gram_heatmaps.png",
        dpi=180,
    )
    plt.close(figure)

    minimum_eigenvalue = min(
        float(row["minimum_eigenvalue"]) for row in gram_rows
    )
    audit = {
        "pairing_gram_rows": len(gram_rows),
        "maximum_pair_count": 5,
        "maximum_catalan_dimension": 42,
        "all_reflection_involutions_passed": all(
            bool(row["reflection_involution"]) for row in gram_rows
        ),
        "all_feature_factorizations_exact": all(
            bool(row["feature_factorization_exact"]) for row in gram_rows
        ),
        "all_gram_matrices_positive_definite_numerically": all(
            bool(row["positive_definite_numerically"]) for row in gram_rows
        ),
        "minimum_observed_eigenvalue": minimum_eigenvalue,
        "penrose_reflection_doubles_nonnegative": all(
            int(row["evaluation"]) >= 0
            for row in penrose_rows
            if bool(row["is_norm_square"])
        ),
        "negative_mixed_gluing_observed": any(
            int(row["evaluation"]) < 0
            for row in penrose_rows
            if not bool(row["is_norm_square"])
        ),
        "four_color_proof_claimed": False,
        "literature_novelty_claimed": False,
    }
    (OUTPUT_DIR / "picture_language_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    table = "\n".join(
        "| {pair_count} | {catalan_dimension} | {loop_weight} | "
        "{rank} | {minimum_eigenvalue:.6g} | {feature_factorization_exact} |".format(
            **row
        )
        for row in gram_rows
    )
    penrose_table = "\n".join(
        "| {fixture} | {evaluation} | {is_norm_square} | {role} |".format(
            **row
        )
        for row in penrose_rows
    )
    report = f"""# Jaffe-Liu Picture-Language Reflection Gate

## Exact pairing kernel

For each noncrossing pairing `P` of `2n` boundary points, assign the indicator
vector of all `d`-color boundary assignments constant on the pairs of `P`.
The exact inner product of two such vectors is

```text
G(P,Q) = d ^ loops(P glued to Q).
```

This is a constructive Gram factorization, not merely a numerical PSD test.

| pairs | Catalan dimension | loop weight | rank | min eigenvalue | exact feature factorization |
| ---: | ---: | ---: | ---: | ---: | --- |
{table}

![Loop Gram heatmaps](loop_gram_heatmaps.png)

## Penrose half-pictures

The boundary vector of each half-picture is obtained by exactly contracting
all internal colors of the signed three-color Levi-Civita tensor.

| fixture | evaluation | norm square | role |
| --- | ---: | --- | --- |
{penrose_table}

A reflected double is `sum_c A(c)^2` and is nonnegative. The orientation-
reversed tripod has boundary vector `-A`, so its mixed gluing with the original
tripod is `-6`. Thus arbitrary gluing is not itself a positive quantity.

## Consequence for the Four Color program

The picture-language formalism supplies a precise target: rewrite a useful
smoothing reduction or diagram evaluation as a reflected double, or as a sum
of such doubles, while preserving the colorable-state existence predicate.
The present fixtures prove positivity only for the declared kernels. They do
not show that a general Penrose contraction has this form and do not establish
a new Four Color theorem.

## Audit

- Reflection is involutive on every enumerated pairing.
- Every loop kernel equals its explicit color-feature Gram matrix.
- Every reflected Penrose fixture is nonnegative.
- A negative mixed-gluing control is retained.
- No literature-novelty or Four Color proof claim is made.
"""
    (OUTPUT_DIR / "PICTURE_LANGUAGE_REFLECTION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

