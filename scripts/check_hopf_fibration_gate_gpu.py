"""CUDA audit of classical complex Hopf-fibration identities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hopf_fibration_gate import (  # noqa: E402
    chern_number_midpoint,
    gauss_linking_number,
    hopf_fiber,
    hopf_map,
    phase_rotate,
    projective_projector,
    projector_distance_residual,
    stereographic_s3,
)


OUTPUT_DIR = ROOT / "results" / "hopf_fibration_gate_gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def random_complex(
    rows: int,
    dimension: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.complex(
        torch.randn(
            rows,
            dimension,
            device=device,
            dtype=torch.float64,
            generator=generator,
        ),
        torch.randn(
            rows,
            dimension,
            device=device,
            dtype=torch.float64,
            generator=generator,
        ),
    )


def audit_dimension(
    dimension: int,
    samples: int,
    chunk_size: int,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    maxima = torch.zeros(6, device=device, dtype=torch.float64)
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, samples, chunk_size):
        current = min(chunk_size, samples - start)
        left = random_complex(
            current,
            dimension,
            device=device,
            generator=generator,
        )
        right = random_complex(
            current,
            dimension,
            device=device,
            generator=generator,
        )
        phases = 2.0 * math.pi * torch.rand(
            current,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        projector = projective_projector(left)
        rotated = projective_projector(phase_rotate(left, phases))
        identity_residual = projector @ projector - projector
        trace_residual = (
            projector.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
        )
        hermitian_residual = projector - projector.mH
        distance_residual = projector_distance_residual(left, right)
        phase_residual = projector - rotated
        hopf_residual = torch.zeros((), dtype=torch.float64, device=device)
        if dimension == 2:
            image = hopf_map(left)
            rotated_image = hopf_map(phase_rotate(left, phases))
            hopf_residual = torch.maximum(
                (torch.linalg.vector_norm(image, dim=-1) - 1.0).abs().max(),
                (image - rotated_image).abs().max(),
            )
        chunk_maxima = torch.stack(
            (
                identity_residual.abs().max(),
                trace_residual.abs().max(),
                hermitian_residual.abs().max(),
                distance_residual.abs().max(),
                phase_residual.abs().max(),
                hopf_residual,
            )
        )
        maxima = torch.maximum(maxima, chunk_maxima)
    synchronize(device)
    elapsed = time.perf_counter() - started
    values = maxima.cpu().tolist()
    return {
        "complex_dimension": dimension,
        "projective_space": f"CP^{dimension - 1}",
        "samples": samples,
        "projector_idempotence_max_error": values[0],
        "projector_trace_max_error": values[1],
        "projector_hermitian_max_error": values[2],
        "projector_distance_identity_max_error": values[3],
        "u1_phase_invariance_max_error": values[4],
        "hopf_map_max_error": values[5],
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def main() -> None:
    args = parse_args()
    device = select_device(args.allow_cpu)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = min(args.samples, 10_000) if args.smoke else args.samples
    dimensions = (2,) if args.smoke else (2, 3, 5)
    print(
        f"device={device} gpu="
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    rows = []
    for dimension in dimensions:
        row = audit_dimension(
            dimension,
            samples,
            args.chunk_size,
            device=device,
            seed=args.seed + dimension,
        )
        rows.append(row)
        print(
            f"{row['projective_space']}: samples={samples:,} "
            f"phase_error={row['u1_phase_invariance_max_error']:.3e} "
            f"distance_error={row['projector_distance_identity_max_error']:.3e}"
        )
    theta_steps = 256 if args.smoke else 16_384
    chern_number = chern_number_midpoint(theta_steps, device=device)
    fiber_samples = 256 if args.smoke else 2_048
    first_fiber = stereographic_s3(
        hopf_fiber(
            math.pi / 2.0,
            0.0,
            fiber_samples,
            device=device,
        )
    )
    second_fiber = stereographic_s3(
        hopf_fiber(
            math.pi / 2.0,
            math.pi / 2.0,
            fiber_samples,
            device=device,
        )
    )
    linking_number = gauss_linking_number(first_fiber, second_fiber)
    write_csv(OUTPUT_DIR / "hopf_projective_identity_rows.csv", rows)
    topological_rows = [
        {
            "fixture": "first_chern_number_CP1",
            "resolution": theta_steps,
            "observed": chern_number,
            "expected": 1.0,
            "absolute_error": abs(chern_number - 1.0),
        },
        {
            "fixture": "distinct_hopf_fiber_linking",
            "resolution": fiber_samples,
            "observed": linking_number,
            "expected": -1.0 if linking_number < 0 else 1.0,
            "absolute_error": abs(abs(linking_number) - 1.0),
        },
    ]
    write_csv(OUTPUT_DIR / "hopf_topological_fixture_rows.csv", topological_rows)
    device_report = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
        "nvidia_smi": nvidia_smi(),
    }
    (OUTPUT_DIR / "device_report.json").write_text(
        json.dumps(device_report, indent=2) + "\n",
        encoding="utf-8",
    )
    tolerance = 2e-12
    chern_tolerance = 1e-5 if args.smoke else 1e-8
    linking_tolerance = 1e-4 if args.smoke else 1e-5
    audit = {
        "cuda_used": device.type == "cuda",
        "samples_total": sum(int(row["samples"]) for row in rows),
        "projective_identities_pass": all(
            max(
                float(row["projector_idempotence_max_error"]),
                float(row["projector_trace_max_error"]),
                float(row["projector_hermitian_max_error"]),
                float(row["projector_distance_identity_max_error"]),
                float(row["u1_phase_invariance_max_error"]),
                float(row["hopf_map_max_error"]),
            )
            < tolerance
            for row in rows
        ),
        "chern_number_pass": abs(chern_number - 1.0) < chern_tolerance,
        "fiber_linking_pass": (
            abs(abs(linking_number) - 1.0) < linking_tolerance
        ),
        "chern_number_tolerance": chern_tolerance,
        "fiber_linking_tolerance": linking_tolerance,
        "imports_unification_claims": False,
        "imports_particle_predictions": False,
        "source_peer_reviewed": False,
    }
    (OUTPUT_DIR / "hopf_fibration_gate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    table = "\n".join(
        "| {projective_space} | {samples} | "
        "{projector_idempotence_max_error:.3e} | "
        "{u1_phase_invariance_max_error:.3e} | "
        "{projector_distance_identity_max_error:.3e} | "
        "{hopf_map_max_error:.3e} |".format(**row)
        for row in rows
    )
    report = f"""# Complex Hopf-Fibration Source Gate

## Source

Jennifer Lorraine Nielsen, *The Complex Hopf Fibration as the Canonical
Space for Gauge-Gravity Unification: The Field, Universal Action, and
Particle Spectrum*, DOI `10.20944/preprints202604.0315.v1`. The linked
manuscript is explicitly marked as a non-peer-reviewed preprint.

## Independently reproduced classical layer

| space | samples | idempotence | U(1) invariance | distance identity | Hopf-map error |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

- Numerical first Chern number of `S^3 -> CP^1`:
  **{chern_number:.12f}**.
- Gauss linking number of two distinct projected Hopf fibers:
  **{linking_number:.12f}**.

The projector representation `P_z = z z*` makes the quotient by the common
`U(1)` phase explicit. It also gives the exact projective distance identity

```text
||P_z - P_w||_F^2 = 2(1 - |<z,w>|^2).
```

These identities support the repository's existing Hopf/Bateman fixtures and
give a controlled geometric source for linked field-line tests.

## Claims not imported

This gate does **not** treat the following preprint claims as established:

- that the shell sequence uniquely forces the Standard Model gauge sectors;
- that gravity follows from the bundle without further physical assumptions;
- that a proposed Beltrami spectrum determines particle masses or couplings;
- that anomaly cancellation, dark sectors, or singularity removal follow;
- or that phenomenological numerical predictions are validated.

The standard facts `S^3 ~= SU(2)` and `S^5 ~= SU(3)/SU(2)` do not by
themselves prove that the associated physical gauge sectors are uniquely
selected. Likewise, representability by `BU(1) ~= CP^infinity` is a
classification result under a strong universality hypothesis; its
applicability to a physical unified theory is an additional premise.

## Hex and Four Color relevance

The Hopf fibration is not a Hex or Four Color theorem. Its role here is as a
validated source of linked fibers and projective coordinates for magnetic
fixtures. Any later planar Hex observable is applied only after declaring a
projection or threshold slice.
"""
    (OUTPUT_DIR / "HOPF_FIBRATION_SOURCE_GATE_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
