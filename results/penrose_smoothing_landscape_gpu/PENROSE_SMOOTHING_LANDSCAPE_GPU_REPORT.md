# Penrose Smoothing Landscape CUDA Report

## Scope

This is an exhaustive finite census of the Kauffman-Silver-Williams
smoothing cube. For a plane Tait graph, the GPU computes the mod-2 Laplacian
nullity of every edge-subset state and finds every strict local maximum.
Only those maxima return to the CPU, where their component graphs are
reconstructed and tested for 3-colorability exactly.

This is not a new proof of the Four Color Theorem.

## CUDA audit

- CUDA used: **True**
- Device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- PyTorch: `2.12.0.dev20260408+cu128`
- CUDA runtime: `12.8`
- Peak allocated memory: **1391483904 bytes**
- Exhaustive states processed: **71,938,688**
- CUDA rank/maxima time: **13.965 seconds**

## Odd-wheel pattern

For `W_(2k+2)`, whose rim is the odd cycle `C_(2k+1)`, the observed
nonzero maximum count is

```text
(4^k - 1) / 3.
```

The zero state has component graph equal to the 4-chromatic odd wheel. Every
observed nonzero maximum has nullity three and component graph `K3`.

| Fixture | States | Maxima | Formula | Nonzero 3-colorable | Nonzero not 3-colorable | All nonzero K3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| odd_wheel_W4 | 64 | 2 | 2 | 1 | 0 | True |
| odd_wheel_W6 | 1024 | 6 | 6 | 5 | 0 | True |
| odd_wheel_W8 | 16384 | 22 | 22 | 21 | 0 | True |
| odd_wheel_W10 | 262144 | 86 | 86 | 85 | 0 | True |
| odd_wheel_W12 | 4194304 | 342 | 342 | 341 | 0 | True |
| odd_wheel_W14 | 67108864 | 1366 | 1366 | 1365 | 0 | True |

The count is compatible with a coloring-orbit explanation: fix the hub color,
properly color the odd rim with the other three colors, then quotient by the
six permutations of those colors. The number is
`(2^(2k+1)-2)/6 = (4^k-1)/3`.

That observation is a **proof sketch only**. A complete proof must show that
the orbit-to-state map is bijective and that every nonzero strict maximum has
component graph `K3`.

## Boundary control

- 3-connected, 4-chromatic planar atlas graphs: **27**
- Graphs with at least one 3-colorable maximum: **27**
- Graphs with a nonzero maximum that is not 3-colorable:
  **25**

Thus the attractive assertion

```text
every nonzero strict local maximum is 3-colorable
```

is false already in the small planar atlas. The odd-wheel behavior is
family-specific. This is the main diagnostic result: component-count ascent
alone does not solve the 3-colorability burden in Theorem 6.6.

## Relation to the classical proof

Robertson-Sanders-Seymour-Thomas prove reducibility for 633 configurations
and unavoidability using 32 discharging rules. This census supplies neither
ingredient. Its possible value is narrower: the odd-wheel family gives a
closed-form test case for studying how smoothing maxima encode colorings,
while the atlas rows provide compact counterexamples for proposed universal
landscape rules.

## Status

- Exhaustive CUDA census: `passed`
- CPU reconstruction/nullity agreement: `passed`
- Odd-wheel closed form: `conjecture_with_orbit_proof_sketch`
- Universal nonzero-maximum rule: `falsified`
- New Four Color proof: `false`

## Sources

- Kauffman, Silver, Williams, Theorems 5.3 and 6.6:
  https://arxiv.org/abs/2604.16635
- Robertson, Sanders, Seymour, Thomas:
  https://thomas.math.gatech.edu/FC/fourcolor.html
