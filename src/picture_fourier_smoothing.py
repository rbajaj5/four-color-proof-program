"""Exact Walsh-Fourier diagnostics on Boolean smoothing pictures."""

from __future__ import annotations

from collections.abc import Sequence


def walsh_hadamard(values: Sequence[int]) -> list[int]:
    """Return the unnormalized Walsh-Hadamard transform."""
    output = [int(value) for value in values]
    if not output or len(output) & (len(output) - 1):
        raise ValueError("input length must be a positive power of two")
    width = 1
    while width < len(output):
        for start in range(0, len(output), 2 * width):
            for offset in range(width):
                left = output[start + offset]
                right = output[start + width + offset]
                output[start + offset] = left + right
                output[start + width + offset] = left - right
        width *= 2
    return output


def xor_convolution(
    first: Sequence[int],
    second: Sequence[int],
) -> list[int]:
    """Convolve two functions on a Boolean cube."""
    if len(first) != len(second) or not first:
        raise ValueError("inputs must have the same positive length")
    if len(first) & (len(first) - 1):
        raise ValueError("input length must be a power of two")
    return [
        sum(
            int(first[state]) * int(second[index ^ state])
            for state in range(len(first))
        )
        for index in range(len(first))
    ]


def pointwise_product(
    first: Sequence[int],
    second: Sequence[int],
) -> list[int]:
    """Multiply two cube functions pointwise."""
    if len(first) != len(second):
        raise ValueError("inputs must have the same length")
    return [
        int(left) * int(right)
        for left, right in zip(first, second, strict=True)
    ]


def inverse_walsh(spectrum: Sequence[int]) -> list[int]:
    """Invert an exactly divisible unnormalized Walsh transform."""
    transformed = walsh_hadamard(spectrum)
    size = len(transformed)
    if any(value % size for value in transformed):
        raise ValueError("spectrum does not have an integral inverse")
    return [value // size for value in transformed]


def strict_local_maximum_indicator(
    values: Sequence[int],
    dimension: int,
) -> list[int]:
    """Mark strict one-bit local maxima of a cube function."""
    if len(values) != 1 << dimension:
        raise ValueError("value count does not match cube dimension")
    return [
        int(
            all(
                int(values[state])
                > int(values[state ^ (1 << coordinate)])
                for coordinate in range(dimension)
            )
        )
        for state in range(len(values))
    ]


def strict_nash_indicator(
    common_payoff: Sequence[int],
    player_count: int,
) -> list[int]:
    """Strict pure equilibria of the identical-interest bit-flip game."""
    return strict_local_maximum_indicator(common_payoff, player_count)


def translation_kernel_is_positive(values: Sequence[int]) -> bool:
    """Test positive semidefiniteness through the finite Bochner criterion."""
    return min(walsh_hadamard(values)) >= 0

