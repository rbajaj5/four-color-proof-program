"""Exact majority coarse-graining for triangular Hex/Y boards."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator


Coordinate = tuple[int, int]


def triangular_coordinates(size: int) -> tuple[Coordinate, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    return tuple(
        (q, r)
        for q in range(size)
        for r in range(size - q)
    )


def triangular_neighbors(size: int, coordinate: Coordinate) -> tuple[Coordinate, ...]:
    q, r = coordinate
    candidates = (
        (q - 1, r),
        (q + 1, r),
        (q, r - 1),
        (q, r + 1),
        (q - 1, r + 1),
        (q + 1, r - 1),
    )
    return tuple(
        (left, right)
        for left, right in candidates
        if left >= 0 and right >= 0 and left + right < size
    )


def compact_mask_to_coloring(size: int, mask: int) -> dict[Coordinate, bool]:
    coordinates = triangular_coordinates(size)
    if mask < 0 or mask >= 1 << len(coordinates):
        raise ValueError("mask is outside the triangular board")
    return {
        coordinate: bool((mask >> index) & 1)
        for index, coordinate in enumerate(coordinates)
    }


def coloring_to_compact_mask(
    size: int,
    coloring: dict[Coordinate, bool],
) -> int:
    coordinates = triangular_coordinates(size)
    if set(coloring) != set(coordinates):
        raise ValueError("coloring must contain every board coordinate exactly once")
    return sum(
        int(coloring[coordinate]) << index
        for index, coordinate in enumerate(coordinates)
    )


def majority_reduce(
    size: int,
    coloring: dict[Coordinate, bool],
) -> dict[Coordinate, bool]:
    """Apply the Karlin-Peres triangular majority reduction once."""

    if size < 2:
        raise ValueError("cannot reduce a one-cell board")
    if set(coloring) != set(triangular_coordinates(size)):
        raise ValueError("coloring does not match the board")
    reduced: dict[Coordinate, bool] = {}
    for q, r in triangular_coordinates(size - 1):
        triangle = (
            coloring[q + 1, r],
            coloring[q, r],
            coloring[q, r + 1],
        )
        reduced[q, r] = sum(triangle) >= 2
    return reduced


def reduce_to_single_color(
    size: int,
    coloring: dict[Coordinate, bool],
) -> bool:
    current = dict(coloring)
    for current_size in range(size, 1, -1):
        current = majority_reduce(current_size, current)
    return current[0, 0]


def monochromatic_component(
    size: int,
    coloring: dict[Coordinate, bool],
    *,
    color: bool,
    start: Coordinate,
) -> frozenset[Coordinate]:
    if coloring[start] != color:
        return frozenset()
    seen = {start}
    queue = deque([start])
    while queue:
        coordinate = queue.popleft()
        for neighbor in triangular_neighbors(size, coordinate):
            if neighbor not in seen and coloring[neighbor] == color:
                seen.add(neighbor)
                queue.append(neighbor)
    return frozenset(seen)


def components(
    size: int,
    coloring: dict[Coordinate, bool],
    *,
    color: bool,
) -> Iterator[frozenset[Coordinate]]:
    unseen = {
        coordinate
        for coordinate in triangular_coordinates(size)
        if coloring[coordinate] == color
    }
    while unseen:
        start = min(unseen)
        component = monochromatic_component(
            size,
            coloring,
            color=color,
            start=start,
        )
        yield component
        unseen.difference_update(component)


def component_touches_all_sides(
    size: int,
    component: frozenset[Coordinate],
) -> bool:
    return (
        any(q == 0 for q, _ in component)
        and any(r == 0 for _, r in component)
        and any(q + r == size - 1 for q, r in component)
    )


def has_y(
    size: int,
    coloring: dict[Coordinate, bool],
    *,
    color: bool,
) -> bool:
    return any(
        component_touches_all_sides(size, component)
        for component in components(size, coloring, color=color)
    )


def y_winners(
    size: int,
    coloring: dict[Coordinate, bool],
) -> tuple[bool, bool]:
    """Return ``(blue_has_Y, yellow_has_Y)`` with blue represented by True."""

    return (
        has_y(size, coloring, color=True),
        has_y(size, coloring, color=False),
    )
