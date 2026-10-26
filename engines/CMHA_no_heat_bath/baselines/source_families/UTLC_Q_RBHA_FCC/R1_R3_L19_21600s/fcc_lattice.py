#!/usr/bin/env python3
"""Face-centered-cubic lattice utilities for Q-RBHA.

The FCC lattice is represented by integer triples whose coordinate sum is
even.  An edge changes two coordinates by one unit, giving twelve neighbors
for an interior site.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FCC_DELTAS = (
    (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
    (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
    (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
)


def fcc_lattice_coords(side: int) -> list[tuple[int, int, int]]:
    if side < 2:
        raise ValueError("FCC lattice side length must be at least 2")
    return [
        (x, y, z)
        for x in range(side)
        for y in range(side)
        for z in range(side)
        if (x + y + z) % 2 == 0
    ]


def fcc_neighbor_sets(
    coords: list[tuple[int, int, int]],
) -> list[set[int]]:
    coord_to_site = {coordinate: site for site, coordinate in enumerate(coords)}
    out: list[set[int]] = []
    for x, y, z in coords:
        neighbors: set[int] = set()
        for dx, dy, dz in FCC_DELTAS:
            site = coord_to_site.get((x + dx, y + dy, z + dz))
            if site is not None:
                neighbors.add(site)
        out.append(neighbors)
    return out


def fcc_graph_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> int:
    dx = abs(first[0] - second[0])
    dy = abs(first[1] - second[1])
    dz = abs(first[2] - second[2])
    return max(dx, dy, dz, (dx + dy + dz) // 2)


def model_context(model_json: Path):
    data = json.loads(model_json.read_text(encoding="utf-8"))
    if data.get("model") != "HP" or data.get("lattice") != "3d_fcc":
        raise ValueError("input JSON must be a 3D FCC HP model")
    coords = [tuple(map(int, coordinate)) for coordinate in data["coords"]]
    coord_to_site = {coordinate: site for site, coordinate in enumerate(coords)}
    neighbor_sets = fcc_neighbor_sets(coords)
    site_count = len(coords)
    neighbor_bool = np.zeros((site_count, site_count), dtype=np.bool_)
    for first, neighbors in enumerate(neighbor_sets):
        for second in neighbors:
            neighbor_bool[first, second] = True
    hmask = np.asarray([1 if residue == "H" else 0 for residue in data["seq"]], dtype=np.int8)
    return data, coords, coord_to_site, neighbor_sets, neighbor_bool, hmask


def validate_lattice(side: int) -> dict[str, int]:
    coords = fcc_lattice_coords(side)
    neighbors = fcc_neighbor_sets(coords)
    interior = [len(values) for site, values in enumerate(neighbors) if all(0 < axis < side - 1 for axis in coords[site])]
    if interior and max(interior) != 12:
        raise AssertionError("an interior FCC site must have twelve neighbors")
    for first, values in enumerate(neighbors):
        for second in values:
            if first not in neighbors[second]:
                raise AssertionError("FCC adjacency must be symmetric")
            if fcc_graph_distance(coords[first], coords[second]) != 1:
                raise AssertionError("FCC edge has incorrect graph distance")
    return {
        "L": side,
        "sites": len(coords),
        "directed_edges": sum(len(values) for values in neighbors),
        "maximum_degree": max(map(len, neighbors)),
    }
