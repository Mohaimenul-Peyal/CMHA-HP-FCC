#!/usr/bin/env python3
"""Independently validate a saved 3D FCC HP fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcc_lattice import FCC_DELTAS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.best.read_text(encoding="utf-8"))
    sequence = str(data["seq"])
    coords = [tuple(map(int, coordinate)) for coordinate in data["coords"]]
    delta_set = set(FCC_DELTAS)
    duplicate_count = len(coords) - len(set(coords))
    broken_links = []
    for index in range(len(coords) - 1):
        first, second = coords[index], coords[index + 1]
        delta = tuple(second[axis] - first[axis] for axis in range(3))
        if delta not in delta_set:
            broken_links.append(index)
    contacts = []
    for first in range(len(sequence)):
        if sequence[first] != "H":
            continue
        for second in range(first + 2, len(sequence)):
            if sequence[second] != "H":
                continue
            delta = tuple(coords[second][axis] - coords[first][axis] for axis in range(3))
            if delta in delta_set:
                contacts.append((first, second))
    report = {
        "valid": duplicate_count == 0 and not broken_links,
        "N": len(sequence),
        "coordinate_count": len(coords),
        "duplicate_sites": duplicate_count,
        "broken_chain_links": broken_links,
        "HH": len(contacts),
        "energy": -len(contacts),
        "saved_HH": data.get("HH"),
        "score_matches": int(data.get("HH", -1)) == len(contacts),
    }
    print(json.dumps(report, indent=2))
    if not report["valid"] or not report["score_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
