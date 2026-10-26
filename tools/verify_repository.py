#!/usr/bin/env python3
"""Check frozen runtime hashes, portable artifact references and release integrity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(full_manifest=False):
    source = json.loads((ROOT / "frozen_source_manifest.json").read_text())
    for row in source:
        path = ROOT / row["repository_path"]
        if not path.is_file() or digest(path) != row["public_sha256"]:
            raise ValueError(f"Frozen source mismatch: {row['repository_path']}")
        if row["public_sha256"] != row["original_sha256"] and row["adaptation"] != "Relative output default in unused legacy batch helper; no search-logic changes":
            raise ValueError("Undeclared adaptation of a frozen source file")
    with (ROOT / "results/PRIMARY_RESULTS.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 31:
        raise ValueError("Expected 31 primary rows")
    for row in rows:
        for field in ("fold_file", "profile"):
            if not (ROOT / row[field]).is_file():
                raise ValueError(f"Missing {field}: {row[field]}")
    if len(list((ROOT / "profiles/ablations").glob("*/*.json"))) != 28:
        raise ValueError("Expected 28 matched-ablation profiles")
    checked = 0
    if full_manifest:
        for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
                raise ValueError(f"Release integrity mismatch: {name}")
            checked += 1
    return dict(frozen_source_files=len(source), primary_profiles=31, ablation_profiles=28,
                release_files_checked=checked, all_checks_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="store_true", help="Also verify every immutable distributed file")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))
