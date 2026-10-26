"""Sequence-only catalogs for bounded P-run windows with movable H flanks."""
import numpy as np


def build_catalog(hmask, stage_caps=(12, 16, 20, 24)):
    hmask = np.asarray(hmask, dtype=np.int8)
    n = len(hmask)
    runs = []
    i = 0
    while i < n:
        if hmask[i]:
            i += 1
            continue
        a = i
        while i < n and not hmask[i]:
            i += 1
        runs.append((a, i))
    tables = {}
    for stage_cap in stage_caps:
        cap = min(12, int(stage_cap) // 2)
        if cap in tables:
            continue
        rows = []
        for rid, (p, q) in enumerate(runs):
            left, right = p - 1 if p > 0 else -1, q if q < n else -1
            if left < 0 and right < 0:
                continue
            lo, hi = max(0, p - 1), min(n, q + 1)
            length = max(6, hi - lo)
            if length > cap or length > n:
                continue
            for a in range(max(0, hi - length), min(lo, n - length) + 1):
                rows.append((a, a + length, rid, left, right))
        windows = np.asarray(rows, dtype=np.int32).reshape((-1, 5))
        members, offsets = [], [0]
        for anchor in range(n):
            members.extend(w for w, row in enumerate(rows) if anchor in row[3:5])
            offsets.append(len(members))
        tables[cap] = (windows, np.asarray(offsets, np.int32), np.asarray(members, np.int32))
    return runs, tables


def describe_catalog(hmask, stage_caps=(12, 16, 20, 24)):
    runs, tables = build_catalog(hmask, stage_caps)
    return dict(indexing='zero-based, half-open intervals',
        runs=[dict(run_id=i, start=a, end=b, length=b-a) for i, (a, b) in enumerate(runs)],
        stage_tables={str(cap): dict(window_count=len(windows),
            eligible_run_ids=sorted(set(map(int, windows[:, 2]))),
            columns=['start', 'end', 'run_id', 'left_H_flank', 'right_H_flank'],
            windows=windows.tolist()) for cap, (windows, _, _) in tables.items()},
        policy='Whole P runs plus all available adjacent H flanks; pad to six residues. '
               'Skip oversized runs rather than exceed existing caps. No P-only window.')
