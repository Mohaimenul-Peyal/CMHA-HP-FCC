"""Unmodified historical initializer, lattice utilities and lightweight exporter."""
import sys
import math
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = PACKAGE_ROOT.parent.parent / 'runs'
HISTORICAL_SOURCE = PACKAGE_ROOT / 'baselines' / 'source_families' / 'UTLC_Q_RBHA_FCC' / 'R1_R3_L19_21600s'
sys.path.insert(0, str(HISTORICAL_SOURCE))
from export_3d_FCC_HP_cluster_qubo_light import export_lightweight_model
from fcc_lattice import fcc_lattice_coords, fcc_neighbor_sets, model_context


def pairwise_contact_saw(*args,**kwargs):
    # Dry-run/profile inspection must not create JIT cache files.
    from hp3d_stage_common import pairwise_contact_saw as initialize
    return initialize(*args,**kwargs)


def validate_coordinates(sequence, coordinates, side):
    """Independent coordinate-based validation, not the incremental site scorer."""
    try:
        if any(len(p)!=3 or any(not math.isfinite(v) or int(v)!=v for v in p) for p in coordinates):
            return {'valid':False,'errors':['nonintegral_or_malformed_coordinates'],'HH':None}
        points = [tuple(map(int, p)) for p in coordinates]
    except (TypeError,ValueError,OverflowError):
        return {'valid':False,'errors':['nonintegral_or_malformed_coordinates'],'HH':None}
    errors = []
    if len(points) != len(sequence):
        errors.append('coordinate_count')
    if len(set(points)) != len(points):
        errors.append('site_collision')
    if any(len(p) != 3 or any(v < 0 or v >= side for v in p) or sum(p) % 2 for p in points):
        errors.append('fcc_box_or_parity')
    if errors:
        return {'valid': False, 'errors': errors, 'HH': None}
    def adjacent(a, b):
        return sum((x-y)**2 for x, y in zip(a, b)) == 2
    if any(not adjacent(points[i], points[i+1]) for i in range(len(points)-1)):
        errors.append('chain_link')
    contacts = [(i, j) for i in range(len(points)) if sequence[i] == 'H'
                for j in range(i+2, len(points)) if sequence[j] == 'H' and adjacent(points[i], points[j])]
    return {'valid': not errors, 'errors': errors, 'HH': len(contacts), 'contacts': contacts}
