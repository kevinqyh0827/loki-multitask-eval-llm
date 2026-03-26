#!/usr/bin/env python3
"""Project XML morphology files to the [11, 14] raw design vector space.

The design vector encodes each morphology as an [11, 14] matrix where:
  - 11 rows = max tokens (torso + up to 10 limbs, zero-padded)
  - 14 columns = per-node features (continuous, categorical, binary, structural)

Usage:
    python tools/xml_to_design_space.py --xml path/to/morphology.xml
    python tools/xml_to_design_space.py --pt path/to/morphology.pt
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_METAMORPH_DIR = os.path.join(_PROJECT_ROOT, "metamorph")

# Column documentation
DESIGN_VECTOR_COLUMNS = [
    "orient_r",       # 0: limb height, normalized [0,1] from [0.2, 0.4]
    "jx_gear",        # 1: joint-x gear, normalized [0,1] from [150, 300]
    "jy_gear",        # 2: joint-y gear, normalized [0,1] from [150, 300]
    "density",        # 3: body density, normalized [0,1] from [500, 1000]
    "theta_class",    # 4: orientation theta class [0,8) -> [0, 45, ..., 315] degrees
    "phi_class",      # 5: orientation phi class [0,3) -> [90, 135, 180] degrees
    "jx_range_class", # 6: joint-x range class [0,13) index into JOINT_ANGLE_LIST
    "jy_range_class", # 7: joint-y range class [0,13) index into JOINT_ANGLE_LIST
    "jx_bit",         # 8: has joint-x (0 or 1)
    "jy_bit",         # 9: has joint-y (0 or 1)
    "torso_mode",     # 10: 0=horizontal_y, 1=vertical (only meaningful for row 0)
    "attach_site",    # 11: -1=torso, 0=btm, 1=mid attachment point
    "EOS",            # 12: end-of-sequence marker (1 on last active limb)
    "depth",          # 13: DFS tree depth (0=torso, 1+=limbs)
]

# Denormalization constants (from tools/config.py)
LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX = 0.2, 0.4
JOINT_GEAR_MIN, JOINT_GEAR_MAX = 150.0, 300.0
# Head (torso) density uses same range as limb density
HEAD_DENSITY_MIN, HEAD_DENSITY_MAX = 500.0, 1000.0
LIMB_DENSITY_MIN, LIMB_DENSITY_MAX = 500.0, 1000.0
THETA_STEP = 45  # degrees
THETA_VALUES = list(range(0, 360, THETA_STEP))  # [0, 45, 90, ..., 315]
PHI_VALUES = [90, 135, 180]  # degrees
TORSO_MODES = {0: "horizontal_y", 1: "vertical"}
ATTACH_SITES = {-1: "torso", 0: "btm", 1: "mid"}
JOINT_ANGLE_LIST = [
    [-30, 0], [0, 30], [-30, 30], [-45, 45], [-45, 0], [0, 45],
    [-60, 0], [0, 60], [-60, 60], [-90, 0], [0, 90], [-60, 30], [-30, 60],
]

N_TOKENS = 11
D_TOKEN = 14


def xml_to_design_vector(xml_path):
    """Convert an XML morphology file to its [11, 14] design vector.

    Internally runs the XML -> PKL -> vec pipeline, handling cwd changes
    required by the metamorph imports and suppressing module-level prints.

    Args:
        xml_path: Absolute or relative path to a UNIMAL XML morphology file.

    Returns:
        Tuple of (vec, mask), each a numpy array of shape [11, 14].
        vec contains the raw design vector (normalized continuous features,
        class indices for categoricals, binary flags, and structural info).
        mask indicates which features are active per token (True = active).
    """
    xml_path = os.path.abspath(xml_path)
    saved_cwd = os.getcwd()
    try:
        os.chdir(_METAMORPH_DIR)

        # Import xml_to_pkl (prints to stdout during execution, we suppress)
        with contextlib.redirect_stdout(io.StringIO()):
            from tools.xml_2_pkl import xml_to_pkl

        # Import pkl_to_vec (has a module-level print at import time, suppress it)
        with contextlib.redirect_stdout(io.StringIO()):
            from tools.pkl_2_vec_new import pkl_to_vec

        # Run the pipeline (both functions print, suppress all output)
        with contextlib.redirect_stdout(io.StringIO()):
            pkl_dict = xml_to_pkl(xml_path)
            vec_tensor, mask_tensor = pkl_to_vec(pkl_dict)
    finally:
        os.chdir(saved_cwd)

    return vec_tensor.numpy().astype(np.float64), mask_tensor.numpy().astype(bool)


def design_vector_from_pt(pt_path):
    """Load a design vector from a .pt file.

    Cluster tar files store morphologies as .pt tensors of shape [11, 28],
    which is [vec | mask] concatenated along the feature dimension.

    Args:
        pt_path: Path to a .pt file containing a [11, 28] tensor.

    Returns:
        Tuple of (vec, mask), each a numpy array of shape [11, 14].
    """
    combined = torch.load(pt_path, map_location="cpu", weights_only=True)
    if combined.dim() == 2 and combined.shape == (N_TOKENS, D_TOKEN * 2):
        vec = combined[:, :D_TOKEN].numpy().astype(np.float64)
        mask = combined[:, D_TOKEN:].numpy().astype(bool)
    elif combined.dim() == 2 and combined.shape == (N_TOKENS, D_TOKEN):
        # Plain vec without mask
        vec = combined.numpy().astype(np.float64)
        mask = np.ones_like(vec, dtype=bool)
    else:
        raise ValueError(
            f"Unexpected tensor shape {combined.shape}; "
            f"expected [{N_TOKENS}, {D_TOKEN * 2}] or [{N_TOKENS}, {D_TOKEN}]"
        )
    return vec, mask


def get_sequence_length(vec):
    """Determine the number of active tokens in a design vector.

    Finds the EOS marker (column 12) to determine how many rows are real
    (non-padding) tokens.

    Args:
        vec: numpy array of shape [11, 14].

    Returns:
        Integer number of active tokens (rows), between 1 and 11.
    """
    eos_col = vec[:, 12]
    eos_indices = np.where(eos_col == 1)[0]
    if len(eos_indices) > 0:
        return int(eos_indices[0]) + 1

    # Fallback: find the last non-zero row
    for i in range(vec.shape[0] - 1, -1, -1):
        if np.any(vec[i] != 0):
            return i + 1
    return 1


def _denormalize(value, min_val, max_val):
    """Reverse min-max normalization: [0, 1] -> [min_val, max_val]."""
    return value * (max_val - min_val) + min_val


def design_vector_to_readable(vec, mask=None):
    """Convert a raw design vector to a human-readable list of dicts.

    For each active token (row), denormalizes continuous features, converts
    class indices to meaningful values, and maps codes to string labels.

    Args:
        vec: numpy array of shape [11, 14].
        mask: optional numpy array of shape [11, 14] (True = feature active).

    Returns:
        List of dicts, one per active token. Each dict contains:
          - 'token_idx': row index (0 = torso)
          - 'is_torso': bool
          - For each of the 14 features: a sub-dict with 'raw', 'decoded',
            and optionally 'masked' (if mask provided).
    """
    seq_len = get_sequence_length(vec)
    result = []

    for row_idx in range(seq_len):
        row = vec[row_idx]
        row_mask = mask[row_idx] if mask is not None else None
        is_torso = (row_idx == 0)

        token = {"token_idx": row_idx, "is_torso": is_torso}

        def _feature(col, name, raw_val, decoded_val):
            entry = {"raw": raw_val, "decoded": decoded_val}
            if row_mask is not None:
                entry["masked"] = bool(row_mask[col])
            token[name] = entry

        # Column 0: orient_r (limb height)
        if is_torso:
            _feature(0, "orient_r", float(row[0]), "N/A (torso)")
        else:
            height = _denormalize(row[0], LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX)
            _feature(0, "orient_r", float(row[0]), f"{height:.4f} m")

        # Column 1: jx_gear
        if is_torso:
            _feature(1, "jx_gear", float(row[1]), "N/A (torso)")
        else:
            gear_x = _denormalize(row[1], JOINT_GEAR_MIN, JOINT_GEAR_MAX)
            _feature(1, "jx_gear", float(row[1]), f"{gear_x:.1f}")

        # Column 2: jy_gear
        if is_torso:
            _feature(2, "jy_gear", float(row[2]), "N/A (torso)")
        else:
            gear_y = _denormalize(row[2], JOINT_GEAR_MIN, JOINT_GEAR_MAX)
            _feature(2, "jy_gear", float(row[2]), f"{gear_y:.1f}")

        # Column 3: density
        if is_torso:
            density = _denormalize(row[3], HEAD_DENSITY_MIN, HEAD_DENSITY_MAX)
        else:
            density = _denormalize(row[3], LIMB_DENSITY_MIN, LIMB_DENSITY_MAX)
        _feature(3, "density", float(row[3]), f"{density:.1f} kg/m^3")

        # Column 4: theta_class
        theta_idx = int(row[4])
        if theta_idx >= 0 and theta_idx < len(THETA_VALUES):
            theta_deg = THETA_VALUES[theta_idx]
            _feature(4, "theta_class", theta_idx, f"{theta_deg} deg")
        else:
            _feature(4, "theta_class", theta_idx, "N/A")

        # Column 5: phi_class
        phi_idx = int(row[5])
        if phi_idx >= 0 and phi_idx < len(PHI_VALUES):
            phi_deg = PHI_VALUES[phi_idx]
            _feature(5, "phi_class", phi_idx, f"{phi_deg} deg")
        else:
            _feature(5, "phi_class", phi_idx, "N/A")

        # Column 6: jx_range_class
        jx_cls = int(row[6])
        if jx_cls >= 0 and jx_cls < len(JOINT_ANGLE_LIST):
            _feature(6, "jx_range_class", jx_cls, str(JOINT_ANGLE_LIST[jx_cls]))
        else:
            _feature(6, "jx_range_class", jx_cls, "N/A")

        # Column 7: jy_range_class
        jy_cls = int(row[7])
        if jy_cls >= 0 and jy_cls < len(JOINT_ANGLE_LIST):
            _feature(7, "jy_range_class", jy_cls, str(JOINT_ANGLE_LIST[jy_cls]))
        else:
            _feature(7, "jy_range_class", jy_cls, "N/A")

        # Column 8: jx_bit
        _feature(8, "jx_bit", int(row[8]), "yes" if row[8] == 1 else "no")

        # Column 9: jy_bit
        _feature(9, "jy_bit", int(row[9]), "yes" if row[9] == 1 else "no")

        # Column 10: torso_mode
        mode_val = int(row[10])
        mode_str = TORSO_MODES.get(mode_val, "N/A")
        _feature(10, "torso_mode", mode_val, mode_str)

        # Column 11: attach_site
        site_val = int(row[11])
        site_str = ATTACH_SITES.get(site_val, "N/A")
        _feature(11, "attach_site", site_val, site_str)

        # Column 12: EOS
        _feature(12, "EOS", int(row[12]), "yes" if row[12] == 1 else "no")

        # Column 13: depth
        _feature(13, "depth", int(row[13]), int(row[13]))

        result.append(token)

    return result


def print_design_vector(vec, mask=None):
    """Pretty-print a design vector as a formatted table.

    Shows column names, raw values, and denormalized/decoded values for
    each active (non-padded) row.

    Args:
        vec: numpy array of shape [11, 14].
        mask: optional numpy array of shape [11, 14].
    """
    readable = design_vector_to_readable(vec, mask)
    seq_len = len(readable)

    print(f"Design vector: {seq_len} active tokens (of {N_TOKENS} max)")
    print("=" * 90)

    for token in readable:
        row_idx = token["token_idx"]
        label = "TORSO" if token["is_torso"] else f"LIMB {row_idx}"
        print(f"\n--- Token {row_idx}: {label} ---")

        header = f"  {'Feature':<16s} {'Raw':>8s}   {'Decoded':<20s}"
        if mask is not None:
            header += f" {'Active':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for col_idx, col_name in enumerate(DESIGN_VECTOR_COLUMNS):
            feat = token[col_name]
            raw_str = f"{feat['raw']:>8.3f}" if isinstance(feat["raw"], float) else f"{feat['raw']:>8d}"
            decoded_str = f"{feat['decoded']:<20s}" if isinstance(feat["decoded"], str) else f"{str(feat['decoded']):<20s}"
            line = f"  {col_name:<16s} {raw_str}   {decoded_str}"
            if mask is not None:
                active = feat.get("masked", False)
                line += f" {'*' if active else ' ':>6s}"
            print(line)

    print("\n" + "=" * 90)
    print(f"Raw tensor shape: [{vec.shape[0]}, {vec.shape[1]}]")
    if mask is not None:
        active_count = int(mask[:seq_len].sum())
        total_count = seq_len * D_TOKEN
        print(f"Active features: {active_count}/{total_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Project XML morphology files to the [11, 14] raw design vector space."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xml", type=str, help="Path to a UNIMAL XML morphology file")
    group.add_argument("--pt", type=str, help="Path to a .pt design vector file ([11, 28] or [11, 14])")
    parser.add_argument(
        "--raw", action="store_true",
        help="Print raw numpy arrays instead of the formatted table"
    )
    args = parser.parse_args()

    if args.xml:
        print(f"Converting XML: {args.xml}")
        vec, mask = xml_to_design_vector(args.xml)
    else:
        print(f"Loading .pt: {args.pt}")
        vec, mask = design_vector_from_pt(args.pt)

    if args.raw:
        print(f"\nDesign vector (shape {vec.shape}):")
        print(vec)
        print(f"\nMask (shape {mask.shape}):")
        print(mask)
    else:
        print()
        print_design_vector(vec, mask)


if __name__ == "__main__":
    main()
