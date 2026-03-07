"""Reverse pipeline: Convert VAE decoder output back to MuJoCo XML morphologies.

This mirrors the forward pipeline in pkl_2_vec_new.py and xml_2_pkl.py but in reverse:
    VAE decoder output -> postprocess -> 14-dim vec -> MuJoCo XML string

Usage:
    from tools.vec_to_morphology import postprocess_vae_output, vec_to_xml, vec_to_xml_safe
"""

import os
import sys
import numpy as np
import torch
from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from derl.derl.utils.geom import sph2cart
from derl.derl.utils import xml as xu
from tools.config import (
    HEAD_DENSITY_MIN, HEAD_DENSITY_MAX,
    LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX,
    LIMB_DENSITY_MIN, LIMB_DENSITY_MAX,
    JOINT_GEAR_MIN, JOINT_GEAR_MAX,
)
from tools.util import (
    denormalize, check_dfs_validity,
    THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS,
    TOTAL_CATEGORY, BINARY_TOKEN, CONTINUOUS_TOKEN,
)

try:
    from derl.config import cfg
except ImportError:
    from derl.derl.config import cfg

# Categorical value tables
THETA_VALUES = np.radians(np.arange(THETA_CLASS) * 45)  # [0, 45, ..., 315] degrees
PHI_VALUES = np.radians(np.arange(PHI_CLASS) * 45 + 90)  # [90, 135, 180] degrees
JOINT_ANGLE_LIST = cfg.BODY.JOINT_ANGLE_LIST  # 13 entries

# Fixed constants from morphology generation
TORSO_RADIUS = 0.10
LIMB_RADIUS = 0.05
HEAD_Z_POS = 2.0  # conservative height; render camera auto-fits anyway
CONDIM = 3

# Template path
TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "derl", "derl", "envs", "assets", "unimal.xml"
)


def postprocess_vae_output(recon_x_num, recon_x_cat, recon_x_depth):
    """Convert raw VAE decoder outputs to a 14-dim vec matching pkl_2_vec_new format.

    Args:
        recon_x_num: [B, 11, 4] sigmoid-activated continuous values
        recon_x_cat: [B, 11, 42] logits for categoricals (37) + binary (5)
        recon_x_depth: [B, 11, 11] depth classification logits

    Returns:
        Tensor [B, 11, 14] in the same format as pkl_2_vec_new output:
            [orient_r, jx_gear, jy_gear, density,
             theta_class, phi_class, jx_range_class, jy_range_class,
             jx_bit, jy_bit, torso_mode, attach_site, EOS,
             depth]
    """
    # Split categorical logits: theta(8) + phi(3) + jx_range(13) + jy_range(13) + binary(5)
    cat_logits, binary_logits = torch.split(
        recon_x_cat, [TOTAL_CATEGORY, BINARY_TOKEN], dim=-1
    )
    theta_logits, phi_logits, jx_logits, jy_logits = torch.split(
        cat_logits, [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS], dim=-1
    )

    # Argmax for categoricals
    theta_class = torch.argmax(theta_logits, dim=-1, keepdim=True).float()
    phi_class = torch.argmax(phi_logits, dim=-1, keepdim=True).float()
    jx_range_class = torch.argmax(jx_logits, dim=-1, keepdim=True).float()
    jy_range_class = torch.argmax(jy_logits, dim=-1, keepdim=True).float()

    # Threshold for binaries: [jx_bit, jy_bit, torso_mode, attach_site, EOS]
    binary_vals = (binary_logits > 0).float()

    # Depth: argmax
    depth = torch.argmax(recon_x_depth, dim=-1, keepdim=True).float()

    # Assemble [B, 11, 14]
    vec = torch.cat([
        recon_x_num,                # 4: orient_r, jx_gear, jy_gear, density
        theta_class,                # 1
        phi_class,                  # 1
        jx_range_class,             # 1
        jy_range_class,             # 1
        binary_vals,                # 5: jx_bit, jy_bit, torso_mode, attach_site, EOS
        depth,                      # 1
    ], dim=-1)

    return vec


def _find_seq_len(vec):
    """Find sequence length from EOS bit (position 12).

    Zero-padding mechanism (from the LOKI paper):
    Each morphology is represented as a sequence of [torso, limb_1, ..., limb_N]
    where N varies (2 to 10 limbs). If a morphology has fewer than N_TOKENS (11)
    limbs+torso, the sequence is zero-padded: rows beyond the last real limb are
    all zeros. The EOS bit (position 12) = 1 marks the LAST real limb.

    After VAE decoding, padded positions may produce non-zero outputs because the
    VAE is generative. We rely on EOS to find the true sequence boundary.

    Returns: seq_len >= 2 (torso + at least 1 limb)
    """
    for i in range(vec.shape[0]):
        if vec[i, 12] == 1:
            return max(i + 1, 2)
    # No EOS found (can happen with interpolated latents) —
    # find last row with non-zero depth or meaningful continuous values
    for i in range(vec.shape[0] - 1, 0, -1):
        if vec[i, 13] > 0 or any(abs(vec[i, j]) > 0.01 for j in range(4)):
            return i + 1
    return 2


def _apply_mask_conventions(vec, seq_len):
    """Apply the per-feature mask conventions from the forward pipeline.

    The forward pipeline (pkl_2_vec_new.py) produces a per-feature boolean mask
    for each token. During VAE training, masked features are zeroed out via
    `masked_fill(mask == 0, 0)` and excluded from loss computation. However,
    after VAE decoding (especially from interpolated latents), the decoder may
    output arbitrary values at masked positions. This function enforces the
    correct mask conventions so that vec_to_xml reads only valid features.

    Mask rules:
      Row 0 (torso, depth=0):
        Active:  density[3], torso_mode[10], EOS[12], depth[13]
        Masked:  orient_r[0], jx_gear[1], jy_gear[2], theta[4], phi[5],
                 jx_range[6], jy_range[7], jx_bit[8], jy_bit[9], attach_site[11]

      Rows 1..seq_len-1 (limbs):
        Active:  orient_r[0], density[3], theta[4], phi[5], jx_bit[8],
                 jy_bit[9], EOS[12], depth[13]
        Conditionally active (only if corresponding bit=1):
                 jx_gear[1] and jx_range[6] if jx_bit[8]=1
                 jy_gear[2] and jy_range[7] if jy_bit[9]=1
        Conditionally active: attach_site[11] (only if parent is a limb, not torso)
        Masked:  torso_mode[10]

      Rows seq_len..10 (zero-padded):
        ALL features zeroed out — these tokens represent nothing.
    """
    # Torso row: zero out masked features
    vec[0, 0] = 0.0   # orient_r
    vec[0, 1] = 0.0   # jx_gear
    vec[0, 2] = 0.0   # jy_gear
    vec[0, 4] = 0      # theta
    vec[0, 5] = 0      # phi
    vec[0, 6] = 0      # jx_range
    vec[0, 7] = 0      # jy_range
    vec[0, 8] = 0      # jx_bit
    vec[0, 9] = 0      # jy_bit
    vec[0, 11] = 0     # attach_site
    vec[0, 13] = 0     # depth (torso is always 0)

    # Limb rows: enforce joint mask consistency
    for i in range(1, seq_len):
        jx_bit = int(round(float(vec[i, 8])))
        jy_bit = int(round(float(vec[i, 9])))
        if jx_bit == 0:
            vec[i, 1] = 0.0  # jx_gear masked
            vec[i, 6] = 0    # jx_range masked
        if jy_bit == 0:
            vec[i, 2] = 0.0  # jy_gear masked
            vec[i, 7] = 0    # jy_range masked
        vec[i, 10] = 0       # torso_mode masked for limbs

    # Zero out padded rows beyond EOS
    if seq_len < vec.shape[0]:
        vec[seq_len:, :] = 0


def _repair_depth_sequence(depths):
    """Attempt to repair an invalid depth sequence."""
    if len(depths) == 0:
        return [0]
    repaired = [0]  # torso is always depth 0
    for i in range(1, len(depths)):
        d = int(depths[i])
        prev = repaired[-1]
        if d > prev + 1:
            d = prev + 1
        if d < 1:
            d = 1
        repaired.append(d)
    return repaired


def _repair_joints(vec, seq_len):
    """Ensure each non-torso limb has at least one joint bit set.

    In the forward pipeline, every limb must have at least one joint
    (either x, y, or xy). If the VAE decodes both jx_bit and jy_bit
    as 0, we force jx_bit=1 to create a valid morphology.
    """
    for i in range(1, seq_len):
        jx_bit = vec[i, 8]
        jy_bit = vec[i, 9]
        if jx_bit == 0 and jy_bit == 0:
            vec[i, 8] = 1


def vec_to_xml(vec, template_path=None):
    """Convert a [11, 14] vec to a MuJoCo XML string.

    Args:
        vec: Tensor or ndarray of shape [11, 14] (or [N, 14] with N <= 11)
        template_path: Path to unimal.xml template (uses default if None)

    Returns:
        XML string suitable for MuJoCo loading

    Raises:
        ValueError: if the morphology cannot be constructed
    """
    if template_path is None:
        template_path = TEMPLATE_PATH

    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy().copy()
    else:
        vec = np.array(vec, dtype=float).copy()

    # Step 1: Find sequence length
    seq_len = _find_seq_len(torch.tensor(vec) if not isinstance(vec, torch.Tensor) else vec)
    seq_len = min(seq_len, 11)
    if seq_len < 2:
        raise ValueError("Morphology must have at least torso + 1 limb")

    # Step 2: Apply mask conventions (zero out masked/padded features)
    _apply_mask_conventions(vec, seq_len)

    # Step 3: Extract and validate depth sequence
    depths = vec[:seq_len, 13].astype(int).tolist()
    if not check_dfs_validity(depths):
        depths = _repair_depth_sequence(depths)
        if not check_dfs_validity(depths):
            raise ValueError(f"Cannot repair depth sequence: {depths}")
        for i in range(seq_len):
            vec[i, 13] = depths[i]

    # Step 3: Repair joints
    _repair_joints(vec, seq_len)

    # Step 4: Parse torso (row 0)
    torso_density = denormalize(float(vec[0, 3]), HEAD_DENSITY_MIN, HEAD_DENSITY_MAX)
    torso_density = np.clip(torso_density, HEAD_DENSITY_MIN, HEAD_DENSITY_MAX)
    torso_mode_val = int(round(float(vec[0, 10])))
    torso_mode = "vertical" if torso_mode_val == 1 else "horizontal_y"

    # Step 5: Load template and build XML
    root, tree = xu.etree_from_xml(template_path)
    worldbody = root.find("worldbody")
    actuator_elem = root.find("actuator")
    sensor_elem = root.find("sensor")

    # Build torso
    torso = xu.body_elem("torso/0", [0, 0, HEAD_Z_POS])
    # Free joint
    torso.append(xu.joint_elem("root", "free", "free"))
    # IMU site
    torso.append(xu.site_elem("root", None, "imu_vel"))
    # Head geom
    torso.append(etree.Element("geom", {
        "name": "torso/0",
        "type": "sphere",
        "size": str(TORSO_RADIUS),
        "condim": str(CONDIM),
        "density": str(round(torso_density, 1)),
    }))
    # Camera
    torso.append(etree.fromstring(
        '<camera name="side" pos="0 -7 2" xyaxes="1 0 0 0 1 2" mode="trackcom"/>'
    ))
    # Growth site at origin
    torso.append(xu.site_elem("torso/0", [0, 0, 0], "growth_site"))
    # Bottom position site
    torso.append(xu.site_elem("torso/btm_pos/0", [0, 0, -TORSO_RADIUS], "btm_pos_site"))
    # Touch sensor site
    torso.append(xu.site_elem(
        "torso/touch/0", None, "touch_site", str(TORSO_RADIUS + 0.01)
    ))
    # Torso growth site based on mode
    if torso_mode == "horizontal_y":
        torso.append(xu.site_elem(
            f"torso/{torso_mode}/0", [-TORSO_RADIUS, 0, 0], "torso_growth_site"
        ))
    else:
        torso.append(xu.site_elem(
            f"torso/{torso_mode}/0", [0, 0, -TORSO_RADIUS], "torso_growth_site"
        ))

    # Step 6: Build tree structure from depth sequence
    # parent_stack[d] = (limb_index, body_element) for the most recent node at depth d
    parent_stack = {0: (-1, torso)}  # depth 0 -> torso

    for i in range(1, seq_len):
        depth = int(vec[i, 13])
        parent_depth = depth - 1
        parent_idx, parent_body = parent_stack[parent_depth]

        # Parse limb parameters
        orient_r = denormalize(float(vec[i, 0]), LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX)
        orient_r = np.clip(orient_r, LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX)
        jx_gear = denormalize(float(vec[i, 1]), JOINT_GEAR_MIN, JOINT_GEAR_MAX)
        jx_gear = np.clip(jx_gear, JOINT_GEAR_MIN, JOINT_GEAR_MAX)
        jy_gear = denormalize(float(vec[i, 2]), JOINT_GEAR_MIN, JOINT_GEAR_MAX)
        jy_gear = np.clip(jy_gear, JOINT_GEAR_MIN, JOINT_GEAR_MAX)
        limb_density = denormalize(float(vec[i, 3]), LIMB_DENSITY_MIN, LIMB_DENSITY_MAX)
        limb_density = np.clip(limb_density, LIMB_DENSITY_MIN, LIMB_DENSITY_MAX)

        theta_idx = int(round(float(vec[i, 4]))) % THETA_CLASS
        phi_idx = int(round(float(vec[i, 5]))) % PHI_CLASS
        jx_range_idx = int(round(float(vec[i, 6]))) % len(JOINT_ANGLE_LIST)
        jy_range_idx = int(round(float(vec[i, 7]))) % len(JOINT_ANGLE_LIST)

        jx_bit = int(round(float(vec[i, 8])))
        jy_bit = int(round(float(vec[i, 9])))
        attach_site_val = float(vec[i, 11])

        theta = THETA_VALUES[theta_idx]
        phi = PHI_VALUES[phi_idx]
        h = orient_r  # limb height
        r = LIMB_RADIUS

        # Determine parent radius and attachment site position
        if parent_idx == -1:
            # Attached to torso
            p_r = TORSO_RADIUS
            # Determine which torso site: growth_site at [0,0,0] or torso_growth_site
            # Use the main growth_site at [0,0,0] for torso attachments
            site_pos = [0, 0, 0]
        else:
            # Attached to a limb
            p_r = LIMB_RADIUS
            # Get the parent limb's theta, phi, r, h to compute site positions
            p_h = denormalize(float(vec[parent_stack[parent_depth][0] + 1, 0]),
                              LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX) if parent_idx >= 0 else 0
            # Look up parent's orientation from its vec row
            p_vec_row = None
            for row_idx in range(1, seq_len):
                if row_idx - 1 == parent_idx:  # limb indices are 0-based from row 1
                    p_vec_row = row_idx
                    break

            if p_vec_row is not None:
                p_theta_idx = int(round(float(vec[p_vec_row, 4]))) % THETA_CLASS
                p_phi_idx = int(round(float(vec[p_vec_row, 5]))) % PHI_CLASS
                p_theta = THETA_VALUES[p_theta_idx]
                p_phi = PHI_VALUES[p_phi_idx]
                p_h = denormalize(float(vec[p_vec_row, 0]), LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX)
                p_h = np.clip(p_h, LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX)

                # Determine attachment site type
                if attach_site_val >= 0.5:
                    # mid site: sph2cart(r + h/2, theta, phi)
                    site_pos = sph2cart(p_r + p_h / 2, p_theta, p_phi)
                else:
                    # btm site: sph2cart(r + h, theta, phi)
                    site_pos = sph2cart(p_r + p_h, p_theta, p_phi)
            else:
                site_pos = [0, 0, 0]

        # Compute limb position: site_pos + sph2cart(p_r, theta, phi)
        offset = sph2cart(p_r, theta, phi)
        pos = xu.add_list(site_pos, offset)

        limb_idx = i - 1  # 0-based limb index
        name = f"limb/{limb_idx}"
        limb = xu.body_elem(name, pos)

        # Joint position: sph2cart(r, pi+theta, pi-phi)
        theta_j = np.pi + theta
        if theta_j >= 2 * np.pi:
            theta_j -= 2 * np.pi
        joint_pos = sph2cart(r, theta_j, np.pi - phi)
        joint_pos_str = xu.arr2str(joint_pos)

        # Add joints
        joint_axes = []
        joint_gears = []
        if jx_bit:
            jx_range = JOINT_ANGLE_LIST[jx_range_idx]
            limb.append(xu.joint_elem(
                f"limbx/{limb_idx}", "hinge", "normal_joint",
                axis=xu.axis2arr("x"),
                range_=xu.arr2str(jx_range),
                pos=joint_pos_str,
            ))
            joint_axes.append("x")
            joint_gears.append(round(jx_gear, 1))
        if jy_bit:
            jy_range = JOINT_ANGLE_LIST[jy_range_idx]
            limb.append(xu.joint_elem(
                f"limby/{limb_idx}", "hinge", "normal_joint",
                axis=xu.axis2arr("y"),
                range_=xu.arr2str(jy_range),
                pos=joint_pos_str,
            ))
            joint_axes.append("y")
            joint_gears.append(round(jy_gear, 1))

        # Geom: capsule from [0,0,0] to sph2cart(r+h, theta, phi)
        x_t, y_t, z_t = sph2cart(r + h, theta, phi)
        limb.append(etree.Element("geom", {
            "name": name,
            "type": "capsule",
            "fromto": xu.arr2str([0.0, 0.0, 0.0, x_t, y_t, z_t]),
            "size": str(LIMB_RADIUS),
            "density": str(round(limb_density, 1)),
        }))

        # Mid site
        x_mid, y_mid, z_mid = sph2cart(r + h / 2, theta, phi)
        limb.append(xu.site_elem(
            f"limb/mid/{limb_idx}", [x_mid, y_mid, z_mid], "growth_site"
        ))

        # Bottom site
        x_end, y_end, z_end = sph2cart(r + h, theta, phi)
        limb.append(xu.site_elem(
            f"limb/btm/{limb_idx}", [x_end, y_end, z_end], "growth_site"
        ))

        # Bottom position site (for head height calculation)
        x_bp, y_bp, z_bp = sph2cart(2 * r + h, theta, phi)
        limb.append(xu.site_elem(
            f"limb/btm_pos/{limb_idx}", [x_bp, y_bp, z_bp], "btm_pos_site"
        ))

        # Touch sensor site
        limb.append(xu.site_elem(
            f"limb/touch/{limb_idx}", None, "touch_site",
            str(LIMB_RADIUS + 0.01),
            xu.arr2str([0.0, 0.0, 0.0, x_t, y_t, z_t]),
            "capsule",
        ))

        # Attach limb to parent
        parent_body.append(limb)

        # Add actuators
        for axis, gear in zip(joint_axes, joint_gears):
            act_name = f"limb{axis}/{limb_idx}"
            actuator_elem.append(xu.actuator_elem(act_name, gear))

        # Update parent stack
        parent_stack[depth] = (limb_idx, limb)

    # Add torso to worldbody
    worldbody.append(torso)

    # Add sensors
    # Accelerometer, gyro, velocimeter on torso
    sensor_elem.append(etree.Element("accelerometer", {"name": "torso_accel", "site": "root"}))
    sensor_elem.append(etree.Element("gyro", {"name": "torso_gyro", "site": "root"}))
    sensor_elem.append(etree.Element("velocimeter", {"name": "torso_vel", "site": "root"}))
    sensor_elem.append(etree.Element("subtreeangmom", {"name": "torso_angmom", "body": "torso/0"}))

    # Touch sensors for torso and each limb
    sensor_elem.append(etree.Element("touch", {
        "name": "torso/touch/0", "site": "torso/touch/0"
    }))
    for i in range(1, seq_len):
        limb_idx = i - 1
        sensor_elem.append(etree.Element("touch", {
            "name": f"limb/touch/{limb_idx}", "site": f"limb/touch/{limb_idx}"
        }))

    return xu.etree_to_str(root)


def vec_to_xml_safe(vec, template_path=None):
    """Safe wrapper that returns None for invalid morphologies."""
    try:
        return vec_to_xml(vec, template_path)
    except (ValueError, IndexError, KeyError) as e:
        print(f"vec_to_xml_safe: failed with {type(e).__name__}: {e}")
        return None


def vec_to_xml_file(vec, output_path, template_path=None):
    """Convert vec to XML and save to file. Returns True on success."""
    xml_str = vec_to_xml_safe(vec, template_path)
    if xml_str is None:
        return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_str)
    return True
