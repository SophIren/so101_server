"""Locally validated SO-101 camera geometry; does not command hardware."""
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from fit_handeye import forward

ROOT = Path(__file__).resolve().parent


def camera_to_base(positions, allow_extrapolation=False):
    """Return T_base_camera, positions in calibrated motor encoder counts."""
    calibration = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
    extrinsics = json.loads((ROOT / 'robot_setup/handeye_transform.json').read_text())
    if not extrinsics.get('independent_validation', {}).get('passed'):
        raise RuntimeError('Camera-to-robot calibration is not validated')
    for name, c in calibration.items():
        if name not in positions or not c['range_min'] <= positions[name] <= c['range_max']:
            raise ValueError(f'Joint position missing or outside calibration: {name}')
    if not allow_extrapolation:
        for name, bounds in extrinsics['validated_encoder_bounds'].items():
            if not bounds[0] <= positions[name] <= bounds[1]:
                raise ValueError(f'{name}: outside locally validated hand-eye region')
    joints = ET.parse(ROOT / 'robot_setup/so101_kinematics.urdf').getroot().findall('joint')
    base_gripper = forward(positions, extrinsics['joint_signs'], calibration,
                           joints, extrinsics['zero_corrections_rad'])
    return base_gripper @ np.asarray(extrinsics['T_gripper_camera'])


def camera_point_to_base(point_m, positions):
    """Transform a measured 3D point in camera coordinates (metres).

    A 2D pixel alone has no depth and is insufficient for this transformation.
    Accuracy was checked only near the sampled poses; this is not a motion planner.
    """
    point = np.asarray(point_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite 3D camera point in metres')
    return (camera_to_base(positions) @ np.r_[point, 1])[:3]
