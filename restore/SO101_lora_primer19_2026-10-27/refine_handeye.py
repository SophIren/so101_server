#!/usr/bin/env python3
"""Refine software-only URDF joint zeros, camera mount and fixed target pose."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fit_handeye import ROOT, forward, metrics, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--intrinsics', type=Path,
                        default=ROOT / 'camera/calibration/camera_intrinsics.json')
    args = parser.parse_args()
    result = json.loads((args.dataset / 'handeye_result.json').read_text())
    metadata = json.loads((args.dataset / 'metadata.json').read_text())
    intr = json.loads(args.intrinsics.read_text())
    K, distortion = np.asarray(intr['camera_matrix']), np.asarray(intr['distortion_coefficients'])
    cal = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
    joints = ET.parse(ROOT / 'robot_setup/so101_kinematics.urdf').getroot().findall('joint')
    objects = np.zeros((104, 3), np.float64)
    objects[:, :2] = np.mgrid[0:8, 0:13].T.reshape(-1, 2) * .04
    views = []
    for v in metadata['views']:
        if not v.get('corners_found'):
            continue
        corners = np.asarray(v['corners'], np.float32).reshape(-1, 1, 2)
        ok, r, t = cv2.solvePnP(objects, corners, K, distortion)
        if not ok:
            continue
        predicted, _ = cv2.projectPoints(objects, r, t, K, distortion)
        if float(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=2)))) <= 1:
            views.append((v, corners, transform(cv2.Rodrigues(r)[0], t)))
    train, held = result['training_indices'], result['held_out_indices']
    signs = result['solution']['signs']
    X = np.asarray(result['solution']['T_gripper_camera'])
    Y = np.asarray(result['solution']['T_base_board'])
    initial = np.r_[Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
                    Rotation.from_matrix(Y[:3, :3]).as_rotvec(), Y[:3, 3], np.zeros(3)]
    corrected = ['shoulder_lift', 'elbow_flex', 'wrist_flex']

    def decode(p):
        return (transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6]),
                transform(Rotation.from_rotvec(p[6:9]).as_matrix(), p[9:12]),
                dict(zip(corrected, p[12:])))

    def residuals(p, indices, regularize=False):
        x, y, zeros = decode(p)
        all_errors = []
        for i in indices:
            v, observed, _ = views[i]
            bg = forward(v['positions'], signs, cal, joints, zeros)
            ct = np.linalg.inv(bg @ x) @ y
            if ct[2, 3] <= .05:
                return np.full(len(indices) * 208 + (3 if regularize else 0), 1e5)
            r = Rotation.from_matrix(ct[:3, :3]).as_rotvec()
            projected, _ = cv2.projectPoints(objects, r, ct[:3, 3], K, distortion)
            all_errors.extend((projected.astype(float) - observed).ravel())
        if regularize:
            all_errors.extend(p[12:] * 2)
        return np.asarray(all_errors)

    # Keep angles near the measured midpoint. End joints' zeros are absorbed
    # into the base-board and gripper-camera transforms, avoiding gauge freedom.
    lower = np.r_[np.full(12, -np.inf), np.full(3, -np.deg2rad(20))]
    upper = np.r_[np.full(12, np.inf), np.full(3, np.deg2rad(20))]
    fitted = least_squares(lambda p: residuals(p, train, True), initial,
                           bounds=(lower, upper), diff_step=1e-4,
                           loss='soft_l1', f_scale=.5, max_nfev=200)
    x, y, zeros = decode(fitted.x)
    board_poses = [forward(v['positions'], signs, cal, joints, zeros) @ x @ ct
                   for v, _, ct in views]
    test_metrics = metrics([board_poses[i] for i in held], y)
    rms = lambda indices: float(np.sqrt(np.mean(residuals(fitted.x, indices).reshape(-1, 2).__pow__(2).sum(axis=1))))
    bounded = any(abs(v) >= np.deg2rad(19.5) for v in zeros.values())
    accepted = (fitted.success and not bounded and rms(held) < 1
                and test_metrics['translation_rms_mm'] < 5
                and test_metrics['rotation_rms_deg'] < 1)
    report = {'status': 'locally_validated' if accepted else 'not_validated',
              'dataset': str(args.dataset), 'views': len(views),
              'training_indices': train, 'held_out_indices': held,
              'training_projection_rms_px': rms(train), 'held_out_projection_rms_px': rms(held),
              'held_out_fixed_board_error': test_metrics,
              'zero_corrections_rad': zeros,
              'zero_corrections_deg': {n: float(np.rad2deg(v)) for n, v in zeros.items()},
              'T_gripper_camera': x.tolist(), 'T_base_board': y.tolist(), 'joint_signs': signs,
              'optimizer_success': bool(fitted.success), 'offset_bound_reached': bounded,
              'hardware_offsets_modified': False,
              'scope': 'Valid only near sampled poses, conditioned on supplied URDF and rigid mounts; no independent external length metrology.'}
    (args.dataset / 'handeye_refined.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
