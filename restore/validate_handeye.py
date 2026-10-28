#!/usr/bin/env python3
"""Evaluate an existing hand-eye solution on a separate, unfitted series."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from fit_handeye import ROOT, forward, metrics, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--intrinsics', type=Path,
                        default=ROOT / 'camera/calibration/camera_intrinsics.json')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT / 'robot_setup/handeye_candidate.json')
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    data = json.loads((args.dataset / 'metadata.json').read_text())
    motion = json.loads((args.dataset / 'motion_log.json').read_text())
    assert data['board_fixed_by_user'] and data['encoder_image_bracketing'] and motion['success']
    cal = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
    intr = json.loads(args.intrinsics.read_text())
    K, distortion = np.asarray(intr['camera_matrix']), np.asarray(intr['distortion_coefficients'])
    joints = ET.parse(ROOT / 'robot_setup/so101_kinematics.urdf').getroot().findall('joint')
    X, Y = np.asarray(candidate['T_gripper_camera']), np.asarray(candidate['T_base_board'])
    objects = np.zeros((104, 3), np.float64)
    objects[:, :2] = np.mgrid[0:8, 0:13].T.reshape(-1, 2) * .04
    views, board = [], []
    for v in data['views']:
        if not v.get('corners_found'):
            continue
        points = np.asarray(v['corners'], np.float64).reshape(-1, 1, 2)
        B = forward(v['positions'], candidate['joint_signs'], cal, joints,
                    candidate['zero_corrections_rad'])
        C = np.linalg.inv(B @ X) @ Y
        rotation = cv2.Rodrigues(C[:3, :3])[0]
        predicted, _ = cv2.projectPoints(objects, rotation, C[:3, 3], K, distortion)
        rms = float(np.sqrt(np.mean(np.sum((predicted - points) ** 2, axis=2))))
        ok, r, t = cv2.solvePnP(objects, points, K, distortion)
        if not ok:
            raise RuntimeError('PnP failed on independent view')
        board.append(B @ X @ transform(cv2.Rodrigues(r)[0], t))
        views.append({'label': v['label'], 'rms_px': rms, 'used_for_fit': False})
    if len(views) < 4:
        raise RuntimeError('Not enough independent views')
    pose_error = metrics(board, Y)
    rms = float(np.sqrt(np.mean([v['rms_px'] ** 2 for v in views])))
    accepted = rms < 1 and pose_error['translation_rms_mm'] < 5 and pose_error['rotation_rms_deg'] < 1
    report = {'passed': accepted, 'dataset': str(args.dataset), 'views': views,
              'projection_rms_px': rms, 'fixed_board_pose_error': pose_error,
              'candidate_was_refitted': False,
              'scope': 'Independent local-pose validation against a fixed board, not external absolute metrology.'}
    (args.dataset / 'independent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
