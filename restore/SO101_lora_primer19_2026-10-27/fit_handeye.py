#!/usr/bin/env python3
"""Fit and independently check a local eye-in-hand transform from a fixed-board series."""
import argparse
import itertools
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def transform(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = np.asarray(translation).reshape(3)
    return value


def forward(positions, signs, calibration, joints, zero_corrections=None):
    current = 'base_link'
    result = np.eye(4)
    while current != 'gripper_frame_link':
        candidates = [j for j in joints if j.find('parent').get('link') == current
                      and j.find('child').get('link') != 'moving_jaw_so101_v1_link']
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous kinematic chain at {current}')
        joint = candidates[0]
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        result = result @ transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)
        if joint.get('type') != 'fixed':
            name = joint.get('name')
            c = calibration[name]
            midpoint = (c['range_min'] + c['range_max']) / 2
            radians = signs[NAMES.index(name)] * (positions[name] - midpoint) * 2 * np.pi / 4096
            radians += (zero_corrections or {}).get(name, 0)
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            result = result @ transform(Rotation.from_rotvec(axis * radians).as_matrix(), [0, 0, 0])
        current = joint.find('child').get('link')
    return result


def metrics(base_board, reference):
    translations = [np.linalg.norm(t[:3, 3] - reference[:3, 3]) for t in base_board]
    angles = [Rotation.from_matrix(reference[:3, :3].T @ t[:3, :3]).magnitude() for t in base_board]
    return {'translation_rms_mm': float(np.sqrt(np.mean(np.square(translations))) * 1000),
            'rotation_rms_deg': float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))),
            'translation_max_mm': float(max(translations) * 1000)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--intrinsics', type=Path,
                        default=ROOT / 'camera/calibration/camera_intrinsics.json')
    args = parser.parse_args()
    data = json.loads((args.dataset / 'metadata.json').read_text())
    motion = json.loads((args.dataset / 'motion_log.json').read_text())
    if not data.get('board_fixed_by_user') or not motion.get('success'):
        raise RuntimeError('Need a completed fixed-board series')
    intrinsics = json.loads(args.intrinsics.read_text())
    K = np.asarray(intrinsics['camera_matrix'])
    distortion = np.asarray(intrinsics['distortion_coefficients'])
    calibration = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
    joints = ET.parse(ROOT / 'robot_setup/so101_kinematics.urdf').getroot().findall('joint')
    objects = np.zeros((104, 3), np.float32)
    objects[:, :2] = np.mgrid[0:8, 0:13].T.reshape(-1, 2) * .04
    views = []
    for v in data['views']:
        if not v.get('corners_found') or v['image_size'] != [1280, 720]:
            continue
        corners = np.asarray(v['corners'], np.float32).reshape(-1, 1, 2)
        ok, r, t = cv2.solvePnP(objects, corners, K, distortion)
        if not ok:
            continue
        predicted, _ = cv2.projectPoints(objects, r, t, K, distortion)
        error = float(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=2))))
        if error > 1:
            continue
        views.append((v, transform(cv2.Rodrigues(r)[0], t), error))
    if len(views) < 12:
        raise RuntimeError(f'Only {len(views)} valid synchronized views; need at least 12')
    held_out = [i for i in range(len(views)) if i % 5 == 3]
    training = [i for i in range(len(views)) if i not in held_out]
    ranking = []
    for signs in itertools.product((-1, 1), repeat=5):
        B = [forward(v['positions'], signs, calibration, joints) for v, _, _ in views]
        C = [c for _, c, _ in views]
        try:
            r, t = cv2.calibrateHandEye(
                [B[i][:3, :3] for i in training], [B[i][:3, 3] for i in training],
                [C[i][:3, :3] for i in training], [C[i][:3, 3] for i in training],
                method=cv2.CALIB_HAND_EYE_PARK)
        except cv2.error:
            continue
        X = transform(r, t)
        if not np.isfinite(X).all() or abs(np.linalg.det(r) - 1) > .01 or np.linalg.norm(t) > .5:
            continue
        board = [B[i] @ X @ C[i] for i in range(len(views))]
        reference = transform(Rotation.from_matrix(np.stack([board[i][:3, :3] for i in training])).mean().as_matrix(),
                              np.mean([board[i][:3, 3] for i in training], axis=0))
        train_metrics = metrics([board[i] for i in training], reference)
        test_metrics = metrics([board[i] for i in held_out], reference)
        score = train_metrics['translation_rms_mm'] + 2 * train_metrics['rotation_rms_deg']
        ranking.append({'signs': list(signs), 'score': score, 'train': train_metrics, 'held_out': test_metrics,
                        'T_gripper_camera': X.tolist(), 'T_base_board': reference.tolist()})
    if not ranking:
        raise RuntimeError('No physically plausible hand-eye solution; check geometry and excitation')
    ranking.sort(key=lambda x: x['score'])
    best = ranking[0]
    approved = (best['held_out']['translation_rms_mm'] < 3
                and best['held_out']['rotation_rms_deg'] < 1)
    report = {'status': 'locally_validated_candidate' if approved else 'not_validated',
              'dataset': str(args.dataset), 'views': len(views), 'training_indices': training,
              'held_out_indices': held_out, 'joint_names': NAMES, 'solution': best,
              'other_candidates': ranking[1:4],
              'assumptions': ['Camera rigid on gripper frame; target fixed in robot base frame',
                              'URDF geometry matches physical SO-101',
                              'Joint zero at measured range midpoint; 4096 counts per revolution'],
              'scope': 'Only the measured local pose range. Independent metric workspace validation remains necessary.'}
    (args.dataset / 'handeye_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('other_candidates',)}, indent=2))


if __name__ == '__main__':
    main()
