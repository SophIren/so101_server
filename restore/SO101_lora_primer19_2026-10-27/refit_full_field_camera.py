#!/usr/bin/env python3
"""Compare low-order lens models on held-out peripheral views, without promotion."""
import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('peripheral_dataset', type=Path)
    args = parser.parse_args()
    motion = json.loads((args.peripheral_dataset / 'motion_log.json').read_text())
    if not motion.get('success'):
        raise RuntimeError('Peripheral collection is not complete')
    entries = []
    fresh_external = []
    for path in sorted((ROOT / 'camera/calibration').glob('*/metadata.json')):
        data = json.loads(path.read_text())
        for view in data.get('views', []):
            if view.get('corners_found') and view.get('image_size') == [1280, 720]:
                if (path.parent.resolve() == args.peripheral_dataset.resolve()
                        and view.get('label', '').startswith('edge_pan_roll_')):
                    fresh_external.append((path.parent / view['file'],
                        np.asarray(view['corners'], np.float32).reshape(-1, 1, 2)))
                else:
                    entries.append((path.parent / view['file'], view))
    for path in sorted((ROOT / 'camera/calibration/manual_views').glob('*.json')):
        view = json.loads(path.read_text())
        if view.get('corners_found'):
            entries.append((path.parent / view['file'], view))
    # Repeated baseline frames should not outweigh the newly observed periphery.
    unique = []
    for path, view in entries:
        points = np.asarray(view['corners'], np.float32).reshape(-1, 1, 2)
        if any(np.sqrt(np.mean((points - prev[2]) ** 2)) < 7 for prev in unique):
            continue
        unique.append((path, view, points))
    held = []
    peripheral_index = 0
    for i, (path, view, points) in enumerate(unique):
        if args.peripheral_dataset.resolve() == path.parent.resolve():
            peripheral_index += 1
            if peripheral_index % 4 == 2:
                held.append(i)
    if len(held) < 3:
        raise RuntimeError('Need at least three independent peripheral holdouts')
    train = [i for i in range(len(unique)) if i not in held]
    external = []
    external.extend(fresh_external)
    for path in sorted((ROOT / 'camera/validation').glob('*.json')):
        view = json.loads(path.read_text())
        if view.get('corners_found'):
            external.append((path, np.asarray(view['corners'], np.float32).reshape(-1, 1, 2)))
    obj = np.zeros((104, 3), np.float32)
    obj[:, :2] = np.mgrid[0:8, 0:13].T.reshape(-1, 2) * .04

    def evaluate(K, dist, points):
        ok, r, t = cv2.solvePnP(obj, points, K, dist)
        if not ok:
            return 1e6
        predicted, _ = cv2.projectPoints(obj, r, t, K, dist)
        return float(np.sqrt(np.mean(np.sum((predicted - points) ** 2, axis=2))))

    models = []
    for name, flags in [('k1_k2_k3', 0), ('k1_k2', cv2.CALIB_FIX_K3),
                         ('k1_only', cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)]:
        rms, K, dist, _, _ = cv2.calibrateCamera(
            [obj.copy() for _ in train], [unique[i][2] for i in train],
            (1280, 720), None, None, flags=flags)
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (1280, 720), cv2.CV_32FC1)
        dx_dy, dx_dx = np.gradient(mx)
        dy_dy, dy_dx = np.gradient(my)
        det = dx_dx * dy_dy - dx_dy * dy_dx
        test = [{'file': str(unique[i][0].relative_to(ROOT)),
                 'rms_px': evaluate(K, dist, unique[i][2])} for i in held]
        old_test = [{'file': str(p.relative_to(ROOT)), 'rms_px': evaluate(K, dist, points)}
                    for p, points in external]
        held_rms = float(np.sqrt(np.mean([v['rms_px'] ** 2 for v in test])))
        monotonic = bool(np.isfinite(det).all() and det.min() > .05)
        models.append({'model': name, 'camera_matrix': K.tolist(),
                       'distortion_coefficients': dist.ravel().tolist(),
                       'training_rms_px': float(rms), 'held_out_rms_px': held_rms,
                       'held_out_views': test, 'prior_control_views': old_test,
                       'minimum_full_map_jacobian': float(det.min()),
                       'folded_pixels': int((det <= 0).sum()),
                       'accepted': monotonic and held_rms < 1.0
                           and max(v['rms_px'] for v in test) < 1.5})
    output = ROOT / 'camera/recalibration' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    valid = sorted((m for m in models if m['accepted']), key=lambda m: m['held_out_rms_px'])
    report = {'input_unique_views': len(unique), 'training_views': len(train),
              'held_out_views': len(held), 'image_size': [1280, 720],
              'square_size_m': .04, 'pattern_inner_corners': [8, 13],
              'models': models, 'selected': valid[0] if valid else None,
              'peripheral_dataset': str(args.peripheral_dataset),
              'status': 'candidate_not_promoted' if valid else 'more_data_or_model_needed'}
    (output / 'model_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT', output / 'model_comparison.json')
    for m in models:
        print(m['model'], 'train', m['training_rms_px'], 'held_out', m['held_out_rms_px'],
              'folds', m['folded_pixels'], 'accepted', m['accepted'])


if __name__ == '__main__':
    main()
