"""Camera logging checks with synthetic frames and an emulated motor bus."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_log import CameraLog
from policy_client import load_calibration
import run_policy


def records(logger):
    return [json.loads(line) for line in (logger.directory / 'frames.jsonl').read_text().splitlines()]


class CameraLogTests(unittest.TestCase):
    def test_snapshots_match_images_and_actions_including_replan(self):
        from fake_bus import FakeBus
        from so101_arm import Arm
        cal = load_calibration()
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in cal.items()}
        arm = Arm(calibration=cal)
        arm.bus = FakeBus(pose, cal)
        camera = Mock()
        camera.read.return_value = np.full((8, 12, 3), [17, 80, 240], dtype=np.uint8)
        cameras = {'base': camera, 'wrist': None}
        client = Mock(method_name='test')
        client.infer.return_value = np.full((2, 6), 50, np.float32)
        args = argparse.Namespace(prompt='test', hz=1000, max_ticks=10, steps=3,
                                  replan=True, max_seconds=5, fill_right_wrist='black')
        with tempfile.TemporaryDirectory() as directory, patch.object(run_policy, 'ROOT', Path(directory)):
            logger = CameraLog(Path(directory) / 'frames')
            observation = run_policy.build_observation(pose, run_policy.grab(cameras), cal, 'black')
            logger.observation(observation, step=0)
            run_policy.run_session(arm, client, cameras, cal, args,
                                   initial_chunk=client.infer.return_value, camera_log=logger)
            log = json.loads(next(Path(directory).glob('logs/*.json')).read_text())
            frames = records(logger)
            self.assertEqual([r['stage'] for r in frames],
                             ['observation', 'execution', 'execution', 'observation', 'execution', 'final'])
            self.assertEqual(log['camera_log'], str(logger.directory))
            self.assertEqual(log['final_camera_frame'], 5)
            self.assertEqual([s['camera_frame'] for s in log['steps']], [1, 2, 4])
            self.assertEqual([s['observation_frame'] for s in log['steps']], [0, 0, 3])
            self.assertEqual([frames[i]['step'] for i in (1, 2, 4)], [0, 1, 2])
            for key in run_policy.IMAGE_KEYS:
                name = key.removeprefix('observation.images.')
                saved = cv2.imread(str(logger.directory / frames[0]['files'][name]))
                np.testing.assert_array_equal(saved, observation[key])
            self.assertEqual(set(frames[-1]['files']), {'base'})
            self.assertTrue(all(r['timestamp'] for r in frames))
            self.assertEqual(client.infer.call_count, 1)

    def test_initial_images_survive_inference_error_and_resources_close(self):
        cal = load_calibration()
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in cal.items()}
        arm = Mock()
        arm.positions.return_value = pose
        camera = Mock()
        camera.read.return_value = np.zeros((8, 12, 3), np.uint8)
        client = Mock(method_name='test', request_fields={}, response_fields={})
        client.infer.side_effect = RuntimeError('server inference failed')
        camera_module = SimpleNamespace(Camera=Mock(return_value=camera),
                                        load_config=lambda: {'camera': {}})
        arm_module = SimpleNamespace(Arm=Mock(return_value=arm))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'so101_follower_calibration.json').write_text(json.dumps(cal))
            with patch.object(run_policy, 'ROOT', root), \
                    patch.object(sys, 'argv', ['run_policy.py', '--no-wrist']), \
                    patch.dict(sys.modules, {'so101_camera': camera_module, 'so101_arm': arm_module}), \
                    patch('policy_client.PolicyClient', return_value=client):
                with self.assertRaisesRegex(RuntimeError, 'server inference failed'):
                    run_policy.main()
            manifest = next(root.glob('logs/*_frames/frames.jsonl'))
            record = json.loads(manifest.read_text())
            self.assertEqual(record['stage'], 'observation')
            self.assertEqual(set(record['files']), {'base_0_rgb'})
            for filename in record['files'].values():
                self.assertTrue((manifest.parent / filename).is_file())
            arm.close.assert_called_once()
            camera.release.assert_called_once()
            arm.enable.assert_not_called()
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
