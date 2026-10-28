"""Hardware-free checks of recording hooks and the complete dataset pipeline."""
from collections import deque
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pyarrow.parquet as pq

from dataset_capture import CameraStream, RawEpisode
from dataset_tools import validate_raw_episode, read_jsonl
from make_training_config import generate
from prepare_dataset import build
from record_dataset import Recorder, main as record
from teleop_loader import EXPORT, load_teleop, resolve_controller
from validate_dataset import validate


class PipelineTests(unittest.TestCase):
    def test_macos_controller_aliases_are_not_two_devices(self):
        ports = [SimpleNamespace(serial_number='5B79016829', device='/dev/cu.usbmodem5B790168291'),
                 SimpleNamespace(serial_number='5B79016829', device='/dev/tty.usbmodem5B790168291')]
        with patch('serial.tools.list_ports.comports', return_value=ports):
            self.assertEqual(resolve_controller('/nonexistent/leader', '5B79016829'),
                             '/dev/cu.usbmodem5B790168291')

    def test_recording_controls_and_alignment_exclusion(self):
        import time
        fc = json.loads((EXPORT / 'so101_follower_calibration.json').read_text())
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in fc.items()}
        class Stream:
            def snapshot(self, at):
                return at - .01, np.zeros((4, 4, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(dataset=Path(directory), task='test', no_preview=True,
                                   max_camera_skew=.1, height=4, width=4, episodes=1,
                                   episode_seconds=30)
            keys = iter(['s', 'k'])
            recorder = Recorder(args, {'base_0_rgb': Stream(), 'left_wrist_0_rgb': Stream()},
                                fc, lambda: next(keys))
            self.assertFalse(recorder.poll(False))
            now = time.monotonic()
            for i, aligning in enumerate((True, False, False)):
                recorder.sample(leader=pose, state=pose, action=pose, health={},
                                observation_time=now + i * .1, command_time=now + i * .1,
                                aligning=aligning, gripper_contact=False)
            self.assertTrue(recorder.poll(False))
            recorder.close()
            path = next((Path(directory) / 'episodes').glob('episode_*'))
            self.assertEqual(len(read_jsonl(path / 'samples.jsonl')), 2)
            self.assertEqual(json.loads((path / 'episode.json').read_text())['outcome'], 'success')

    def test_training_config_selects_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps({'repo_id': 'local/test', 'so101_synthetic': False}))
            (root / 'meta/train_episodes.json').write_text('[0, 1, 2]')
            config = root / 'lora.yaml'
            generate(root, config, '/base/so101', '/server/data', '/server/out')
            data = json.loads(config.read_text())
            self.assertEqual(data['dataset']['episodes'], [0, 1, 2])
            self.assertEqual(data['policy']['empty_cameras'], 1)
            self.assertEqual([k for k in data['policy']['input_features'] if k.startswith('observation.images.')],
                             ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
                              'observation.images.right_wrist_0_rgb'])
            self.assertEqual(data['policy']['pretrained_path'], '/base/so101')
            self.assertEqual(data['dataset']['root'], '/server/data')

    def test_complete_pipeline_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / 'raw', Path(directory) / 'lerobot'
            record(['--simulate', '--dataset', str(raw), '--episodes', '2',
                    '--episode-seconds', '1.2', '--width', '64', '--height', '48'])
            metadata = json.loads((raw / 'metadata.json').read_text())
            paths = sorted((raw / 'episodes').glob('episode_*'))
            _, rows = validate_raw_episode(paths[0], metadata)
            self.assertNotEqual(rows[0]['action'], rows[0]['observation.state'])
            failed = raw / 'episodes' / 'episode_failed'
            shutil.copytree(paths[0], failed)
            info = json.loads((failed / 'episode.json').read_text())
            info['outcome'] = 'failed'
            (failed / 'episode.json').write_text(json.dumps(info))
            build(raw, output, 'local/pipeline-check', allow_synthetic=True)
            info = validate(output)
            self.assertEqual(info['total_episodes'], 2)
            self.assertEqual(info['fps'], 10)
            self.assertEqual(info['splits'], {'train': '0:1', 'val': '1:2'})
            self.assertEqual([k for k, v in info['features'].items() if v['dtype'] == 'video'],
                             ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb'])
            table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
            self.assertEqual(len(table), 12)
            np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(12) / 10, atol=1e-6)
            report = json.loads((output / 'conversion_report.json').read_text())
            self.assertEqual(report['excluded'][0]['reason'], 'failed')
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                generate(output, Path(directory) / 'lora.yaml', '/base', '/data', '/out')
            # A modified provenance file must fail checksum verification.
            audit = output / 'audit/episode_000000.json'
            audit.write_text(audit.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                validate(output)

    def test_resume_and_timing_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw'
            args = ['--simulate', '--dataset', str(raw), '--episodes', '1',
                    '--episode-seconds', '.4', '--width', '64', '--height', '48']
            record(args)
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                record([*args, '--fps', '15'])
            path = next((raw / 'episodes').glob('episode_*'))
            rows = read_jsonl(path / 'samples.jsonl')
            rows[1]['observation_time'] = rows[0]['observation_time']
            (path / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                validate_raw_episode(path, json.loads((raw / 'metadata.json').read_text()))

    def test_writer_failure_marks_episode_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = RawEpisode(directory, 'test')
            with patch('dataset_capture.cv2.imwrite', return_value=False):
                episode.append({'observation_time': 1.0}, {'base_0_rgb': np.zeros((4, 4, 3), np.uint8)})
                episode.jobs.join()
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                episode.finish('success')
            self.assertEqual(json.loads((episode.path / 'episode.json').read_text())['status'], 'interrupted')

    def test_camera_does_not_use_future_frames(self):
        stream = CameraStream.__new__(CameraStream)
        stream.name, stream.error = 'test', None
        stream.lock = threading.Lock()
        stream.frames = deque([(1.8, np.zeros((2, 2, 3), np.uint8)),
                               (2.1, np.ones((2, 2, 3), np.uint8))])
        stamp, frame = stream.snapshot(at=2.0)
        self.assertEqual(stamp, 1.8)
        self.assertEqual(frame.max(), 0)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            stream.snapshot(at=3)

    def test_existing_teleop_hooks_capture_actual_follower_commands(self):
        from fake_bus import FakeBus
        driver = load_teleop()
        lc = json.loads((EXPORT / 'leader_setup/so101_leader_calibration.json').read_text())
        fc = json.loads((EXPORT / 'so101_follower_calibration.json').read_text())
        leader = FakeBus({n: 2047 for n in lc}, lc)
        follower = FakeBus({n: 2047 for n in fc}, fc)
        contexts = []
        for bus in (leader, follower):
            manager = MagicMock()
            manager.__enter__.return_value = bus
            contexts.append(manager)
        samples, polls = [], []
        def poll(aligning):
            polls.append(aligning)
            return len(polls) > 3
        def sample(**row):
            self.assertEqual(row['action'], follower.goals)
            self.assertTrue(all(isinstance(v, int) for v in row['action'].values()))
            samples.append(row)
        with tempfile.TemporaryDirectory() as directory, patch('serial.Serial', side_effect=contexts):
            driver.main(['--execute', '--full-range', '--seconds', '2'], on_sample=sample,
                        poll=poll, run_log_path=Path(directory) / 'teleop.json', period=.001)
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[0]['aligning'])
        self.assertFalse(samples[-1]['aligning'])
        self.assertEqual(set(follower.torque.values()), {1})
        self.assertEqual(set(leader.torque.values()), {0})


if __name__ == '__main__':
    unittest.main()
