"""VLASH wire and control-loop regression checks; no cameras or serial ports.

Run: .venv/bin/python -m unittest -v test_policy_transport
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import grpc
import numpy as np
import torch

from mock_vlash_server import TransportServicer
from policy_client import PolicyClient, _load_proto_files, lerobot_to_counts, load_calibration
from vlash_transport import decode_actions, encode_observation, validate_actions

ROOT = Path(__file__).resolve().parent


class TransportTests(unittest.TestCase):
    def setUp(self):
        # Compile the human-readable server .proto independently of the built-in descriptor.
        pool, _ = _load_proto_files([ROOT / 'proto/vlash_transport.proto'])
        self.servicer = TransportServicer()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4), options=[
            ('grpc.max_receive_message_length', 4 * 1024 * 1024)])
        self.servicer.add_to_server(self.server, pool)
        self.port = self.server.add_insecure_port('127.0.0.1:0')
        self.server.start()
        self.client = PolicyClient('127.0.0.1', self.port, timeout=10)
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[:] = [17, 80, 240]  # Distinct BGR channels expose color-order errors.
        self.observation = {'observation.state': np.arange(6, dtype=np.float32)}
        for name in ('base', 'left_wrist'):
            self.observation[f'observation.images.{name}_0_rgb'] = frame

    def tearDown(self):
        self.client.close()
        self.server.stop(0).wait(timeout=5)

    def test_no_reflection_round_trip_and_replan(self):
        self.client.connect()
        self.assertEqual(self.client.transport(), 'vlash')
        self.assertEqual(self.client.method_name, '/transport.RemoteInference/LaunchInference')
        for value in (3.0, 9.0):
            self.observation['observation.state'][:] = value
            actions = self.client.infer(self.observation, 'take the box')
            self.assertEqual(actions.shape, (32, 6))
            np.testing.assert_allclose(actions[0], value + 0.1)
        self.assertEqual(self.servicer.events,
                         ['Ready', 'LaunchInference', 'GetActionChunk',
                          'LaunchInference', 'GetActionChunk'])
        call = self.servicer.calls[-1]
        self.assertEqual(call['states'], [1, 2, 2, 3])
        self.assertTrue(all(size <= 2 * 1024 * 1024 for size in call['sizes']))
        self.assertEqual(call['batch']['task'], ['take the box'])
        self.assertEqual({k for k in call['batch'] if k.startswith('observation.images.')},
                         {'observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb'})
        for key, tensor in call['batch'].items():
            if key.startswith('observation.images.'):
                self.assertEqual(tuple(tensor.shape), (1, 3, 480, 640))
                self.assertEqual(tensor.device.type, 'cpu')
                np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([240, 80, 17]) / 255,
                                           rtol=1e-6)

    def test_explicit_proto_and_single_part(self):
        self.client.connect(proto_files=[ROOT / 'proto/vlash_transport.proto'])
        tiny = {'observation.state': np.zeros(6, np.float32),
                'observation.images.base_0_rgb': np.zeros((2, 3, 3), np.uint8)}
        self.assertEqual(self.client.infer(tiny, 'test').shape, (32, 6))
        self.assertEqual(self.servicer.calls[-1]['states'], [3])

    def test_server_error_keeps_details(self):
        self.client.connect()
        self.servicer.error = 'Inference error: bad checkpoint'
        with self.assertRaisesRegex(RuntimeError, 'GetActionChunk.*bad checkpoint'):
            self.client.infer(self.observation, 'test')

    def test_check_server_cli_without_hardware(self):
        import run_policy
        output = io.StringIO()
        with patch.object(sys, 'argv', ['run_policy.py', '--port', str(self.port),
                                       '--check-server', '--timeout', '5']), \
                patch.dict(sys.modules, {'so101_camera': None, 'so101_arm': None}), \
                redirect_stdout(output):
            run_policy.main()
        self.assertIn('VLASH Ready OK', output.getvalue())
        self.assertEqual(self.servicer.events, ['Ready'])


class PayloadTests(unittest.TestCase):
    def test_two_camera_observation_has_no_fake_third_image(self):
        from run_policy import build_observation
        cal = load_calibration()
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in cal.items()}
        base = np.full((4, 5, 3), [11, 20, 30], np.uint8)
        wrist = np.full((4, 5, 3), [40, 50, 60], np.uint8)
        observation = build_observation(pose, {'base': base, 'wrist': wrist}, cal)
        batch = pickle.loads(encode_observation(observation, 'test'))
        self.assertNotIn('observation.images.right_wrist_0_rgb', batch)
        for name, pixel in (('base', [30, 20, 11]), ('left_wrist', [60, 50, 40])):
            image = batch[f'observation.images.{name}_0_rgb']
            self.assertEqual(tuple(image.shape), (1, 3, 4, 5))
            np.testing.assert_allclose(image[0, :, 0, 0], np.array(pixel) / 255, rtol=1e-6)
        single = build_observation(pose, {'base': base, 'wrist': None}, cal)
        self.assertEqual(set(single), {'observation.state', 'observation.images.base_0_rgb'})

    def test_cuda_pickle_on_cpu(self):
        expected = torch.arange(192, dtype=torch.float32).reshape(32, 6)
        # Write real torch storage with a CUDA location tag, on this CPU-only Mac.
        with patch('torch.serialization.location_tag', return_value='cuda:0'):
            payload = pickle.dumps(expected)
        self.assertIn(b'cuda:0', payload)
        np.testing.assert_array_equal(decode_actions(payload), expected.numpy())

    def test_bad_actions_rejected_before_motor_conversion(self):
        for actions in (np.zeros((0, 6)), np.zeros((4, 7)), np.zeros(6),
                        np.full((2, 6), np.nan), np.full((2, 6), np.inf)):
            with self.subTest(shape=actions.shape):
                with self.assertRaises(ValueError):
                    validate_actions(actions)
        with self.assertRaises(ValueError):
            lerobot_to_counts([float('nan')] * 6, load_calibration())


class ControlTests(unittest.TestCase):
    def test_chunks_feed_local_arm_with_step_limits_and_replan(self):
        from fake_bus import FakeBus
        from so101_arm import Arm
        import run_policy
        cal = load_calibration()
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in cal.items()}
        arm = Arm(calibration=cal)
        arm.bus = FakeBus(pose, cal)
        class Camera:
            def read(self):
                return np.zeros((2, 3, 3), np.uint8)
        class Client:
            method_name = '/transport.RemoteInference/LaunchInference'
            calls = 0
            def infer(self, observation, prompt):
                self.calls += 1
                return np.full((3, 6), 80, np.float32)
        client = Client()
        args = argparse.Namespace(prompt='test', hz=1000, max_ticks=10, steps=5,
                                  replan=True, max_seconds=5, fill_right_wrist='base')
        with tempfile.TemporaryDirectory() as directory, patch.object(run_policy, 'ROOT', Path(directory)):
            run_policy.run_session(arm, client, {'base': Camera(), 'wrist': None}, cal, args,
                                   initial_chunk=np.full((3, 6), 70, np.float32))
            log = json.loads(next(Path(directory).glob('logs/*.json')).read_text())
        self.assertEqual(client.calls, 1)  # First chunk was reused, then replan once.
        self.assertEqual(len(log['steps']), 5)
        for step in log['steps']:
            self.assertTrue(all(abs(step['goal'][n] - step['before'][n]) <= 10 for n in cal))
        self.assertEqual(set(arm.torque_state().values()), {1})  # Stops holding position.

    def test_time_limit_after_slow_inference_prevents_movement(self):
        from fake_bus import FakeBus
        from so101_arm import Arm
        import run_policy
        cal = load_calibration()
        pose = {n: (c['range_min'] + c['range_max']) // 2 for n, c in cal.items()}
        arm = Arm(calibration=cal)
        arm.bus = FakeBus(pose, cal)
        class Camera:
            def read(self):
                return np.zeros((2, 3, 3), np.uint8)
        class Client:
            method_name = 'test'
            def infer(self, observation, prompt):
                time.sleep(.05)
                return np.full((3, 6), 80, np.float32)
        args = argparse.Namespace(prompt='test', hz=30, max_ticks=10, steps=0,
                                  replan=False, max_seconds=.02, fill_right_wrist='base')
        with tempfile.TemporaryDirectory() as directory, patch.object(run_policy, 'ROOT', Path(directory)):
            run_policy.run_session(arm, Client(), {'base': Camera()}, cal, args)
            log = json.loads(next(Path(directory).glob('logs/*.json')).read_text())
        self.assertEqual(log['steps'], [])
        self.assertEqual(arm.positions(), pose)


if __name__ == '__main__':
    unittest.main()
