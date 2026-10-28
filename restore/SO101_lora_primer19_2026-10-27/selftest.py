#!/usr/bin/env python3
"""Offline self-test: units, arm logic, and the policy client.

    python selftest.py            # everything that needs no hardware or network

The policy client check starts a local mock server (mock_vlash_server.py) on a
free port, drives one inference through it, and shuts it down. Nothing here
opens a serial port or a camera.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = sys.executable
failures = []


def check(name, condition, detail=''):
    status = 'ok  ' if condition else 'FAIL'
    print(f'  [{status}] {name}{f" — {detail}" if detail and not condition else ""}')
    if not condition:
        failures.append(name)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_units():
    from policy_client import counts_to_lerobot, lerobot_to_counts, load_calibration

    print('\nunits')
    cal = load_calibration()
    pose = {'shoulder_pan': 2100, 'shoulder_lift': 2000, 'elbow_flex': 2000,
            'wrist_flex': 2600, 'wrist_roll': 2048, 'gripper': 3000}
    values = counts_to_lerobot(pose, cal)
    check('body joints normalised to -100..100',
          all(-100.0 <= v <= 100.0 for v in values[:5]))
    check('gripper normalised to 0..100', 0.0 <= values[5] <= 100.0)
    check('round trip is exact', lerobot_to_counts(values, cal) == pose)
    exact = True
    for name, c in cal.items():
        for edge in ('range_min', 'range_max'):
            probe = dict(pose)
            probe[name] = c[edge]
            exact &= lerobot_to_counts(counts_to_lerobot(probe, cal), cal)[name] == c[edge]
    check('limits round trip exactly', exact)


def test_arm():
    from fake_bus import FakeBus
    from policy_client import load_calibration
    from so101_arm import Arm

    print('\narm (emulated bus)')
    cal = load_calibration()
    config = json.loads((ROOT / 'robot_setup/robot_config.json').read_text())
    start = {'shoulder_pan': 2100, 'shoulder_lift': 2000, 'elbow_flex': 2000,
             'wrist_flex': 2600, 'wrist_roll': 2048, 'gripper': 3000}

    arm = Arm(calibration=cal, config=config)
    arm.bus = FakeBus(start, cal)
    check('positions read back', arm.positions() == start)
    arm.enable()
    check('torque enabled on all six', set(arm.torque_state().values()) == {1})

    goal = dict(start)
    goal['shoulder_pan'] = 2200
    moved = arm.hold_and_check(goal, tolerance=5)
    check('goal reached', moved['shoulder_pan'] == 2200, str(moved['shoulder_pan']))

    out = dict(start)
    out['shoulder_pan'] = cal['shoulder_pan']['range_max'] + 50
    try:
        arm.hold(out)
        check('out-of-range goal rejected', False)
    except ValueError:
        check('out-of-range goal rejected', True)

    arm.disable()
    check('torque disabled', set(arm.torque_state().values()) == {0})

    stuck = Arm(calibration=cal, config=config)
    stuck.bus = FakeBus(start, cal, stuck={'wrist_flex'})
    stuck.enable()
    goal = dict(start)
    goal['wrist_flex'] = 2800
    try:
        stuck.hold_and_check(goal, tolerance=5, abort_tracking=45)
        check('stalled joint aborts', False)
    except RuntimeError:
        check('stalled joint aborts', True)

    hot = Arm(calibration=cal, config=config)
    hot.bus = FakeBus(start, cal, hot={'elbow_flex'})
    hot.enable()
    try:
        hot.health()
        check('over-temperature raises', False)
    except RuntimeError as error:
        check('over-temperature raises', 'temperature' in str(error), str(error))


def test_policy_client():
    print('\npolicy client (mock server)')
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from grpc_reflection.v1alpha import reflection

    import vlash_test_pb2 as pb
    import vlash_test_pb2_grpc as pb_grpc
    from mock_vlash_server import Servicer

    port = free_port()
    servicer = Servicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_RemoteInferenceServicer_to_server(servicer, server)
    reflection.enable_server_reflection(
        (pb.DESCRIPTOR.services_by_name['RemoteInference'].full_name, reflection.SERVICE_NAME),
        server)
    bound = server.add_insecure_port(f'127.0.0.1:{port}')
    if not bound:
        check('mock server binds a port', False, str(port))
        return
    server.start()
    client = None
    try:
        from policy_client import PolicyClient, counts_to_lerobot, load_calibration

        client = PolicyClient('127.0.0.1', port, timeout=10)
        client.connect()
        check('RPC discovered via reflection', client.method_name.endswith('/Infer'),
              client.method_name)

        cal = load_calibration()
        pose = {'shoulder_pan': 2100, 'shoulder_lift': 2000, 'elbow_flex': 2000,
                'wrist_flex': 2600, 'wrist_roll': 2048, 'gripper': 3000}
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        observation = {
            'observation.state': counts_to_lerobot(pose, cal),
            'observation.images.base_0_rgb': frame,
            'observation.images.left_wrist_0_rgb': frame,
            'observation.images.right_wrist_0_rgb': frame,
        }
        actions = client.infer(observation, 'take the box')
        check('action chunk shape (n, 6)', actions.ndim == 2 and actions.shape[1] == 6,
              str(actions.shape))

        last = servicer.calls[-1]
        check('prompt delivered', last['prompt'] == 'take the box')
        check('state delivered', len(last['state']) == 6)
        check('three decodable 640x480 frames',
              all(shape == [480, 640, 3] for shape in last['images']), str(last['images']))
    finally:
        if client is not None:
            client.close()
        server.stop(0).wait(timeout=5)


if __name__ == '__main__':
    test_units()
    test_arm()
    test_policy_client()
    print()
    if failures:
        print('FAILED:', ', '.join(failures))
        raise SystemExit(1)
    print('All offline checks passed. No serial port or camera was opened.')
