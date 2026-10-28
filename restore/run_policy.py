#!/usr/bin/env python3
"""Drive the SO-101 follower from a natural-language prompt through the vlash server.

    python run_policy.py --probe                       # list cameras, no motion
    python run_policy.py --host 10.0.0.5 --prompt "take the box"      # dry run: no motion
    python run_policy.py --host 10.0.0.5 --prompt "take the box" --execute --supported

--execute is required for any motion, mirroring robot_control.py. The arm is
moved by absolute position goals clamped to the calibrated ranges; every step
re-checks tracking error, temperature, load and voltage.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

IMAGE_KEYS = ['observation.images.base_0_rgb',
              'observation.images.left_wrist_0_rgb',
              'observation.images.right_wrist_0_rgb']


def build_observation(state_counts, frames, calibration, fill='omit'):
    from policy_client import counts_to_lerobot

    state = counts_to_lerobot(state_counts, calibration)
    observation = {'observation.state': state}
    base = frames.get('base')
    wrist = frames.get('wrist')
    if base is None:
        raise RuntimeError('No base camera frame')
    observation[IMAGE_KEYS[0]] = base
    if fill == 'omit':
        if wrist is not None:
            observation[IMAGE_KEYS[1]] = wrist
        return observation
    # Explicit legacy mode for servers that still require all three images.
    if wrist is None:
        wrist = base
    right = {'black': np.zeros_like(base),
             'wrist': wrist,
             'base': base}[fill]
    observation[IMAGE_KEYS[1]] = wrist
    observation[IMAGE_KEYS[2]] = right
    return observation


def grab(cameras):
    return {name: (camera.read() if camera is not None else None)
            for name, camera in cameras.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1', help='vlash server host')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--prompt', default='take the box')
    parser.add_argument('--method', help='override RPC; VLASH LaunchInference is selected automatically')
    parser.add_argument('--proto', nargs='+', metavar='FILE',
                        help='override schema with server .proto file(s); VLASH schema is built in')
    parser.add_argument('--proto-path', nargs='+', metavar='DIR',
                        help='extra include directories for --proto imports')
    parser.add_argument('--descriptor-set', nargs='+', metavar='FILE',
                        help='protoc --descriptor_set_out file(s), alternative to --proto')
    parser.add_argument('--dump-descriptor', action='store_true',
                        help='print the server schema and exit')
    parser.add_argument('--check-server', action='store_true',
                        help='connect and check VLASH Ready without opening cameras or serial')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='gRPC timeout in seconds per call, including inference')
    parser.add_argument('--probe', action='store_true', help='list cameras and exit')

    parser.add_argument('--base-camera', type=int, default=0, help='index of the scene camera')
    parser.add_argument('--wrist-camera', type=int, default=1, help='index of the arm-mounted camera')
    parser.add_argument('--no-wrist', action='store_true')
    parser.add_argument('--fill-right-wrist', choices=['omit', 'base', 'wrist', 'black'], default='omit',
                        help='omit missing slots by default; server must mask them (empty_cameras: 1 for two cameras)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--rectify', action='store_true',
                        help='undistort 1280x720 frames (needs --width 1280 --height 720)')

    parser.add_argument('--execute', action='store_true', help='enable motors and move')
    parser.add_argument('--supported', action='store_true',
                        help='arm is on a stable support and the workspace is clear')
    parser.add_argument('--port-name', help='override the serial port')
    parser.add_argument('--hz', type=float, default=15.0, help='control loop rate')
    parser.add_argument('--max-ticks', type=int, default=60,
                        help='max encoder counts a joint may move per step')
    parser.add_argument('--steps', type=int, default=0, help='0 = one full action chunk')
    parser.add_argument('--replan', action='store_true',
                        help='re-query the policy after each chunk instead of stopping')
    parser.add_argument('--max-seconds', type=float, default=30.0)
    parser.add_argument('--save-frames', help='directory for the frames sent to the policy')
    parser.add_argument('--no-camera-log', action='store_true',
                        help='disable automatic camera snapshots (enabled by default)')
    args = parser.parse_args()
    if args.execute and not args.supported:
        parser.error('--supported is required: the arm must rest on a stable support')
    if (args.hz <= 0 or args.max_ticks <= 0 or args.max_seconds <= 0
            or args.timeout <= 0 or args.steps < 0):
        parser.error('hz, max-ticks, max-seconds and timeout must be positive; steps must be >= 0')

    if args.probe:
        from so101_camera import Camera, probe_cameras, load_config
        config = load_config()
        found = probe_cameras(4)
        working = [(i, w, h) for i, w, h, error in found if error is None]
        for index, width, height, error in found:
            print(f'camera {index}: ' + (f'{width}x{height}' if error is None else error))
        if working:
            # A Mac often exposes a virtual device (Continuity Camera, Desk View)
            # next to the real ones, so the index alone does not identify a camera.
            import cv2
            directory = Path(args.save_frames or 'logs/cameras')
            directory.mkdir(parents=True, exist_ok=True)
            for index, _, _ in working:
                camera = Camera(index, config['camera'], 640, 480, 30)
                try:
                    frame = camera.read()
                finally:
                    camera.release()
                path = directory / f'camera_{index}.png'
                cv2.imwrite(str(path), frame)
                print(f'  saved {path}')
            print('Open those files: the arm camera looks down at the gripper, '
                  'the scene camera sees the table. Pass the indices with '
                  '--base-camera / --wrist-camera.')
        if not working:
            print('No camera delivered a frame. Grant camera access to the terminal '
                  '(System Settings -> Privacy & Security -> Camera), then retry.')
            print('While tclaude is sandboxed it cannot open the camera at all; '
                  'run this command from a normal terminal.')
        return

    from policy_client import PolicyClient

    client = PolicyClient(args.host, args.port, timeout=args.timeout, method=args.method)
    cameras = {}
    arm = None
    camera_log = None
    try:
        client.connect(proto_files=args.proto, proto_path=args.proto_path,
                       descriptor_set=args.descriptor_set)
        print(f'RPC {client.method_name}')
        print(f'request fields: {sorted(client.request_fields)}')
        print(f'response fields: {sorted(client.response_fields)}')
        if args.check_server:
            if client.transport() != 'vlash':
                raise RuntimeError('--check-server requires the VLASH RemoteInference service')
            print(f'VLASH Ready OK at {client.address}')
            return
        if args.dump_descriptor:
            from policy_client import describe
            print(describe(client.methods, client.pool))
            return

        from so101_camera import Camera, load_config
        calibration = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
        config = load_config()

        def open_or_warn(index, label):
            try:
                return Camera(index, config['camera'], args.width, args.height,
                              int(args.fps), rectify=args.rectify)
            except Exception as error:
                print(f'{label} camera {index} unavailable: {error}')
                return None

        cameras['base'] = open_or_warn(args.base_camera, 'base')
        if cameras['base'] is None:
            raise RuntimeError('The scene camera is required; run --probe to find its index')
        if not args.no_wrist:
            cameras['wrist'] = open_or_warn(args.wrist_camera, 'wrist')
            if cameras['wrist'] is None:
                raise RuntimeError('The wrist camera is required for two-camera inference; '
                                   'check --wrist-camera, or use --no-wrist with a one-camera server config')

        from so101_arm import Arm
        from policy_client import counts_to_lerobot

        arm = Arm(port=args.port_name, calibration=calibration, config=config)
        arm.open()
        if not args.no_camera_log:
            from camera_log import CameraLog
            session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            camera_log = CameraLog(ROOT / 'logs' / f'policy_{session_id}_frames', {
                'prompt': args.prompt, 'base_camera': args.base_camera,
                'wrist_camera': None if args.no_wrist else args.wrist_camera,
                'fill_right_wrist': args.fill_right_wrist,
                'execute': args.execute,
                'description': 'observation: exact policy input images; execution/final: physical camera snapshots',
            })
        frames = grab(cameras)
        state = arm.positions()
        print('positions', state)
        print('lerobot units', np.round(counts_to_lerobot(state, calibration), 2))

        observation = build_observation(state, frames, calibration, args.fill_right_wrist)
        if camera_log is not None:
            camera_log.observation(observation, step=0)
        started = time.monotonic()
        actions = client.infer(observation, args.prompt)
        print(f'first inference: {actions.shape} in {time.monotonic() - started:.2f}s')
        print('first action', np.round(actions[0], 2))
        print('last action ', np.round(actions[-1], 2))

        if args.save_frames:
            import cv2
            directory = Path(args.save_frames)
            directory.mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                if frame is not None:
                    cv2.imwrite(str(directory / f'{name}.png'), frame)
            print('frames saved to', directory)

        if not args.execute:
            print('DRY RUN: no motion. Add --execute --supported to move the arm.')
            return
        run_session(arm, client, cameras, calibration, args, initial_chunk=actions,
                    camera_log=camera_log)
    finally:
        for camera in cameras.values():
            if camera is not None:
                camera.release()
        if arm is not None:
            arm.close()
        client.close()


def run_session(arm, client, cameras, calibration, args, initial_chunk=None, camera_log=None):
    from policy_client import lerobot_to_counts

    log = {'started': datetime.datetime.now().isoformat(), 'prompt': args.prompt,
           'rpc': client.method_name, 'steps': []}
    if camera_log is not None:
        log['camera_log'] = str(camera_log.directory)
    log_path = ROOT / 'logs' / ('policy_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    log_path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    period = 1.0 / args.hz
    enabled = False
    try:
        start = arm.positions()
        log['start'] = start
        for name, c in calibration.items():
            if not c['range_min'] <= start[name] <= c['range_max']:
                raise RuntimeError(f'{name} starts outside its calibrated range')
        # If enable fails after powering some joints, still try to hold them.
        enabled = True
        arm.enable()
        current = arm.positions()

        executed = 0
        while True:
            if args.steps and executed >= args.steps:
                break
            if time.monotonic() - started > args.max_seconds:
                print('time limit reached')
                break
            if initial_chunk is not None:
                chunk, initial_chunk = initial_chunk, None
            else:
                current = arm.positions()
                frames = grab(cameras)
                observation = build_observation(current, frames, calibration, args.fill_right_wrist)
                if camera_log is not None:
                    camera_log.observation(observation, step=executed)
                chunk = client.infer(observation, args.prompt)
            for step, action in enumerate(chunk):
                if time.monotonic() - started >= args.max_seconds:
                    print('time limit reached')
                    break
                step_started = time.monotonic()
                current = arm.positions()
                goal = lerobot_to_counts(action, calibration)
                smoothed = {}
                for name in calibration:
                    delta = goal[name] - current[name]
                    smooth = int(np.clip(delta, -args.max_ticks, args.max_ticks))
                    smoothed[name] = current[name] + smooth
                actual = arm.hold_and_check(smoothed)
                health = arm.health()
                for name, row in health.items():
                    if row['temp'] >= 45 or row['load'] >= 850 or not 105 <= row['voltage'] <= 135:
                        raise RuntimeError(f'{name}: unsafe telemetry {row}')
                log['steps'].append({'step': executed, 'before': current, 'goal': smoothed,
                                     'actual': actual, 'health': health})
                if camera_log is not None:
                    entry = log['steps'][-1]
                    entry['observation_frame'] = camera_log.last_observation
                    entry['camera_frame'] = camera_log.save(
                        grab(cameras), 'execution', step=executed, state=actual)
                current = actual
                executed += 1
                if args.steps and executed >= args.steps:
                    break
                remaining = args.max_seconds - (time.monotonic() - started)
                time.sleep(max(0.0, min(period - (time.monotonic() - step_started), remaining)))
            if chunk.shape[0] == 0:
                print('policy returned an empty chunk; stopping')
                break
            if not args.replan:
                break
            print(f'{executed} steps, replanning')
        log['completed'] = True
    except BaseException as error:
        log['error'] = str(error)
        print('STOP:', error, flush=True)
        raise
    finally:
        if enabled:
            # Keep holding: releasing torque here would drop a raised arm.
            try:
                held = arm.positions()
                arm.hold(held)
                print('HOLDING at', held)
                print('The arm is still powered. Support it, then run:')
                print('    python disable_torque.py --supported')
            except Exception as error:
                print('could not hold:', error)
        if camera_log is not None:
            try:
                log['final_camera_frame'] = camera_log.save(grab(cameras), 'final')
            except Exception as error:
                log['camera_log_error'] = str(error)
                print('could not save final camera frame:', error)
        log['finished'] = datetime.datetime.now().isoformat()
        log_path.write_text(json.dumps(log, indent=2) + '\n')
        print('log:', log_path)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
