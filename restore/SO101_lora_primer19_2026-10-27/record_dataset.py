#!/usr/bin/env python3
"""Record two-camera demonstrations using the calibrated, supplied teleop."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
from pathlib import Path
import select
import sys
import threading
import time

import cv2
import numpy as np

from dataset_capture import CameraStream, RawEpisode, write_json
from policy_client import JOINT_ORDER, counts_to_lerobot
from teleop_loader import EXPORT, load_teleop, resolve_controller

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb')


@contextmanager
def terminal_keys():
    if not sys.stdin.isatty():
        raise RuntimeError('Recording requires an interactive terminal for s/k/f/d/q controls')
    import termios
    import tty
    previous = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ''
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)


def dataset_metadata(args, lc, fc):
    return {'format': 'so101-raw-v1', 'configuration': {
        'synthetic': args.simulate, 'fps': args.fps,
        'cameras': {CAMERAS[0]: args.base_camera, CAMERAS[1]: args.wrist_camera},
        'image_size': [args.width, args.height],
        'camera_processing': {'rectified': False, 'stored_format': 'lossless PNG',
                              'decoded_training_color': 'RGB'},
        'joint_order': JOINT_ORDER, 'leader_calibration': lc, 'follower_calibration': fc,
        'units': 'arm joints calibrated RANGE_M100_100; gripper calibrated RANGE_0_100',
        'action_convention': 'absolute rounded goals actually sent to follower, in follower calibration',
        'speed_counts_per_second': args.speed,
        'teleop_sha256': hashlib.sha256((EXPORT / 'teleop_so101.py').read_bytes()).hexdigest(),
    }}


def ensure_dataset(directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'metadata.json'
    if path.exists():
        if json.loads(path.read_text())['configuration'] != metadata['configuration']:
            raise RuntimeError('Dataset configuration/calibration differs from the existing recording. '
                               'Use a new --dataset directory; do not mix camera layouts or units.')
    else:
        metadata['created'] = datetime.datetime.now().astimezone().isoformat()
        write_json(path, metadata)


class Recorder:
    def __init__(self, args, streams, calibration, read_key):
        self.args, self.streams, self.calibration, self.read_key = args, streams, calibration, read_key
        self.episode = None
        self.finished = []
        self.errors = []
        self.successes = 0
        self.preview = not args.no_preview
        self.aligned_announced = False

    def show_preview(self, aligning):
        for name, stream in self.streams.items():
            _, frame = stream.snapshot()
            label = 'ALIGNING' if aligning else ('RECORDING' if self.episode else 'IDLE')
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow(f'Dataset: {name}', frame)
        return cv2.waitKey(1)

    def finish_episode(self, outcome, error=None):
        episode, self.episode = self.episode, None
        if episode is None:
            return
        def finish():
            try:
                episode.finish(outcome, error)
            except Exception as problem:
                self.errors.append(problem)
        worker = threading.Thread(target=finish, name='episode-finalize')
        worker.start()
        self.finished.append(worker)
        if outcome == 'success' and error is None:
            self.successes += 1

    def poll(self, aligning):
        if self.errors:
            raise RuntimeError(f'Dataset writer failed: {self.errors[0]}') from self.errors[0]
        if self.episode and self.episode.error:
            raise RuntimeError(f'Dataset writer failed: {self.episode.error}')
        if not aligning and not self.aligned_announced:
            self.aligned_announced = True
            print('\nREADY. s=start, k=save successful episode, f=keep failed episode, '
                  'd=discard, q=stop and hold. Reset the scene between episodes.', flush=True)
        key = self.read_key().lower()
        if self.preview:
            code = self.show_preview(aligning)
            if code != -1:
                key = chr(code & 255).lower()
        if key in ('q', '\x1b'):
            return True
        if key == 's' and self.episode is None:
            if aligning:
                print('Wait for POSE ALIGNED before recording.', flush=True)
            else:
                self.episode = RawEpisode(self.args.dataset, self.args.task)
                print('RECORDING. Perform the entire task; k saves success.', flush=True)
        if key in ('k', 'f', 'd') and self.episode:
            self.finish_episode({'k': 'success', 'f': 'failed', 'd': 'discard'}[key])
        if self.episode and time.monotonic() - self.episode.started >= self.args.episode_seconds:
            self.finish_episode('failed')
            print('Episode time limit: kept as failed, excluded from training by default.', flush=True)
        return bool(self.args.episodes and self.successes >= self.args.episodes)

    def sample(self, *, leader, state, action, health, observation_time, command_time,
               aligning, gripper_contact):
        if self.episode is None or aligning:
            return
        snapshots = {name: stream.snapshot(observation_time) for name, stream in self.streams.items()}
        times = [stamp for stamp, _ in snapshots.values()]
        if max(times) - min(times) > self.args.max_camera_skew:
            raise RuntimeError('Camera timestamps are too far apart; episode interrupted')
        frames = {name: frame for name, (_, frame) in snapshots.items()}
        for name, frame in frames.items():
            if frame.shape != (self.args.height, self.args.width, 3):
                raise RuntimeError(f'{name}: expected {self.args.width}x{self.args.height}, got {frame.shape}')
        self.episode.append({
            'observation_time': observation_time, 'command_time': command_time,
            'wall_time': time.time(), 'camera_times': {k: t for k, (t, _) in snapshots.items()},
            'observation.state': counts_to_lerobot(state, self.calibration).tolist(),
            'action': counts_to_lerobot(action, self.calibration).tolist(),
            'follower_counts': state, 'target_counts': action, 'leader_counts': leader,
            'health': health, 'gripper_contact': gripper_contact,
        }, frames)

    def close(self, error=None):
        self.finish_episode('interrupted', error or 'Session stopped with an unfinished episode')
        for worker in self.finished:
            worker.join()
        if self.preview:
            cv2.destroyAllWindows()
        if self.errors:
            raise RuntimeError(f'Dataset writer failed: {self.errors[0]}') from self.errors[0]


def simulate(args, calibration):
    """Produce an explicitly synthetic dataset to test the entire file pipeline."""
    count = args.episodes or 2
    n = max(3, int(args.episode_seconds * args.fps))
    for episode_index in range(count):
        episode = RawEpisode(args.dataset, args.task)
        start = time.monotonic()
        for index in range(n):
            pose = {name: (c['range_min'] + c['range_max']) // 2 for name, c in calibration.items()}
            pose['shoulder_pan'] += int(20 * np.sin(index / 4))
            action = dict(pose, shoulder_pan=pose['shoulder_pan'] + 4)
            frames = {}
            for slot_index, name in enumerate(CAMERAS):
                frame = np.zeros((args.height, args.width, 3), np.uint8)
                frame[:] = (20 + 20 * slot_index, 50, 100)
                cv2.rectangle(frame, (index % max(1, args.width - 30), 10),
                              (min(args.width - 1, index % max(1, args.width - 30) + 20), 30),
                              (0, 220, 0), -1)
                frames[name] = frame
            stamp = start + index / args.fps
            episode.append({'observation_time': stamp, 'command_time': stamp,
                            'wall_time': time.time(), 'camera_times': {k: stamp for k in CAMERAS},
                            'observation.state': counts_to_lerobot(pose, calibration).tolist(),
                            'action': counts_to_lerobot(action, calibration).tolist(),
                            'follower_counts': pose, 'target_counts': action, 'leader_counts': pose,
                            'health': {}, 'gripper_contact': False}, frames)
            episode.jobs.join()
        episode.finish('success')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/box_raw'))
    parser.add_argument('--task', default='Pick up the box and place it in the marked area.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--supported', action='store_true')
    parser.add_argument('--resume-holding', action='store_true')
    parser.add_argument('--leader-port')
    parser.add_argument('--follower-port')
    parser.add_argument('--base-camera', type=int, default=1)
    parser.add_argument('--wrist-camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=10, help='Teleop sampling rate; exported working default is 10 Hz')
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--max-camera-skew', type=float, default=.1)
    parser.add_argument('--speed', type=int, default=400)
    parser.add_argument('--session-seconds', type=float, default=300)
    parser.add_argument('--episode-seconds', type=float, default=30)
    parser.add_argument('--episodes', type=int, default=0, help='Stop after this many marked successes; 0=manual')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--simulate', action='store_true', help='File pipeline check only; no serial ports or cameras')
    args = parser.parse_args(argv)
    if not args.task.strip():
        parser.error('task must describe the demonstration')
    if not (1 <= args.fps <= 30 and 0 < args.session_seconds <= 300 and args.episode_seconds > 0
            and args.width > 0 and args.height > 0 and args.max_camera_skew > 0
            and 65 <= args.speed <= 600 and args.episodes >= 0):
        parser.error('Check fps (1..30), session duration (1..300), image size, speed (65..600) and episode limits')
    if args.base_camera == args.wrist_camera:
        parser.error('base and wrist must be different physical cameras')
    if args.execute and not args.supported:
        parser.error('Support the follower and clear its workspace, then pass --supported')
    lc = json.loads((EXPORT / 'leader_setup/so101_leader_calibration.json').read_text())
    fc = json.loads((EXPORT / 'so101_follower_calibration.json').read_text())
    if args.simulate:
        ensure_dataset(args.dataset, dataset_metadata(args, lc, fc))
        simulate(args, fc)
        print('SYNTHETIC DATA ONLY. Do not use these episodes to train a real task.')
        return
    driver = load_teleop()
    leader_port = resolve_controller(driver.CONFIG['leader_port'], '5B79016829', args.leader_port)
    follower_port = resolve_controller(driver.CONFIG['follower_port'], '5B8E114736', args.follower_port)
    if Path(leader_port).resolve() == Path(follower_port).resolve():
        parser.error('Leader and follower must use different serial ports')
    teleop_args = ['--leader-port', leader_port, '--follower-port', follower_port,
                  '--seconds', str(args.session_seconds), '--speed', str(args.speed), '--full-range']
    if args.resume_holding:
        teleop_args.append('--resume-holding')
    if not args.execute:
        driver.main(teleop_args)
        print('Read-only teleop preflight complete. Recording needs --execute --supported.')
        return
    ensure_dataset(args.dataset, dataset_metadata(args, lc, fc))
    from so101_camera import Camera, load_config
    config = load_config()
    streams = {}
    recorder = None
    session_error = None
    try:
        for name, index in zip(CAMERAS, (args.base_camera, args.wrist_camera)):
            streams[name] = CameraStream(Camera(index, config['camera'], args.width,
                                                args.height, args.camera_fps), name)
        deadline = time.monotonic() + 5
        while True:
            try:
                for stream in streams.values():
                    stream.snapshot()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)
        with terminal_keys() as read_key:
            recorder = Recorder(args, streams, fc, read_key)
            if recorder.preview:
                # Window creation can be slow on macOS; do it before enabling motors.
                recorder.show_preview(aligning=True)
            session = args.dataset / 'sessions' / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
            driver.main([*teleop_args, '--execute'], on_sample=recorder.sample, poll=recorder.poll,
                        run_log_path=session, period=1 / args.fps)
    except BaseException as error:
        session_error = repr(error)
        raise
    finally:
        try:
            if recorder:
                recorder.close(session_error)
        finally:
            for stream in streams.values():
                stream.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
