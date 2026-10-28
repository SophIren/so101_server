"""Shared checks, resampling, statistics and video encoding for LeRobot v2.1."""
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from policy_client import counts_to_lerobot

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb')
VIDEO_KEYS = tuple(f'observation.images.{name}' for name in CAMERAS)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ffmpeg_executable():
    installed = shutil.which('ffmpeg')
    if installed:
        return installed
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def validate_raw_episode(path, metadata, *, max_gap=.35, max_age=.3, max_skew=.1):
    path = Path(path)
    info = json.loads((path / 'episode.json').read_text())
    if not isinstance(info.get('task'), str) or not info['task'].strip():
        raise ValueError(f'{path.name}: task instruction is empty')
    rows = read_jsonl(path / 'samples.jsonl')
    if info['status'] != 'complete' or info['samples'] != len(rows) or len(rows) < 2:
        raise ValueError(f'{path.name}: incomplete episode or fewer than two samples')
    calibration = metadata['configuration']['follower_calibration']
    times = np.asarray([r['observation_time'] for r in rows], dtype=np.float64)
    intervals = np.diff(times)
    if not np.isfinite(times).all() or (intervals <= 0).any() or intervals.max() > max_gap:
        raise ValueError(f'{path.name}: nonmonotonic timestamps or control gap > {max_gap}s')
    width, height = metadata['configuration']['image_size']
    for index, row in enumerate(rows):
        if set(row['images']) != set(CAMERAS):
            raise ValueError(f'{path.name}: expected exactly the base and wrist cameras')
        if row['sample_index'] != index:
            raise ValueError(f'{path.name}: noncontiguous sample index {index}')
        for key, counts in (('observation.state', 'follower_counts'), ('action', 'target_counts')):
            vector = np.asarray(row[key], dtype=np.float32)
            if vector.shape != (6,) or not np.isfinite(vector).all():
                raise ValueError(f'{path.name}: invalid {key} at {index}')
            if not np.allclose(vector, counts_to_lerobot(row[counts], calibration), atol=1e-5):
                raise ValueError(f'{path.name}: {key} is not in the follower calibration frame')
        for name, value in row['target_counts'].items():
            c = calibration[name]
            if not isinstance(value, int) or not c['range_min'] <= value <= c['range_max']:
                raise ValueError(f'{path.name}: invalid rounded follower goal {name}={value}')
        stamps = [row['camera_times'][k] for k in CAMERAS]
        ages = [row['observation_time'] - t for t in stamps]
        if not all(math.isfinite(t) for t in stamps) or min(ages) < -1e-6 or max(ages) > max_age:
            raise ValueError(f'{path.name}: stale/future camera frame at {index}')
        if max(stamps) - min(stamps) > max_skew:
            raise ValueError(f'{path.name}: camera skew at {index}')
        if not row['observation_time'] <= row['command_time'] < row['observation_time'] + max_gap:
            raise ValueError(f'{path.name}: command timing invalid at {index}')
        for key in CAMERAS:
            filename = (path / row['images'][key]).resolve()
            if not filename.is_relative_to(path.resolve()):
                raise ValueError(f'{path.name}: image outside the episode directory')
            frame = cv2.imread(str(filename))
            if frame is None or frame.shape != (height, width, 3):
                raise ValueError(f'{path.name}: unreadable/wrong-sized image {filename}')
    return info, rows


def resample_indices(rows, fps):
    times = np.asarray([row['observation_time'] for row in rows], dtype=np.float64)
    relative = times - times[0]
    grid = np.arange(int(np.floor(relative[-1] * fps + 1e-7)) + 1, dtype=np.float64) / fps
    right = np.minimum(np.searchsorted(relative, grid), len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(relative[left] - grid) <= abs(relative[right] - grid), left, right)
    return indices, grid, relative[indices] - grid


class Moments:
    def __init__(self):
        self.count = 0
        self.total = self.squared = self.minimum = self.maximum = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        total, squared = values.sum(axis=0), np.square(values).sum(axis=0)
        low, high = values.min(axis=0), values.max(axis=0)
        if self.count == 0:
            self.total, self.squared, self.minimum, self.maximum = total, squared, low, high
        else:
            self.total += total
            self.squared += squared
            self.minimum = np.minimum(self.minimum, low)
            self.maximum = np.maximum(self.maximum, high)
        self.count += len(values)

    def result(self, image=False):
        mean = self.total / self.count
        std = np.sqrt(np.maximum(0, self.squared / self.count - mean ** 2))
        values = {'min': self.minimum, 'max': self.maximum, 'mean': mean,
                  'std': np.maximum(std, 1e-6)}
        return {**{k: (v.reshape(3, 1, 1) if image else v).tolist() for k, v in values.items()},
                'count': [self.count]}


def encode_video(path, raw_path, rows, indices, camera, fps, size, statistics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    command = [ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-bf', '0',
               '-movflags', '+faststart', str(path)]
    errors = path.with_suffix('.ffmpeg.log')
    try:
        with errors.open('w+b') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
            try:
                for index in indices:
                    frame = cv2.imread(str(Path(raw_path) / rows[int(index)]['images'][camera]))
                    if frame is None or frame.shape != (height, width, 3):
                        raise ValueError(f'Bad frame in {raw_path} for {camera}, index {index}')
                    if statistics is not None:
                        # Image stats are sampled; VISUAL normalization is IDENTITY.
                        rgb = cv2.resize(frame, (32, 32))[:, :, ::-1].astype(np.float64) / 255
                        statistics.add(rgb.reshape(-1, 3))
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    stderr.seek(0)
                    raise RuntimeError(stderr.read().decode(errors='replace'))
            except BaseException:
                process.kill()
                process.wait()
                raise
    finally:
        if path.exists() and path.stat().st_size > 0 and errors.exists() and errors.stat().st_size == 0:
            errors.unlink()


def verify_video(path, expected_frames, fps, size):
    """Decode every frame, not just the MP4 header."""
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        if abs(video.get(cv2.CAP_PROP_FPS) - fps) > 1e-3:
            raise ValueError(f'Incorrect video FPS: {path}')
        count = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame.shape != (size[1], size[0], 3):
                raise ValueError(f'Incorrect video dimensions: {path}')
            count += 1
        if count != expected_frames:
            raise ValueError(f'{path}: decoded {count} frames, expected {expected_frames}')
    finally:
        video.release()
