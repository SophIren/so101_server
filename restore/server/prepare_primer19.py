#!/usr/bin/env python3
"""Convert the pinned PriMeR19 bag recording to the client's SO101 units and slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq

from dataset_tools import Moments, VIDEO_KEYS
from policy_client import JOINT_ORDER
from prepare_dataset import write_parquet, write_json, write_jsonl

REPO = 'PriMeR19/SO-101-teleop'
REVISION = 'b83962819ea1b10eb3c690b04d2094058fbbf244'
CAMERAS = dict(zip(('observation.images.front', 'observation.images.wrist'), VIDEO_KEYS))


def convert_units(data, calibration):
    raw = np.asarray(data['observation.follower.raw_position'], dtype=np.float64)
    degrees = np.asarray(data['observation.state'], dtype=np.float64)
    sent = np.asarray(data['action.goal_sent'], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or degrees.shape != raw.shape or sent.shape != raw.shape:
        raise ValueError('Malformed six-joint state/action')
    if not all(np.isfinite(x).all() for x in (raw, degrees, sent)):
        raise ValueError('Nonfinite source state/action')
    if not np.allclose(degrees[:, :5], (raw[:, :5] - 2048) * 360 / 4096, atol=2e-5):
        raise ValueError('Recorded degree units do not match follower encoders')
    mins = np.array([calibration[name]['range_min'] for name in JOINT_ORDER])
    spans = np.array([calibration[name]['range_max'] - calibration[name]['range_min'] for name in JOINT_ORDER])
    if (spans <= 0).any():
        raise ValueError('Invalid follower calibration spans')
    factors = np.array([200] * 5 + [100])
    offsets = np.array([-100] * 5 + [0])
    state = (raw - mins) / spans * factors + offsets
    if not np.allclose(state[:, 5], degrees[:, 5], atol=2e-5):
        raise ValueError('Recorded gripper percent does not match follower calibration')
    target_counts = sent[:, :5] * 4096 / 360 + 2048
    action = np.column_stack(((target_counts - mins[:5]) / spans[:5] * 200 - 100, sent[:, 5]))
    return state.astype(np.float32), action.astype(np.float32)


def transcode(job):
    source, target, n, fps = job
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                             '-threads', '2', '-i', str(source), '-an', '-c:v', 'libx264',
                             '-threads', '2', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                             '-bf', '0', '-vsync', '0', '-movflags', '+faststart', str(target)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{source}: {result.stderr[-2000:]}')
    # Full decoding/count/shape/FPS checks follow in validate_dataset.
    return target


def build(source, destination, workers=6, limit_episodes=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Output exists: {destination}')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['robot_type'] != 'so101_follower':
        raise ValueError('Unexpected source dataset schema')
    names = [name + '.pos' for name in JOINT_ORDER]
    for key in ('observation.state', 'action.goal_sent'):
        if info['features'][key]['names'] != names:
            raise ValueError(f'Unexpected joint order: {key}')
    follower = json.loads((source / 'robot_calibration.json').read_text())['follower']
    metadata = [row for path in sorted((source / 'meta/episodes').rglob('*.parquet'))
                for row in pq.read_table(path).to_pylist()]
    metadata.sort(key=lambda row: row['episode_index'])
    if [row['episode_index'] for row in metadata] != list(range(info['total_episodes'])):
        raise ValueError('Noncontiguous source episodes')
    tasks = pq.read_table(source / 'meta/tasks.parquet').to_pylist()
    if tasks != [{'task_index': 0, 'task': 'pick_place_bag'}]:
        raise ValueError(f'Unexpected task instructions: {tasks}')
    random.Random(1000).shuffle(metadata)
    if limit_episodes is not None:
        if limit_episodes < 2:
            raise ValueError('At least two episodes are needed for train/validation')
        metadata = metadata[:limit_episodes]
    n_train = len(metadata) - max(1, round(len(metadata) * .1))
    fps = info['fps']
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    (stage / 'meta').mkdir()
    (stage / 'audit').mkdir()
    summaries, episode_stats, provenance, jobs = [], [], [], []
    totals = {key: Moments() for key in ('observation.state', 'action')}
    offset = 0
    try:
        for episode, row in enumerate(metadata):
            original = row['episode_index']
            n = row['length']
            path = source / info['data_path'].format(chunk_index=row['data/chunk_index'],
                                                   file_index=row['data/file_index'])
            original_table = pq.read_table(path)
            data = original_table.to_pydict()
            if (len(original_table) != n or set(data['episode_index']) != {original}
                    or data['frame_index'] != list(range(n))
                    or not np.allclose(data['timestamp'], np.arange(n) / fps, atol=1e-4, rtol=0)):
                raise ValueError(f'Unsupported shared data file or invalid timestamps: {path}')
            state, action = convert_units(data, follower)
            columns = {'observation.state': state, 'action': action,
                       'timestamp': np.arange(n, dtype=np.float32) / fps,
                       'frame_index': np.arange(n, dtype=np.int64),
                       'episode_index': np.full(n, episode, dtype=np.int64),
                       'index': np.arange(offset, offset + n, dtype=np.int64),
                       'task_index': np.zeros(n, dtype=np.int64)}
            chunk = episode // 1000
            write_parquet(stage / f'data/chunk-{chunk:03d}/episode_{episode:06d}.parquet', columns)
            shutil.copy2(path, stage / f'audit/episode_{episode:06d}.source.parquet')
            stats = {}
            for key in totals:
                moment = Moments()
                moment.add(columns[key])
                stats[key] = moment.result()
                if episode < n_train:
                    totals[key].add(columns[key])
            for old, new in CAMERAS.items():
                if row[f'videos/{old}/from_timestamp'] != 0:
                    raise ValueError('Expected one complete video per source episode')
                video = source / info['video_path'].format(
                    video_key=old, chunk_index=row[f'videos/{old}/chunk_index'],
                    file_index=row[f'videos/{old}/file_index'])
                target = stage / f'videos/chunk-{chunk:03d}/{new}/episode_{episode:06d}.mp4'
                if not video.is_file():
                    raise FileNotFoundError(video)
                jobs.append((video, target, n, fps))
                # The original RGB statistics are an estimate after re-encoding;
                # visual normalization is IDENTITY. Counts are metadata frame counts.
                stats[new] = {field: row[f'stats/{old}/{field}'] for field in ('min', 'max', 'mean', 'std')}
                stats[new]['count'] = [n]
            summaries.append({'episode_index': episode, 'tasks': ['pick_place_bag'], 'length': n})
            episode_stats.append({'episode_index': episode, 'stats': stats})
            times = np.asarray(data['timing.unix_time'])
            provenance.append({'episode_index': episode, 'source_episode_index': original,
                               'split': 'train' if episode < n_train else 'val', 'frames': n,
                               'recording_wall_seconds': float(times[-1] - times[0]),
                               'nominal_video_seconds': n / fps})
            offset += n
        print(f'Converted vectors: {len(metadata)} episodes, {offset} frames; encoding {len(jobs)} videos', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, video in enumerate(pool.map(transcode, jobs), 1):
                if i % 12 == 0:
                    print(f'Encoded {i}/{len(jobs)} videos', flush=True)
        features = {key: copy.deepcopy(info['features'][key]) for key in ('observation.state', 'action')}
        for old, new in CAMERAS.items():
            features[new] = {'dtype': 'video', 'shape': [480, 640, 3],
                             'names': ['height', 'width', 'channels'], 'video_info': {
                                 'video.fps': fps, 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                                 'video.channels': 3, 'video.is_depth_map': False, 'has_audio': False}}
        for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
            features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
        global_stats = {key: moment.result() for key, moment in totals.items()}
        train_frames = sum(row['length'] for row in summaries[:n_train])
        for key in VIDEO_KEYS:
            selected = [row['stats'][key] for row in episode_stats[:n_train]]
            counts = np.array([row['count'][0] for row in selected])[:, None, None, None]
            means = np.array([row['mean'] for row in selected])
            mean = (means * counts).sum(0) / counts.sum()
            variance = ((np.array([row['std'] for row in selected]) ** 2 + (means - mean) ** 2) * counts).sum(0) / counts.sum()
            global_stats[key] = {'mean': mean.tolist(), 'std': np.sqrt(variance).tolist(),
                                 'min': np.array([row['min'] for row in selected]).min(0).tolist(),
                                 'max': np.array([row['max'] for row in selected]).max(0).tolist(),
                                 'count': [train_frames]}
        write_json(stage / 'meta/info.json', {
            'codebase_version': 'v2.1', 'robot_type': 'so101_follower', 'repo_id': REPO,
            'total_episodes': len(metadata), 'total_frames': offset, 'total_tasks': 1,
            'total_videos': 2 * len(metadata), 'total_chunks': (len(metadata)-1)//1000+1,
            'chunks_size': 1000, 'fps': fps, 'splits': {'train': f'0:{n_train}', 'val': f'{n_train}:{len(metadata)}'},
            'features': features,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'})
        write_json(stage / 'meta/stats.json', global_stats)
        write_jsonl(stage / 'meta/episodes.jsonl', summaries)
        write_jsonl(stage / 'meta/episodes_stats.jsonl', episode_stats)
        write_jsonl(stage / 'meta/tasks.jsonl', tasks)
        write_json(stage / 'meta/train_episodes.json', list(range(n_train)))
        write_json(stage / 'meta/val_episodes.json', list(range(n_train, len(metadata))))
        report = {'source_repo': REPO, 'source_revision': REVISION,
                  'camera_mapping': CAMERAS, 'state_source': 'observation.follower.raw_position',
                  'action_source': 'action.goal_sent', 'state_units': 'five RANGE_M100_100 + gripper RANGE_0_100',
                  'follower_calibration': follower, 'frame_alignment': 'original row/video frame index preserved; no resampling',
                  'image_statistics': 'source RGB estimates; VISUAL uses IDENTITY normalization',
                  'episodes': provenance, 'train_frames': train_frames, 'val_frames': offset - train_frames}
        write_json(stage / 'conversion_report.json', report)
        for filename in ('robot_calibration.json', 'camera_info.json', 'alerts_per_episode.jsonl', 'episode_failures.jsonl'):
            shutil.copy2(source / filename, stage / 'audit' / filename)
        from validate_dataset import validate
        validate(stage, checksums=False)
        manifest = [f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(stage)}\n'
                    for path in sorted(stage.rglob('*')) if path.is_file()]
        (stage / 'SHA256SUMS').write_text(''.join(manifest))
        stage.rename(destination)
        print('DATASET_READY', destination, flush=True)
    except BaseException:
        # Preserve partial diagnostics instead of silently deleting expensive work.
        print('Incomplete conversion retained for inspection:', stage, flush=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit-episodes', type=int, help='Small real-data subset for a training smoke test')
    args = parser.parse_args()
    build(args.source, args.output, args.workers, args.limit_episodes)
