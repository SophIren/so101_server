#!/usr/bin/env python3
"""Verify a local LeRobot v2.1 dataset, videos, timestamps, splits and checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from dataset_tools import VIDEO_KEYS, ffmpeg_executable, read_jsonl, verify_video


def first_video_timestamp(path):
    result = subprocess.run([ffmpeg_executable(), '-hide_banner', '-copyts', '-i', str(path),
                             '-an', '-vf', 'showinfo', '-frames:v', '1', '-f', 'null', '-'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=60)
    match = re.search(r'\bn:\s*0\s+pts:\s*-?\d+\s+pts_time:([^\s]+)', result.stderr)
    if result.returncode or match is None:
        raise ValueError(f'Could not verify first video PTS: {path}\n{result.stderr[-1500:]}')
    return float(match.group(1))


def validate(root, checksums=True):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1':
        raise ValueError('Expected the v2.1 format used by this VLASH stack')
    visual = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    if visual != list(VIDEO_KEYS):
        raise ValueError(f'Expected ordered base + left wrist slots, got {visual}')
    fps = float(info['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    episodes = read_jsonl(root / 'meta/episodes.jsonl')
    tasks = read_jsonl(root / 'meta/tasks.jsonl')
    episode_stats = read_jsonl(root / 'meta/episodes_stats.jsonl')
    train = json.loads((root / 'meta/train_episodes.json').read_text())
    val = json.loads((root / 'meta/val_episodes.json').read_text())
    if not train or sorted(train + val) != list(range(len(episodes))) or set(train) & set(val):
        raise ValueError('Invalid/disjointness failure in train/val episode split')
    if info['splits']['train'] != f'0:{len(train)}' or train != list(range(len(train))):
        raise ValueError('Training split metadata does not match train_episodes.json')
    if len(tasks) != info['total_tasks'] or [t['task_index'] for t in tasks] != list(range(len(tasks))):
        raise ValueError('Invalid task metadata')
    if len(episode_stats) != len(episodes):
        raise ValueError('Missing episode statistics')
    for index, row in enumerate(episode_stats):
        if row['episode_index'] != index:
            raise ValueError('Episode statistics are out of order')
        for key in ('action', 'observation.state', *VIDEO_KEYS):
            shape = (3, 1, 1) if key in VIDEO_KEYS else (6,)
            for name in ('mean', 'std', 'min', 'max'):
                values = np.asarray(row['stats'][key][name])
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f'Bad episode statistics for {index}/{key}/{name}')
            if row['stats'][key]['count'] != [episodes[index]['length']]:
                raise ValueError('Episode statistics have an incorrect frame count')
    offset, video_count = 0, 0
    for episode_index, episode in enumerate(episodes):
        if episode['episode_index'] != episode_index or episode['length'] < 2:
            raise ValueError('Noncontiguous or empty episode metadata')
        n = episode['length']
        chunk = episode_index // info['chunks_size']
        data_path = root / info['data_path'].format(episode_chunk=chunk, episode_index=episode_index)
        table = pq.read_table(data_path)
        if len(table) != n:
            raise ValueError(f'Frame count mismatch in {data_path}')
        data = table.to_pydict()
        for key in ('action', 'observation.state'):
            values = np.asarray(data[key])
            if values.shape != (n, 6) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {key} in episode {episode_index}')
        if not np.allclose(data['timestamp'], np.arange(n) / fps, atol=1e-4, rtol=0):
            raise ValueError(f'Timestamp/FPS mismatch in episode {episode_index}')
        for key, expected in (('frame_index', np.arange(n)), ('index', np.arange(offset, offset + n)),
                              ('episode_index', np.full(n, episode_index))):
            if not np.array_equal(data[key], expected):
                raise ValueError(f'Invalid {key} in episode {episode_index}')
        if not all(0 <= i < len(tasks) for i in data['task_index']):
            raise ValueError('Unknown task index')
        if episode['tasks'] != [tasks[data['task_index'][0]]['task']]:
            raise ValueError('Task label mismatch')
        for key in VIDEO_KEYS:
            feature = info['features'][key]
            height, width, channels = feature['shape']
            if channels != 3 or abs(feature['video_info']['video.fps'] - fps) > 1e-6:
                raise ValueError('Video feature shape/FPS mismatch')
            path = root / info['video_path'].format(episode_chunk=chunk, video_key=key, episode_index=episode_index)
            verify_video(path, n, fps, (width, height))
            if abs(first_video_timestamp(path)) > 1e-4:
                raise ValueError(f'Video must start at PTS 0, just like the parquet: {path}')
            video_count += 1
        offset += n
    if (offset != info['total_frames'] or len(episodes) != info['total_episodes']
            or video_count != info['total_videos']):
        raise ValueError('Dataset totals do not match files')
    stats = json.loads((root / 'meta/stats.json').read_text())
    training_frames = sum(episodes[i]['length'] for i in train)
    for key in ('action', 'observation.state', *VIDEO_KEYS):
        expected_shape = (3, 1, 1) if key in VIDEO_KEYS else (6,)
        for statistic in ('min', 'max', 'mean', 'std'):
            values = np.asarray(stats[key][statistic])
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid normalization statistics: {key}/{statistic}')
        if (np.asarray(stats[key]['std']) <= 0).any() or stats[key]['count'] != [training_frames]:
            raise ValueError(f'Invalid std or train-only counts for {key}')
    manifest = root / 'SHA256SUMS'
    if checksums and manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, filename = line.split('  ', 1)
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Checksum mismatch: {filename}')
    print(f'VALID: {len(episodes)} episodes, {offset} frames, {video_count} videos, '
          f'{fps:g} fps; train={len(train)}, val={len(val)}', flush=True)
    return info


def check_lerobot(root, repo_id):
    # The exact compatibility layer used by the remote inference branch.
    import vlash.datasets.compat  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root).resolve()
    episodes = json.loads((root / 'meta/train_episodes.json').read_text())
    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, video_backend='torchcodec')
    for i in (0, len(dataset) - 1):
        sample = dataset[i]
        for key in VIDEO_KEYS:
            image = sample[key]
            if image.ndim != 3 or image.shape[0] != 3 or not image.isfinite().all():
                raise ValueError(f'LeRobot returned an invalid image: {key}')
        if sample['action'].shape[-1] != 6 or sample['observation.state'].shape[-1] != 6:
            raise ValueError('LeRobot returned an invalid action/state')
    print('LeRobot + VLASH loader: PASS (first/last training samples and both videos)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--lerobot', action='store_true', help='Additional target-stack loader check (run on server)')
    args = parser.parse_args(argv)
    info = validate(args.dataset)
    if args.lerobot:
        check_lerobot(args.dataset, info.get('repo_id', 'local/so101-task'))


if __name__ == '__main__':
    main()
