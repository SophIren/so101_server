#!/usr/bin/env python3
"""Build a checked LeRobot v2.1 task dataset from recorded successful episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from dataset_capture import write_json
from dataset_tools import (CAMERAS, VIDEO_KEYS, Moments, encode_video, resample_indices,
                           validate_raw_episode)
from policy_client import JOINT_ORDER


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def write_parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    features, arrays = {}, {}
    for name, values in columns.items():
        if name in ('action', 'observation.state'):
            arrays[name] = pa.array(np.asarray(values).tolist(), type=pa.list_(pa.float32(), 6))
            features[name] = {'feature': {'dtype': 'float32', '_type': 'Value'}, 'length': 6, '_type': 'Sequence'}
        else:
            dtype = 'float32' if name == 'timestamp' else 'int64'
            arrays[name] = pa.array(values, type=pa.float32() if dtype == 'float32' else pa.int64())
            features[name] = {'dtype': dtype, '_type': 'Value'}
    table = pa.table(arrays)
    table = table.replace_schema_metadata({b'huggingface': json.dumps({'info': {'features': features}}).encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build(source, destination, repo_id, fps=None, val_fraction=.1, include_failed=False,
          min_seconds=1, allow_synthetic=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Output directory already exists. Choose a new --output; source data is never overwritten.')
    metadata = json.loads((source / 'metadata.json').read_text())
    config = metadata['configuration']
    if config['synthetic'] and not allow_synthetic:
        raise ValueError('Synthetic data is only for pipeline checks; use --allow-synthetic explicitly')
    fps = float(fps or config['fps'])
    if fps > config['fps'] + 1e-6 or fps <= 0:
        raise ValueError('Export FPS must be positive and <= recording FPS; do not upsample demonstrations')
    width, height = config['image_size']
    if width % 2 or height % 2:
        raise ValueError('H.264 yuv420p requires even image dimensions')
    episodes, excluded = [], []
    for path in sorted((source / 'episodes').glob('episode_*')):
        info = json.loads((path / 'episode.json').read_text())
        if info.get('status') != 'complete' or info.get('outcome') not in (
                ('success', 'failed') if include_failed else ('success',)):
            excluded.append({'episode': path.name, 'reason': info.get('outcome', info.get('status'))})
            continue
        info, rows = validate_raw_episode(path, metadata)
        if rows[-1]['observation_time'] - rows[0]['observation_time'] < min_seconds:
            raise ValueError(f'{path.name}: episode shorter than {min_seconds}s')
        episodes.append((path, info, rows))
    if not episodes:
        raise ValueError('No completed successful episodes to export')
    random.Random(1000).shuffle(episodes)
    n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_fraction))) if val_fraction else 0
    n_train = len(episodes) - n_val
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.building-', dir=destination.parent))
    try:
        meta = stage / 'meta'
        meta.mkdir()
        (stage / 'audit').mkdir()
        summaries, episode_stats, tasks, provenance = [], [], [], []
        global_moments = {key: Moments() for key in ('action', 'observation.state', *VIDEO_KEYS)}
        offset = 0
        for episode_index, (path, info, rows) in enumerate(episodes):
            if info['task'] not in tasks:
                tasks.append(info['task'])
            selected, grid, timing_error = resample_indices(rows, fps)
            actual_fps = (len(rows) - 1) / (rows[-1]['observation_time'] - rows[0]['observation_time'])
            if fps > actual_fps * 1.1:
                raise ValueError(f'{path.name}: actual sampling is {actual_fps:.2f} Hz. '
                                 f'Use --fps {max(1, int(actual_fps))} or record at a sustainable rate.')
            if len(grid) < 2:
                raise ValueError(f'{path.name}: too few frames after resampling')
            samples = [rows[int(i)] for i in selected]
            n = len(samples)
            columns = {
                'observation.state': np.asarray([r['observation.state'] for r in samples], np.float32),
                'action': np.asarray([r['action'] for r in samples], np.float32),
                'timestamp': grid.astype(np.float32), 'frame_index': np.arange(n, dtype=np.int64),
                'episode_index': np.full(n, episode_index, np.int64),
                'index': np.arange(offset, offset + n, dtype=np.int64),
                'task_index': np.full(n, tasks.index(info['task']), np.int64),
            }
            chunk = episode_index // 1000
            write_parquet(stage / f'data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet', columns)
            stats = {}
            for key in ('action', 'observation.state'):
                moment = Moments()
                moment.add(columns[key])
                stats[key] = moment.result()
                if episode_index < n_train:
                    global_moments[key].add(columns[key])
            for camera, key in zip(CAMERAS, VIDEO_KEYS):
                moment = Moments()
                video = stage / f'videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4'
                encode_video(video, path, rows, selected, camera, fps, (width, height), moment)
                stats[key] = moment.result(image=True)
                stats[key]['count'] = [n]
                if episode_index < n_train:
                    # Merge pixel moments; counts in output image stats are frame counts.
                    current = global_moments[key]
                    if current.count == 0:
                        current.total, current.squared = moment.total.copy(), moment.squared.copy()
                        current.minimum, current.maximum = moment.minimum.copy(), moment.maximum.copy()
                    else:
                        current.total += moment.total
                        current.squared += moment.squared
                        current.minimum = np.minimum(current.minimum, moment.minimum)
                        current.maximum = np.maximum(current.maximum, moment.maximum)
                    current.count += moment.count
            summaries.append({'episode_index': episode_index, 'tasks': [info['task']], 'length': n})
            episode_stats.append({'episode_index': episode_index, 'stats': stats})
            provenance.append({'episode_index': episode_index, 'raw_episode': path.name,
                               'outcome': info['outcome'], 'split': 'train' if episode_index < n_train else 'val'})
            write_json(stage / 'audit' / f'episode_{episode_index:06d}.json', {
                'raw_episode': path.name, 'raw_sample_indices': selected.tolist(),
                'observation_time_offsets_seconds': timing_error.tolist(),
                'raw_frames': len(rows), 'exported_frames': n, 'actual_sampling_hz': actual_fps,
                'resampled_repeats': int(len(selected) - len(set(selected.tolist())))})
            offset += n
            print(f'Exported episode {episode_index}: {n} frames, 2 videos', flush=True)
        train_frames = sum(r['length'] for r in summaries[:n_train])
        stats = {key: moment.result(image=key in VIDEO_KEYS) for key, moment in global_moments.items()}
        for key in VIDEO_KEYS:
            stats[key]['count'] = [train_frames]
        vector_names = [f'{name}.pos' for name in JOINT_ORDER]
        features = {name: {'dtype': 'float32', 'shape': [6], 'names': vector_names}
                    for name in ('observation.state', 'action')}
        for key in VIDEO_KEYS:
            features[key] = {'dtype': 'video', 'shape': [height, width, 3],
                             'names': ['height', 'width', 'channels'], 'video_info': {
                                 'video.fps': fps, 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                                 'video.is_depth_map': False, 'video.channels': 3, 'has_audio': False}}
        for name in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
            features[name] = {'dtype': 'float32' if name == 'timestamp' else 'int64', 'shape': [1], 'names': None}
        splits = {'train': f'0:{n_train}'}
        if n_val:
            splits['val'] = f'{n_train}:{len(episodes)}'
        write_json(meta / 'info.json', {
            'codebase_version': 'v2.1', 'robot_type': 'so101_follower',
            'total_episodes': len(episodes), 'total_frames': offset, 'total_tasks': len(tasks),
            'total_videos': 2 * len(episodes), 'total_chunks': (len(episodes) - 1) // 1000 + 1,
            'chunks_size': 1000, 'fps': fps, 'splits': splits,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': features, 'repo_id': repo_id, 'so101_synthetic': config['synthetic'],
        })
        write_json(meta / 'stats.json', stats)
        write_jsonl(meta / 'episodes.jsonl', summaries)
        write_jsonl(meta / 'episodes_stats.jsonl', episode_stats)
        write_jsonl(meta / 'tasks.jsonl', [{'task_index': i, 'task': task} for i, task in enumerate(tasks)])
        write_json(meta / 'sources.json', provenance)
        write_json(meta / 'train_episodes.json', list(range(n_train)))
        write_json(meta / 'val_episodes.json', list(range(n_train, len(episodes))))
        write_json(stage / 'recording_metadata.json', metadata)
        write_json(stage / 'conversion_report.json', {'excluded': excluded, 'train_episodes': n_train,
                   'val_episodes': n_val, 'fps': fps, 'stats_split': 'train',
                   'resampling': 'nearest complete observation/action pair on a fixed grid; no upsampling'})
        from validate_dataset import validate
        validate(stage)
        manifest = []
        for filename in sorted(stage.rglob('*')):
            if filename.is_file():
                digest = hashlib.sha256(filename.read_bytes()).hexdigest()
                manifest.append(f'{digest}  {filename.relative_to(stage)}\n')
        (stage / 'SHA256SUMS').write_text(''.join(manifest))
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f'Dataset ready: {destination}; train={n_train}, val={n_val}, fps={fps:g}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True, help='e.g. marks/so101-box')
    parser.add_argument('--fps', type=float, help='Default: original recording rate; may only be reduced')
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--min-seconds', type=float, default=1)
    parser.add_argument('--include-failed', action='store_true', help='Explicitly train on failed episodes as well')
    parser.add_argument('--allow-synthetic', action='store_true', help='File pipeline test only')
    args = parser.parse_args(argv)
    if not 0 <= args.val_fraction < 1 or args.min_seconds < 0:
        parser.error('val-fraction must be in [0,1); min-seconds must be nonnegative')
    build(args.source, args.output, args.repo_id, args.fps, args.val_fraction,
          args.include_failed, args.min_seconds, args.allow_synthetic)


if __name__ == '__main__':
    main()
