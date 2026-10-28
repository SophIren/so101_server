#!/usr/bin/env python3
"""Restore the exact frozen RGB dataset and physically isolated holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import tempfile


def restore_holdout(source, destination, saved):
    """Rebuild just the 12 original holdout episodes, using unchanged conversion."""
    import numpy as np
    import pyarrow.parquet as pq
    from prepare_primer19 import convert_units, transcode, CAMERAS
    from dataset_tools import Moments, VIDEO_KEYS
    from prepare_dataset import write_json, write_jsonl, write_parquet
    from isolate_training_splits import aggregate
    info = json.loads((source / 'meta/info.json').read_text())
    calibration = json.loads((source / 'robot_calibration.json').read_text())['follower']
    if calibration != saved['follower_calibration']:
        raise ValueError('Source calibration has changed')
    rows = {row['episode_index']: row for path in (source / 'meta/episodes').rglob('*.parquet')
            for row in pq.read_table(path).to_pylist()}
    selected = [row for row in saved['episodes'] if row['split'] == 'val']
    if len(selected) != 12 or sum(row['frames'] for row in selected) != 10578:
        raise ValueError('Invalid saved holdout provenance')
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    (stage / 'meta').mkdir()
    summaries, stats, mapping, jobs = [], [], [], []
    offset = 0
    for new, provenance in enumerate(selected):
        original = provenance['source_episode_index']
        row = rows[original]
        n = row['length']
        if n != provenance['frames']:
            raise ValueError('Source episode length changed')
        data = pq.read_table(source / info['data_path'].format(
            chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])).to_pydict()
        if (set(data['episode_index']) != {original} or data['frame_index'] != list(range(n))
                or not np.allclose(data['timestamp'], np.arange(n) / info['fps'], atol=1e-4, rtol=0)):
            raise ValueError('Source row/video alignment changed')
        state, action = convert_units(data, calibration)
        columns = {'observation.state': state, 'action': action,
                   'timestamp': np.arange(n, dtype=np.float32) / info['fps'],
                   'frame_index': np.arange(n, dtype=np.int64), 'index': np.arange(offset, offset + n),
                   'episode_index': np.full(n, new, dtype=np.int64), 'task_index': np.zeros(n, dtype=np.int64)}
        write_parquet(stage / f'data/chunk-000/episode_{new:06d}.parquet', columns)
        ep_stats = {}
        for key in ('observation.state', 'action'):
            moment = Moments()
            moment.add(columns[key])
            ep_stats[key] = moment.result()
        for old, key in CAMERAS.items():
            if row[f'videos/{old}/from_timestamp'] != 0:
                raise ValueError('Source video does not start at frame zero')
            video = source / info['video_path'].format(video_key=old,
                chunk_index=row[f'videos/{old}/chunk_index'], file_index=row[f'videos/{old}/file_index'])
            target = stage / f'videos/chunk-000/{key}/episode_{new:06d}.mp4'
            jobs.append((video, target, n, info['fps']))
            ep_stats[key] = {field: row[f'stats/{old}/{field}'] for field in ('min', 'max', 'mean', 'std')}
            ep_stats[key]['count'] = [n]
        summaries.append({'episode_index': new, 'length': n, 'tasks': ['pick_place_bag']})
        stats.append({'episode_index': new, 'stats': ep_stats})
        mapping.append({'episode_index': new, 'converted_source_episode_index': provenance['episode_index']})
        offset += n
    print(f'Restoring exact holdout: {len(selected)} episodes, {offset} frames, {len(jobs)} RGB videos', flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(transcode, jobs), 1):
            print(f'Encoded holdout {i}/{len(jobs)}', flush=True)
    features = {key: copy.deepcopy(info['features'][key]) for key in ('observation.state', 'action')}
    for key in VIDEO_KEYS:
        features[key] = {'dtype': 'video', 'shape': [480, 640, 3], 'names': ['height', 'width', 'channels'],
                         'video_info': {'video.fps': info['fps'], 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                                        'video.channels': 3, 'video.is_depth_map': False, 'has_audio': False}}
    for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
        features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
    write_json(stage / 'meta/info.json', {'codebase_version': 'v2.1', 'robot_type': 'so101_follower',
        'repo_id': saved['source_repo'], 'total_episodes': 12, 'total_frames': offset, 'total_tasks': 1,
        'total_videos': 24, 'total_chunks': 1, 'chunks_size': 1000, 'fps': info['fps'],
        'splits': {'train': '0:12'}, 'so101_split_role': 'validation', 'features': features,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'})
    write_jsonl(stage / 'meta/episodes.jsonl', summaries)
    write_jsonl(stage / 'meta/episodes_stats.jsonl', stats)
    write_jsonl(stage / 'meta/tasks.jsonl', [{'task_index': 0, 'task': 'pick_place_bag'}])
    write_json(stage / 'meta/stats.json', aggregate([row['stats'] for row in stats]))
    write_json(stage / 'meta/train_episodes.json', list(range(12)))
    write_json(stage / 'meta/val_episodes.json', [])
    report = copy.deepcopy(saved)
    report.update({'physical_split_role': 'validation', 'split_episode_mapping': mapping,
                   'train_frames': 0, 'val_frames': offset})
    write_json(stage / 'conversion_report.json', report)
    from validate_dataset import validate
    validate(stage, checksums=False)
    (stage / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(stage)}\n'
        for path in sorted(stage.rglob('*')) if path.is_file()))
    stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/so101'))
    parser.add_argument('--holdout-only', action='store_true')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from prepare_primer19 import REPO, REVISION, build
    from isolate_training_splits import isolate
    source = args.root / 'datasets/primer19_source'
    snapshot_download(REPO, repo_type='dataset', revision=REVISION, local_dir=source,
                      allow_patterns=['README.md', 'so101_teleop_bag/meta/*',
                                      'so101_teleop_bag/data/*', 'so101_teleop_bag/*.json',
                                      'so101_teleop_bag/*.jsonl',
                                      'so101_teleop_bag/videos/observation.images.front/*',
                                      'so101_teleop_bag/videos/observation.images.wrist/*'],
                      max_workers=6)
    validation = args.root / 'datasets/primer19_bag_validation_v21'
    saved = json.loads((Path(__file__).parent / 'runs/primer19_bag_lora_20261027/conversion_report.json').read_text())
    if args.holdout_only:
        if not validation.exists():
            restore_holdout(source / 'so101_teleop_bag', validation, saved)
    else:
        full = args.root / 'datasets/primer19_bag_v21'
        if not full.exists():
            build(source / 'so101_teleop_bag', full)
        if not validation.exists():
            isolate(full, validation, 'validation')
    restored = json.loads((validation / 'conversion_report.json').read_text())
    for field in ('source_revision', 'episodes', 'follower_calibration', 'camera_mapping',
                  'action_source', 'state_source'):
        if restored[field] != saved[field]:
            raise ValueError(f'Restored dataset does not match original training provenance: {field}')
    print('EXACT_HOLDOUT_RESTORED', validation, flush=True)


if __name__ == '__main__':
    main()
