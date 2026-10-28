#!/usr/bin/env python3
"""Physically isolate train/validation: LeRobot 0.4.1 loads all local data files."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import write_json, write_jsonl, write_parquet
from dataset_tools import read_jsonl, VIDEO_KEYS


def aggregate(rows):
    result = {}
    for key in rows[0]:
        means = np.asarray([row[key]['mean'] for row in rows])
        stds = np.asarray([row[key]['std'] for row in rows])
        counts = np.asarray([row[key]['count'][0] for row in rows])
        weights = counts.reshape((-1,) + (1,)*(means.ndim-1))
        mean = (means*weights).sum(0)/counts.sum()
        variance = ((stds**2+(means-mean)**2)*weights).sum(0)/counts.sum()
        result[key] = {'mean': mean.tolist(), 'std': np.maximum(np.sqrt(variance), 1e-6).tolist(),
                       'min': np.asarray([row[key]['min'] for row in rows]).min(0).tolist(),
                       'max': np.asarray([row[key]['max'] for row in rows]).max(0).tolist(),
                       'count': [int(counts.sum())]}
    return result


def isolate(source, destination, role):
    if destination.exists():
        raise ValueError(f'Output exists: {destination}')
    info = json.loads((source/'meta/info.json').read_text())
    selected = json.loads((source/f'meta/{"train" if role == "train" else "val"}_episodes.json').read_text())
    episodes = read_jsonl(source/'meta/episodes.jsonl')
    all_stats = read_jsonl(source/'meta/episodes_stats.jsonl')
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent))
    (stage/'meta').mkdir()
    summaries, stats, mapping = [], [], []
    offset = 0
    for new, old in enumerate(selected):
        chunk = old//info['chunks_size']
        data = pq.read_table(source/info['data_path'].format(episode_chunk=chunk,episode_index=old)).to_pydict()
        n = episodes[old]['length']
        if set(data['episode_index']) != {old} or len(data['index']) != n:
            raise ValueError('Invalid source episode selection')
        data['episode_index'] = np.full(n,new,dtype=np.int64)
        data['index'] = np.arange(offset,offset+n,dtype=np.int64)
        write_parquet(stage/info['data_path'].format(episode_chunk=new//1000,episode_index=new),data)
        for key in VIDEO_KEYS:
            path = source/info['video_path'].format(episode_chunk=chunk,episode_index=old,video_key=key)
            target = stage/info['video_path'].format(episode_chunk=new//1000,episode_index=new,video_key=key)
            target.parent.mkdir(parents=True,exist_ok=True)
            os.link(path,target)
        summaries.append({'episode_index':new,'length':n,'tasks':episodes[old]['tasks']})
        stats.append({'episode_index':new,'stats':all_stats[old]['stats']})
        mapping.append({'episode_index':new,'converted_source_episode_index':old})
        offset += n
    output_info = copy.deepcopy(info)
    output_info.update({'total_episodes':len(selected),'total_frames':offset,'total_videos':2*len(selected),
                        'total_chunks':(len(selected)-1)//1000+1,
                        'splits':{'train':f'0:{len(selected)}'},'so101_split_role':role})
    write_json(stage/'meta/info.json',output_info)
    write_json(stage/'meta/stats.json',aggregate([row['stats'] for row in stats]))
    write_jsonl(stage/'meta/episodes.jsonl',summaries)
    write_jsonl(stage/'meta/episodes_stats.jsonl',stats)
    write_json(stage/'meta/train_episodes.json',list(range(len(selected))))
    write_json(stage/'meta/val_episodes.json',[])
    (stage/'meta/tasks.jsonl').write_bytes((source/'meta/tasks.jsonl').read_bytes())
    report = json.loads((source/'conversion_report.json').read_text())
    report.update({'physical_split_role':role,'split_episode_mapping':mapping,
                   'train_frames':offset if role=='train' else 0,
                   'val_frames':offset if role!='train' else 0,
                   'video_validation':'hardlinks to already fully decoded and verified source videos'})
    write_json(stage/'conversion_report.json',report)
    (stage/'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(stage)}\n'
        for p in sorted(stage.rglob('*')) if p.is_file()))
    stage.rename(destination)
    print(f'ISOLATED {role}: {len(selected)} episodes, {offset} frames -> {destination}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--prefix',required=True,type=Path)
    args=parser.parse_args()
    for role in ('train','validation'):
        isolate(args.source.resolve(),args.prefix.parent/(args.prefix.name+'_'+role+'_v21'),role)
