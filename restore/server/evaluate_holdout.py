#!/usr/bin/env python3
"""Offline held-out imitation metrics. Does not measure robot task success."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
SEED = 20261028


def fixed_random(indices, chunk_size, dim, inference=False):
    noises, times = [], []
    for index in indices:
        gen = torch.Generator().manual_seed(SEED + int(index) + (1000000 if inference else 0))
        noises.append(torch.randn(chunk_size, dim, generator=gen))
        # Exact inverse CDF for Beta(1.5, 1), as used by PI05.sample_time.
        times.append(torch.rand((), generator=gen).pow(1 / 1.5) * .999 + .001)
    return torch.stack(noises).cuda(), torch.stack(times).cuda()


def to_gpu(batch):
    return {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--flow-stride', type=int, default=1)
    parser.add_argument('--action-stride', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    from vlash.configs import ServerConfig
    from vlash.so101_policy_loader import load_policy
    import vlash.datasets.compat  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    if info.get('so101_split_role') != 'validation':
        raise ValueError('Evaluation requires physically isolated validation data')
    report = json.loads((args.dataset / 'conversion_report.json').read_text())
    heldout = [row for row in report['episodes'] if row['split'] == 'val']
    if len(heldout) != info['total_episodes']:
        raise ValueError('Holdout episode provenance mismatch')
    cfg = ServerConfig(policy={'path': str(args.checkpoint.resolve()), 'device': 'cuda',
                               'empty_cameras': 1, 'n_action_steps': 32,
                               'compile_model': False, 'fuse_qkv': False, 'fuse_gate_up': False})
    policy = load_policy(cfg.policy).cuda().eval()
    dataset = LeRobotDataset('PriMeR19/SO-101-teleop', root=args.dataset,
                            delta_timestamps={'action': [i / info['fps'] for i in range(policy.config.chunk_size)]},
                            video_backend='torchcodec')
    if len(dataset) != 10578 or info['total_episodes'] != 12:
        raise ValueError('Unexpected holdout size')
    episodes = [json.loads(line) for line in (args.dataset / 'meta/episodes.jsonl').read_text().splitlines()]
    result = {'checkpoint': str(args.checkpoint.resolve()), 'dataset': str(args.dataset.resolve()),
              'source_revision': report['source_revision'], 'seed': SEED,
              'task': 'pick_place_bag', 'episodes': len(episodes), 'frames': len(dataset),
              'flow_stride': args.flow_stride, 'action_stride': args.action_stride,
              'action_horizon': 32, 'flow_horizon': policy.config.chunk_size,
              'joint_order': JOINTS, 'normalization': {}, 'per_episode': [],
              'limitations': ['Offline comparison to recorded expert commands, not task success.',
                             'Actions use recorded observations, not a closed-loop rollout.',
                             'One fixed flow time/noise draw per frame; one inference noise draw per anchor.']}
    weights = args.checkpoint / 'lora_adapters/adapter_model.safetensors'
    if not weights.is_file():
        weights = args.checkpoint / 'model.safetensors'
    result['weights_sha256'] = file_hash(weights)
    result['dataset_manifest_sha256'] = file_hash(args.dataset / 'SHA256SUMS')
    for key, buffer in [('state', policy.normalize_inputs.buffer_observation_state),
                        ('action', policy.normalize_targets.buffer_action)]:
        result['normalization'][key] = {'mean': buffer.mean.cpu().tolist(), 'std': buffer.std.cpu().tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sums = np.zeros((len(episodes), 3), dtype=np.float64)
    action_abs = np.zeros((len(episodes), 6), dtype=np.float64)
    action_sq = np.zeros_like(action_abs)
    first_abs, first_sq, hold_abs = [np.zeros_like(action_abs) for _ in range(3)]
    counts, first_counts = np.zeros(len(episodes)), np.zeros(len(episodes))
    start = time.monotonic()
    for phase, stride in [('flow', args.flow_stride), ('action', args.action_stride)]:
        indices, offset = [], 0
        for episode in episodes:
            indices.extend(range(offset, offset + episode['length'], stride))
            offset += episode['length']
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)
        consumed = 0
        for cpu in loader:
            batch = to_gpu(cpu)
            sample_indices = indices[consumed:consumed + len(cpu['index'])]
            eps = cpu['episode_index'].reshape(-1).numpy()
            valid = ~batch['action_is_pad']
            with torch.inference_mode():
                noise, times = fixed_random(sample_indices, policy.config.chunk_size,
                                            policy.config.max_action_dim, inference=(phase == 'action'))
                if phase == 'flow':
                    normalized = policy.normalize_targets(policy.normalize_inputs(batch))
                    images, masks = policy.prepare_images(normalized)
                    tokens, token_masks = policy.prepare_language(normalized)
                    losses = policy.model.forward(images, masks, tokens, token_masks,
                                                  policy.prepare_state(normalized),
                                                  policy.prepare_action(normalized), noise=noise, time=times)
                    losses = losses.float() * valid.unsqueeze(-1)
                    values = torch.stack([losses.sum((1, 2)), losses[:, :, :6].sum((1, 2)),
                                          valid.sum(1).float()], dim=1).cpu().numpy()
                    for j, ep in enumerate(eps):
                        sums[ep] += values[j]
                else:
                    predicted = policy.predict_action_chunk(batch, noise=noise).float()
                    if predicted.shape[1:] != (32, 6) or not torch.isfinite(predicted).all():
                        raise ValueError('Nonfinite or malformed inferred actions')
                    target = batch['action'][:, :32].float()
                    mask = valid[:, :32].unsqueeze(-1)
                    error = predicted - target
                    hold_error = (batch['observation.state'][:, None, :] - target).abs()
                    arrays = [(error.abs() * mask).sum(1), (error.square() * mask).sum(1),
                              error[:, 0].abs(), error[:, 0].square(),
                              (hold_error * mask).sum(1), mask.sum(1)]
                    aa, ss, fa, fs, ha, nn = [value.cpu().numpy() for value in arrays]
                    for j, ep in enumerate(eps):
                        action_abs[ep] += aa[j]
                        action_sq[ep] += ss[j]
                        first_abs[ep] += fa[j]
                        first_sq[ep] += fs[j]
                        hold_abs[ep] += ha[j]
                        counts[ep] += nn[j, 0]
                        first_counts[ep] += 1
            consumed += len(eps)
            if consumed % (args.batch_size * 50) == 0 or consumed == len(indices):
                print(json.dumps({'phase': phase, 'processed': consumed, 'total': len(indices),
                                  'elapsed_s': round(time.monotonic() - start, 1)}), flush=True)
        # Tensor references can retain model KV caches, which sample_actions resets each batch.
        del loader
        gc.collect()
    offset = 0
    for i, ep in enumerate(episodes):
        nframes = len(range(0, ep['length'], args.flow_stride))
        result['per_episode'].append({'episode_index': i, 'source_episode_index': heldout[i]['source_episode_index'],
            'frames': ep['length'], 'flow_frames': nframes,
            'flow_loss_trainer': float(sums[i, 0] / (nframes * policy.config.chunk_size * policy.config.max_action_dim)),
            'flow_mse_valid_six_joints': float(sums[i, 1] / (sums[i, 2] * 6)),
            'action_anchors': int(first_counts[i]), 'valid_action_steps': int(counts[i]),
            'action_chunk_mae_per_joint': (action_abs[i] / counts[i]).tolist(),
            'first_action_mae_per_joint': (first_abs[i] / first_counts[i]).tolist()})
        offset += nframes
    calibration = report['follower_calibration']
    # Normalized joint units -> physical degrees; gripper remains percentage points.
    scale = np.array([(calibration[name]['range_max'] - calibration[name]['range_min']) / 200 * 360 / 4096
                      for name in JOINTS[:5]] + [1.0])
    result['aggregate'] = {
        'flow_frames': offset,
        'flow_loss_trainer': float(sums[:, 0].sum() / (offset * policy.config.chunk_size * policy.config.max_action_dim)),
        'flow_mse_valid_six_joints': float(sums[:, 1].sum() / (sums[:, 2].sum() * 6)),
        'action_anchors': int(first_counts.sum()), 'valid_action_steps': int(counts.sum()),
        'action_chunk_mae_per_joint': (action_abs.sum(0) / counts.sum()).tolist(),
        'action_chunk_rmse_per_joint': np.sqrt(action_sq.sum(0) / counts.sum()).tolist(),
        'first_action_mae_per_joint': (first_abs.sum(0) / first_counts.sum()).tolist(),
        'first_action_mae_degrees_and_gripper_pp': (first_abs.sum(0) / first_counts.sum() * scale).tolist(),
        'action_chunk_mae_degrees_and_gripper_pp': (action_abs.sum(0) / counts.sum() * scale).tolist(),
        'hold_current_state_chunk_mae_per_joint': (hold_abs.sum(0) / counts.sum()).tolist(),
        'elapsed_s': round(time.monotonic() - start, 1)}
    atomic_json(args.output, result)
    print(json.dumps(result['aggregate'], indent=2), flush=True)
    print('HOLDOUT_EVALUATION_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
