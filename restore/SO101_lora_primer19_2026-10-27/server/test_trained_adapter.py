#!/usr/bin/env python3
"""Load a saved task adapter on its base and predict from real offline RGB/state."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    args = parser.parse_args()
    from vlash.configs import ServerConfig
    from vlash.so101_policy_loader import load_policy
    import vlash.datasets.compat  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    cfg = ServerConfig(policy={'path': str(args.checkpoint.resolve()), 'device': 'cuda',
                               'empty_cameras': 1, 'n_action_steps': 32,
                               'compile_model': False, 'fuse_qkv': False, 'fuse_gate_up': False})
    dataset = LeRobotDataset('PriMeR19/SO-101-teleop', root=args.dataset,
                            episodes=[0], video_backend='torchcodec')
    sample = dataset[len(dataset)//2]
    batch = {key: value.unsqueeze(0).cuda() for key, value in sample.items()
             if key == 'observation.state' or key.startswith('observation.images.')}
    batch['task'] = [sample['task']]
    policy = load_policy(cfg.policy)
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    if actions.shape != (1, 32, 6) or not torch.isfinite(actions).all():
        raise RuntimeError(f'Invalid adapter prediction: {actions.shape}')
    print(json.dumps({'checkpoint': str(args.checkpoint.resolve()), 'task': sample['task'],
                      'actions_shape': list(actions.shape), 'all_finite': True,
                      'first_action': actions[0, 0].float().cpu().tolist(),
                      'hardware_accessed': False}, indent=2))
    print('TRAINED_ADAPTER_INFERENCE_PASS')


if __name__ == '__main__':
    main()
