#!/usr/bin/env python3
"""Generate a task LoRA config with two cameras and an explicit training split."""
import argparse
import json
from pathlib import Path

from policy_client import JOINT_ORDER


def generate(dataset, output, base, server_dataset, checkpoints, steps=20000):
    dataset = Path(dataset)
    info = json.loads((dataset / 'meta/info.json').read_text())
    if info.get('so101_split_role') == 'validation':
        raise ValueError('Refusing to train on the physically isolated validation split')
    if info.get('so101_synthetic'):
        raise ValueError('Refusing to generate a real training job for synthetic demonstrations')
    train = json.loads((dataset / 'meta/train_episodes.json').read_text())
    if not train:
        raise ValueError('Training split is empty')
    camera_keys = ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb']
    # make_policy only preserves the base's 3-slot prefix when input_features is explicit.
    # Otherwise it derives TWO features from the dataset, and prepare_images has no
    # missing key to mask even with empty_cameras=1.
    visual_shape = info.get('features', {}).get(camera_keys[0], {}).get('shape', [480, 640, 3])
    height, width = visual_shape[:2]
    inputs = {key: {'type': 'VISUAL', 'shape': [3, height, width]}
              for key in (*camera_keys, 'observation.images.right_wrist_0_rgb')}
    inputs['observation.state'] = {'type': 'STATE', 'shape': [len(JOINT_ORDER)]}
    config = {
        'policy': {'type': 'pi05', 'pretrained_path': base, 'push_to_hub': False,
                   'dtype': 'bfloat16', 'device': 'cuda', 'state_cond': True,
                   'compile_model': False, 'fuse_qkv': False, 'fuse_gate_up': False, 'empty_cameras': 1,
                   'input_features': inputs,
                   'output_features': {'action': {'type': 'ACTION', 'shape': [6]}}},
        'dataset': {'repo_id': info['repo_id'], 'root': server_dataset,
                    'episodes': train, 'video_backend': 'torchcodec'},
        'job_name': 'so101_task_lora', 'output_dir': checkpoints,
        'batch_size': 2, 'grad_accum_steps': 5, 'steps': steps,
        'num_workers': 4, 'seed': 1000, 'use_policy_training_preset': False,
        'optimizer': {'type': 'adamw', 'lr': 5e-5, 'betas': [.9, .95], 'weight_decay': 1e-10},
        'scheduler': {'type': 'cosine_decay_with_warmup', 'num_warmup_steps': min(1000, steps // 10),
                      'peak_lr': 5e-5, 'decay_lr': 2.5e-6, 'num_decay_steps': steps},
        'save_checkpoint': True, 'save_freq': min(2000, max(1, steps // 2)), 'log_freq': 10,
        'wandb': {'enable': False, 'project': 'so101_task', 'disable_artifact': True},
        # The current local deployment is synchronous; no overlap-delay augmentation.
        'max_delay_steps': 0,
        'lora': {'enable': True, 'backend': 'peft', 'r': 16, 'alpha': 16, 'dropout': 0,
                 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
                                    'down_proj', 'out_proj', 'fc1', 'fc2'],
                 'extra_trainable_modules': ['action_in_proj', 'action_out_proj', 'time_mlp_in',
                                            'time_mlp_out', 'state_proj', 'state_mlp_in', 'state_mlp_out',
                                            'embeddings', 'input_layernorm', 'post_attention_layernorm']},
    }
    # JSON is a YAML subset; using JSON avoids a local YAML dependency.
    output = Path(output)
    if output.exists():
        raise ValueError('Config already exists; choose another --output to preserve the previous run')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote YAML-compatible config: {output}; train episodes={len(train)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', default='/opt/so101/models/so101-pi05-base')
    parser.add_argument('--server-dataset', required=True)
    parser.add_argument('--checkpoints', required=True, help='New, not-yet-existing server output directory')
    parser.add_argument('--steps', type=int, default=20000)
    args = parser.parse_args(argv)
    if args.steps < 20:
        parser.error('Use at least 20 steps, including for the smoke run')
    generate(args.dataset, args.output, args.base, args.server_dataset, args.checkpoints, args.steps)


if __name__ == '__main__':
    main()
