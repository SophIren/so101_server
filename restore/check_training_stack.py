#!/usr/bin/env python3
"""Check a generated LoRA job in the intended server Python environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    import vlash.train as training
    source = Path(training.__file__).read_text()
    if 'normalize_buffers.pt' not in source:
        raise RuntimeError(
            f'{training.__file__} does not save LoRA normalization buffers. '
            'Use the training-docker environment/patched training files from so101-pi05-base; '
            'do not deploy a stock VLASH adapter without its normalization statistics.')
    from validate_dataset import validate, check_lerobot
    from configure_server_cameras import verify_images
    info = validate(config['dataset']['root'])
    if info.get('so101_synthetic'):
        raise RuntimeError('Synthetic dataset must not be used to train the real task')
    expected = json.loads((Path(config['dataset']['root']) / 'meta/train_episodes.json').read_text())
    if config['dataset']['episodes'] != expected:
        raise RuntimeError('The training config must explicitly select only training episodes')
    if config['policy']['empty_cameras'] != 1:
        raise RuntimeError('Two cameras require empty_cameras=1')
    expected_cameras = ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
                        'observation.images.right_wrist_0_rgb']
    declared = [k for k in config['policy'].get('input_features', {}) if k.startswith('observation.images.')]
    if declared != expected_cameras:
        raise RuntimeError('Declare all THREE base camera slots in policy.input_features, '
                           'while keeping only TWO real camera streams in the dataset.')
    base = Path(config['policy']['pretrained_path'])
    if not (base / 'model.safetensors').is_file() or not (base / 'config.json').is_file():
        raise RuntimeError('Base checkpoint files are missing')
    if Path(config['output_dir']).exists():
        raise RuntimeError('Training output_dir must not already exist; choose a new run directory')
    check_lerobot(config['dataset']['root'], config['dataset']['repo_id'])
    verify_images()
    print('Training stack:', training.__file__)
    print('READY for a 20-step smoke run. No GPU training was started.')


if __name__ == '__main__':
    main()
