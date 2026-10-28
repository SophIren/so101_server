#!/usr/bin/env python3
"""Check the saved LoRA files and normalization buffers before deployment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Path to the saved pretrained_model directory')
    args = parser.parse_args()
    root = args.checkpoint
    for filename in ('config.json', 'train_config.json', 'normalize_buffers.pt',
                     'lora_adapters/adapter_config.json', 'lora_adapters/adapter_model.safetensors'):
        if not (root / filename).is_file():
            raise RuntimeError(f'Missing adapter file: {root / filename}')
    config = json.loads((root / 'config.json').read_text())
    if config.get('empty_cameras') != 1:
        raise RuntimeError('Expected a two-camera adapter (empty_cameras=1)')
    import torch
    buffers = torch.load(root / 'normalize_buffers.pt', map_location='cpu', weights_only=True)
    # The author's saved file deduplicates target/output statistics: four
    # tensors (state mean/std and action mean/std), restored into three modules.
    for feature in ('observation.state', 'action'):
        for statistic in ('mean', 'std'):
            value = buffers.get(feature, {}).get(statistic)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (6,):
                raise RuntimeError(f'Missing/wrong-shaped normalization buffer: {feature}/{statistic}')
            if not value.isfinite().all():
                raise RuntimeError(f'Nonfinite normalization buffer: {feature}/{statistic}')
            if statistic == 'std' and not (value > 0).all():
                raise RuntimeError(f'Invalid normalization std: {feature}/{statistic}')
    print('Adapter files and finite normalization buffers: PASS')
    print('Use the same base checkpoint and the patched adapter loader; serve.py must load the adapter, '
          'not treat this adapter directory as a complete model.safetensors checkpoint.')


if __name__ == '__main__':
    main()
