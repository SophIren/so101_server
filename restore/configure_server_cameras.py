#!/usr/bin/env python3
"""Run on the server: configure and verify two images plus one masked slot."""
import argparse
import datetime
import json
from pathlib import Path
from types import SimpleNamespace


def verify_images():
    import torch
    import vlash.policies.pi05.modeling_pi05 as modeling
    policy = modeling.PI05Policy.__new__(modeling.PI05Policy)
    torch.nn.Module.__init__(policy)
    keys = ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
            'observation.images.right_wrist_0_rgb']
    policy.config = SimpleNamespace(image_features={key: None for key in keys},
                                    image_resolution=(224, 224), empty_cameras=1)
    batch = {key: torch.zeros(1, 3, 480, 640) for key in keys[:2]}
    images, masks = policy.prepare_images(batch)
    values = [mask.tolist() for mask in masks]
    if len(images) != 3 or values != [[True], [True], [False]]:
        raise RuntimeError(f'Two-camera masking is incorrect in {modeling.__file__}: {values}')
    print('Policy implementation:', modeling.__file__)
    print('Two-camera image masks:', values, 'PASS; weights were not loaded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/opt/so101'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import yaml
    restored = args.project / 'configs/so101_pi05_server.json'
    path = args.config or (restored if restored.exists() else
                           args.project / 'vlash/examples/serve/so101_pi05_server.yaml')
    original = path.read_text()
    config = yaml.safe_load(original)
    checkpoint = Path(config['policy'].get('path') or config['policy']['pretrained_path'])
    model = json.loads((checkpoint / 'config.json').read_text())
    keys = [key for key in model['input_features'] if key.startswith('observation.images.')]
    if keys != ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
                'observation.images.right_wrist_0_rgb']:
        raise RuntimeError(f'Unexpected checkpoint camera order: {keys}')
    verify_images()
    if args.check_only:
        if config['policy'].get('empty_cameras', model.get('empty_cameras', 0)) != 1:
            raise RuntimeError('Set policy.empty_cameras: 1 in the serve YAML and restart the server')
        print('Serve YAML: two cameras configured. A running server must be restarted after a YAML change.')
        return
    if config['policy'].get('empty_cameras') != 1:
        backup = path.with_name(path.name + '.bak.' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.write_text(original)
        config['policy']['empty_cameras'] = 1
        path.write_text(json.dumps(config, indent=2) + '\n' if path.suffix == '.json' else
                        yaml.safe_dump(config, sort_keys=False))
        print('Backup:', backup)
    print('Configured:', path)
    print('Keep all three image_features in the checkpoint. Send only base + left wrist.')
    print('Restart in the usual server environment: vlash serve', path)


if __name__ == '__main__':
    main()
