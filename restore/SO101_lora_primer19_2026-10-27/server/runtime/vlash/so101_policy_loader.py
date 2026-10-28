"""SO101 base/adapter loader for the restored remote server.

Unlike the original robot run loop, adapters use the base recorded by training
and missing normalization statistics stop startup. No robot drivers are used.
"""
import json
import logging
import os
from pathlib import Path

import torch


def check_normalization(policy):
    for module, feature in ((policy.normalize_inputs, 'observation_state'),
                            (policy.normalize_targets, 'action'),
                            (policy.unnormalize_outputs, 'action')):
        buffer = getattr(module, f'buffer_{feature}')
        if (buffer.mean.shape != (6,) or buffer.std.shape != (6,)
                or not torch.isfinite(buffer.mean).all()
                or not torch.isfinite(buffer.std).all() or not (buffer.std > 0).all()):
            raise ValueError(f'Invalid normalization buffers for {feature}')


def load_policy(cfg):
    from vlash.policies.factory import get_policy_class
    path = Path(cfg.pretrained_path).resolve()
    cls = get_policy_class(cfg.type)
    if (path / 'lora_adapters').is_dir():
        from vlash.configs.train_config import LoRAConfig
        from vlash.lora.apply import apply_lora
        from vlash.lora.checkpoint import load_lora_adapters, merge_lora_into_base
        if path.name != 'pretrained_model':
            raise ValueError('Adapter path must end in pretrained_model')
        train = json.loads((path / 'train_config.json').read_text())
        base = Path(os.environ.get('SO101_BASE_CHECKPOINT') or
                    train['policy']['pretrained_path']).expanduser()
        if not (base / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Training base is missing: {base}. '
                                    'Set SO101_BASE_CHECKPOINT only to the same base weights.')
        stats = torch.load(path / 'normalize_buffers.pt', map_location='cpu', weights_only=True)
        for key in ('observation.state', 'action'):
            for field in ('mean', 'std'):
                value = stats[key][field]
                if value.shape != (6,) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid adapter normalization: {key}/{field}')
            if not (stats[key]['std'] > 0).all():
                raise ValueError(f'Nonpositive adapter std: {key}')
        # LoRA requires the unfused projections used during training.
        cfg.fuse_qkv = cfg.fuse_gate_up = False
        policy = cls.from_pretrained(str(base), config=cfg)
        lora = LoRAConfig(**train['lora'])
        apply_lora(lora, policy, verbose=True)
        if not load_lora_adapters(policy, path.parent):
            raise RuntimeError(f'Adapter was not loaded: {path}')
        merge_lora_into_base(policy, verbose=True)
        with torch.no_grad():
            for module, feature, key in (
                    (policy.normalize_inputs, 'observation_state', 'observation.state'),
                    (policy.normalize_targets, 'action', 'action'),
                    (policy.unnormalize_outputs, 'action', 'action')):
                buffer = getattr(module, f'buffer_{feature}')
                buffer.mean.copy_(stats[key]['mean'])
                buffer.std.copy_(stats[key]['std'])
        logging.info('Adapter merged onto %s; normalization restored from %s', base, path)
    else:
        policy = cls.from_pretrained(str(path), config=cfg)
    check_normalization(policy)
    policy.eval()
    return policy
