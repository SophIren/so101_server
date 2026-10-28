#!/usr/bin/env python3
"""Download the released checkpoint and official tokenizer (no model initialization)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = 'dongyoonkim/so101-pi05-base'
MODEL_REVISION = '56325bafcca9ec3155b8bfdf655be7df17e7f09f'
TOKENIZER_REVISION = '35e4f46485b4d07967e7e9935bc3786aad50687c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/so101'))
    parser.add_argument('--tokenizer-source', default='google/paligemma-3b-pt-224')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    root = args.root.resolve()
    model = root / 'models/so101-pi05-base'
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model,
                      allow_patterns=['config.json', 'model.safetensors', 'train_config.json',
                                      'LICENSE', 'NOTICE', 'README.md'], max_workers=2)
    subprocess.run([sys.executable, str(root / 'so101-pi05-base/scripts/verify_checkpoint.py'),
                    str(model)], check=True)
    config = {'policy': {'path': str(model), 'device': 'cuda', 'dtype': 'bfloat16',
                         'compile_model': False, 'empty_cameras': 1, 'n_action_steps': 32,
                         'fuse_qkv': True, 'fuse_gate_up': True},
              'host': '127.0.0.1', 'port': 50051, 'max_workers': 4,
              'max_message_size': 16 * 1024 * 1024}
    (root / 'configs').mkdir(exist_ok=True)
    destination = root / 'configs/so101_pi05_server.json'
    if not destination.exists():
        destination.write_text(json.dumps(config, indent=2) + '\n')
    tokenizer = root / 'models/paligemma_tokenizer'
    snapshot_download(args.tokenizer_source, local_dir=tokenizer,
                      revision=(TOKENIZER_REVISION if args.tokenizer_source ==
                                'google/paligemma-3b-pt-224' else None),
                      allow_patterns=['*.json', 'tokenizer.model'], max_workers=2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    print('Tokenizer:', type(tok).__name__, 'vocab size:', len(tok))
    print('Assets verified. Serve config:', destination)


if __name__ == '__main__':
    main()
