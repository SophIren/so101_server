#!/usr/bin/env python3
"""Evaluate the base and final adapter, writing independent persistent results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/opt/so101')
    env = os.environ.copy()
    env.update({'PATH': str(root / 'env/bin') + ':' + env.get('PATH', ''),
                'LD_LIBRARY_PATH': str(root / 'env/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
                'HF_HOME': str(root / 'cache/huggingface'),
                'VLASH_PALIGEMMA_PATH': str(root / 'models/paligemma_tokenizer'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
    output = args.run_root / 'evaluation'
    output.mkdir(exist_ok=True)
    for name, checkpoint in [
        ('base', root / 'models/so101-pi05-base'),
        ('step_020000', args.run_root / 'full/train/checkpoints/020000/pretrained_model'),
    ]:
        destination = output / (name + '.json')
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing['checkpoint'] != str(checkpoint.resolve()) or 'aggregate' not in existing:
                raise FileExistsError(f'Refusing to overwrite a different evaluation: {destination}')
            print('ALREADY_COMPLETE', name, flush=True)
            continue
        command = [sys.executable, '-u', str(Path(__file__).with_name('evaluate_holdout.py')),
                   '--checkpoint', str(checkpoint), '--dataset',
                   str(root / 'datasets/primer19_bag_validation_v21'), '--output', str(destination)]
        print('STARTING', name, flush=True)
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('FINISHED', name, flush=True)


if __name__ == '__main__':
    main()
