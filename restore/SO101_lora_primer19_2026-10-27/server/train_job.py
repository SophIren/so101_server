#!/usr/bin/env python3
"""Launch and inspect one detached VLASH training process; no Torch import needed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from manage import process_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop'])
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path, default=Path('/opt/so101'))
    args = parser.parse_args()
    run_dir, root = args.run_dir.resolve(), args.root.resolve()
    metadata = run_dir / 'job.json'
    record = json.loads(metadata.read_text()) if metadata.exists() else None
    running = bool(record and process_token(record['pid']) == record['starttime'])
    if args.command == 'status':
        print(json.dumps({'running': running, **(record or {})}, indent=2))
        if (run_dir / 'train.log').exists():
            print('\nRecent log:')
            print('\n'.join((run_dir / 'train.log').read_text(errors='replace').splitlines()[-15:]))
        return
    if args.command == 'stop':
        if running:
            os.killpg(record['pid'], signal.SIGTERM)
            print('Termination sent; resume uses the last complete checkpoint.')
        else:
            print('Training process is not running.')
        return
    if running:
        raise RuntimeError(f'Already running: PID {record["pid"]}')
    if args.config is None:
        parser.error('start requires --config')
    config = args.config.resolve()
    content = json.loads(config.read_text())
    if not content.get('lora', {}).get('enable'):
        raise ValueError('This launcher requires LoRA enabled')
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'PATH': str(root / 'env/bin') + ':' + env['PATH'],
                'LD_LIBRARY_PATH': str(root / 'env/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
                'HF_HOME': str(root / 'cache/huggingface'),
                'VLASH_PALIGEMMA_PATH': str(root / 'models/paligemma_tokenizer'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                'VLASH_SAVE_LOCAL_RESUME': '1'})
    with (run_dir / 'train.log').open('ab') as output:
        proc = subprocess.Popen([str(root / 'env/bin/python'), '-u', '-m', 'vlash.train',
                                 f'--config_path={config}'], cwd=root / 'restore', env=env,
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': proc.pid, 'starttime': process_token(proc.pid), 'config': str(config),
              'output_dir': content['output_dir'], 'log': str(run_dir / 'train.log')}
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError(f'Training exited with {proc.returncode}; read {run_dir / "train.log"}')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
