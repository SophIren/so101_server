#!/usr/bin/env python3
"""Start/status/stop the dedicated server; uses only the Python standard library."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_token(pid):
    try:
        # starttime is field 22; comm may contain spaces and parentheses.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop'])
    parser.add_argument('--root', type=Path, default=Path('/opt/so101'))
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = root / 'server-process.json'
    record = json.loads(metadata.read_text()) if metadata.exists() else None
    running = bool(record and process_token(record['pid']) == record['starttime'])
    if args.command == 'status':
        print(json.dumps({'running': running, **(record or {})}, indent=2))
        return
    if args.command == 'stop':
        if running:
            os.killpg(record['pid'], signal.SIGTERM)
            for _ in range(40):
                if process_token(record['pid']) != record['starttime']:
                    break
                time.sleep(.25)
            else:
                raise RuntimeError('Server did not stop within 10 seconds; inspect before retrying')
        metadata.unlink(missing_ok=True)
        print('Server stopped')
        return
    if running:
        print('Server process already running:', record['pid'], 'log:', record['log'])
        return
    config = (args.config or root / 'configs/so101_pi05_server.json').resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    env = os.environ.copy()
    env.update({'HF_HOME': str(root / 'cache/huggingface'),
                'VLASH_PALIGEMMA_PATH': str(root / 'models/paligemma_tokenizer'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8',
                'LD_LIBRARY_PATH': str(root / 'env/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
                'PATH': str(root / 'env/bin') + ':' + env['PATH']})
    log = root / 'logs' / ('serve_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('ab') as output:
        proc = subprocess.Popen([str(root / 'env/bin/python'), '-m', 'vlash.serve',
                                 f'--config_path={config}'], cwd=root, env=env,
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': proc.pid, 'starttime': process_token(proc.pid),
              'log': str(log), 'config': str(config)}
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError(f'Server exited with {proc.returncode}; read {log}')
    print('Server process started (model is still loading):', proc.pid)
    print('Log:', log)


if __name__ == '__main__':
    main()
