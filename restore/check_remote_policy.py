#!/usr/bin/env python3
"""Test real remote inference using two synthetic frames; opens no hardware."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from policy_client import PolicyClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--prompt', default='take the box')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error('--requests must be positive')
    client = PolicyClient(args.host, args.port, timeout=args.timeout)
    results = []
    try:
        client.connect()
        observation = {'observation.state': np.zeros(6, dtype=np.float32),
                       'observation.images.base_0_rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                       'observation.images.left_wrist_0_rgb': np.full((480, 640, 3), 127, dtype=np.uint8)}
        for index in range(args.requests):
            start = time.perf_counter()
            actions = client.infer(observation, args.prompt)
            seconds = time.perf_counter() - start
            if actions.shape != (32, 6) or not np.isfinite(actions).all():
                raise RuntimeError(f'Expected 32 finite 6-joint actions, received {actions.shape}')
            print(f'Request {index+1}: {actions.shape}, {seconds:.3f}s, finite=PASS')
            print('First action:', np.round(actions[0], 3))
            results.append({'seconds': seconds, 'shape': list(actions.shape),
                            'actions': actions.tolist()})
    finally:
        client.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'synthetic_inputs': True, 'hardware_accessed': False,
                                         'address': client.address, 'prompt': args.prompt,
                                         'results': results}, indent=2) + '\n')
    print('PASS: real model inference; synthetic images/state; no hardware accessed.')


if __name__ == '__main__':
    main()
