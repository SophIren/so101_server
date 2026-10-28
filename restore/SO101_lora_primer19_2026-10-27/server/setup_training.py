#!/usr/bin/env python3
"""Write an explicit, reproducible LoRA job for a validated two-camera dataset."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_training_config import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--save-freq', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--base', default='/opt/so101/models/so101-pi05-base')
    args = parser.parse_args()
    path = args.run_dir.resolve() / 'lora.json'
    dataset = args.dataset.resolve()
    generate(dataset, path, args.base, str(dataset), str(args.output_dir.resolve()), args.steps)
    config = json.loads(path.read_text())
    config.update({'batch_size': args.batch_size, 'grad_accum_steps': args.grad_accum,
                   'num_workers': 4, 'save_freq': args.save_freq, 'job_name': 'primer19_bag_lora'})
    path.write_text(json.dumps(config, indent=2) + '\n')
    report = json.loads((dataset / 'conversion_report.json').read_text())
    print(json.dumps({'config': str(path), 'steps': args.steps, 'fps': 30,
                      'effective_batch': args.batch_size * args.grad_accum,
                      'train_frames': report['train_frames'], 'val_frames': report['val_frames'],
                      'expected_epochs': args.steps * args.batch_size * args.grad_accum / report['train_frames']}, indent=2))


if __name__ == '__main__':
    main()
