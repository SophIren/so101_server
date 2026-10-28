#!/usr/bin/env python3
"""Inspect SO-101 calibration; optionally apply documented gripper protections."""
import argparse
import datetime
import json
from pathlib import Path

import serial

from verified_telemetry import read_health

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'robot_setup/robot_config.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-gripper-protection', action='store_true')
    parser.add_argument('--port', default=None,
                        help='serial port; by default it is discovered for this platform')
    args = parser.parse_args()
    # The Linux udev symlink does not exist on macOS, so resolve it per platform.
    from so101_arm import resolve_port
    port = resolve_port(args.port)
    cal = json.loads((ROOT / 'so101_follower_calibration.json').read_text())
    directory = ROOT / 'robot_setup'
    directory.mkdir(exist_ok=True)
    report = {'timestamp': datetime.datetime.now().isoformat(), 'motors': {}, 'changes': []}
    with serial.Serial(port, 1000000, timeout=.3, exclusive=True) as bus:
        def tx(i, command, parameters, length):
            bus.reset_input_buffer()
            packet = [i, len(parameters) + 2, command] + parameters
            bus.write(bytes([255, 255] + packet + [(~sum(packet)) & 255]))
            bus.flush()
            response = bus.read(length)
            if (len(response) != length or response[:3] != bytes([255, 255, i])
                    or response[3] != length - 4 or response[4] != 0
                    or sum(response[2:]) % 256 != 255):
                raise RuntimeError(f'ID {i}: invalid response {response.hex()}')
            return response[5:-1]

        def read(i, address, size):
            return int.from_bytes(tx(i, 2, [address, size], size + 6), 'little')

        def write(i, address, size, value):
            tx(i, 3, [address, *value.to_bytes(size, 'little')], 6)
            if read(i, address, size) != value:
                raise RuntimeError(f'ID {i}: write verification failed at {address}')

        for name, c in cal.items():
            i = c['id']
            registers = {str(a): read(i, a, size) for a, size in (
                (3, 2), (5, 1), (6, 1), (9, 2), (11, 2), (13, 1), (16, 2),
                (21, 1), (22, 1), (23, 1), (28, 2), (31, 2), (33, 1),
                (36, 1), (40, 1), (48, 2), (55, 1))}
            assert registers['40'] == 0, f'{name}: must be resting with torque off'
            assert registers['3'] == 777 and registers['5'] == i and registers['6'] == 0
            assert registers['9'] == c['range_min'] and registers['11'] == c['range_max']
            offset = c['homing_offset']
            assert registers['31'] == abs(offset) | (2048 if offset < 0 else 0)
            assert registers['33'] == 0
            report['motors'][name] = {'id': i, 'registers_before': registers,
                                     'health': read_health(i, tx), 'calibration_matches': True}
        backup = directory / ('registers_before_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        backup.write_text(json.dumps(report, indent=2) + '\n')
        if args.apply_gripper_protection:
            i = cal['gripper']['id']
            original_lock = read(i, 55, 1)
            write(i, 55, 1, 0)
            try:
                for address, size, value in ((16, 2, 500), (28, 2, 250), (36, 1, 25)):
                    previous = read(i, address, size)
                    write(i, address, size, value)
                    report['changes'].append({'motor': 'gripper', 'address': address,
                                              'before': previous, 'after': value})
            finally:
                write(i, 55, 1, original_lock)
        for name, c in cal.items():
            i = c['id']
            report['motors'][name]['torque_after'] = read(i, 40, 1)
            assert report['motors'][name]['torque_after'] == 0
        report['passed'] = True
    (directory / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'changes': report['changes'],
                      'temperatures_C': {n: d['health']['temp'] for n, d in report['motors'].items()}}, indent=2))


if __name__ == '__main__':
    main()
