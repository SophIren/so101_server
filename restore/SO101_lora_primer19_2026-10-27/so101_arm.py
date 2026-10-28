"""Thin wrapper over the Feetech bus for the SO-101 follower.

Only absolute position goals are written; every goal is checked against the
calibrated range before it is sent. Nothing here enables torque implicitly.
"""
import json
import os
import sys
import time
from pathlib import Path

import serial

from serial_protocol import exchange
from verified_telemetry import read_health

ROOT = Path(__file__).resolve().parent

REG_TORQUE_ENABLE = 40
REG_ACCELERATION = 41
REG_GOAL_POSITION = 42
REG_PRESENT_POSITION = 56
REG_GOAL_SPEED = 46
REG_PROTECTION_CURRENT = 48
BROADCAST_ID = 254


def resolve_port(configured=None):
    """Find the serial port of this arm.

    The bundled config names the Linux udev symlink /dev/so101_follower. On macOS
    the same controller appears as /dev/tty.usbmodem<SERIAL>. The serial number is
    the one recorded in robot_setup/99-so101.rules.
    """
    import glob
    import re

    import serial.tools.list_ports

    rules = (ROOT / 'robot_setup/99-so101.rules').read_text()
    match = re.search(r'ATTRS\{serial\}=="([0-9A-Za-z]+)"', rules)
    serial_number = match.group(1) if match else None

    candidates = [configured, os.environ.get('SO101_PORT')]
    if configured and os.path.exists(configured):
        return configured
    if sys.platform == 'darwin':
        if serial_number:
            candidates.append(f'/dev/tty.usbmodem{serial_number}1')
            candidates.append(f'/dev/tty.usbmodem{serial_number}')
        candidates.extend(sorted(glob.glob('/dev/tty.usbmodem*')))
    for port in candidates:
        if port and os.path.exists(port):
            return port
    available = [p.device for p in serial.tools.list_ports.comports()]
    raise RuntimeError(
        f'No serial port for the arm. Tried {candidates}.\n'
        f'Ports present: {available or "none"}. '
        'Pass --port-name, or set SO101_PORT.')


class Arm:
    def __init__(self, port=None, calibration=None, config=None, baudrate=None):
        self.config = config or json.loads((ROOT / 'robot_setup/robot_config.json').read_text())
        self.calibration = calibration or json.loads(
            (ROOT / (self.config['joint_calibration'])).read_text())
        self.port = port or self.config['serial_port']
        self.baudrate = baudrate or self.config['baudrate']
        self.bus = None
        self.log = {'retries': []}

    # ------------------------------------------------------------- transport
    def open(self):
        self.port = resolve_port(self.port)
        self.bus = serial.Serial(self.port, self.baudrate, timeout=.15, exclusive=True)
        return self

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def _tx(self, motor_id, command, parameters, length):
        def on_retry(event):
            self.log['retries'].append(event)
        return exchange(self.bus, motor_id, command, parameters, length, on_retry=on_retry)

    def read(self, motor_id, address, size):
        return int.from_bytes(self._tx(motor_id, 2, [address, size], size + 6), 'little')

    def write(self, motor_id, address, size, value):
        self._tx(motor_id, 3, [address] + list(int(value).to_bytes(size, 'little')), 6)
        if self.read(motor_id, address, size) != value:
            raise RuntimeError(f'ID {motor_id}: write verification failed at {address}')

    # ------------------------------------------------------------- observation
    def positions(self):
        return {name: self.read(c['id'], REG_PRESENT_POSITION, 2)
                for name, c in self.calibration.items()}

    def health(self):
        return {name: read_health(c['id'], self._tx) for name, c in self.calibration.items()}

    def torque_state(self):
        return {name: self.read(c['id'], REG_TORQUE_ENABLE, 1)
                for name, c in self.calibration.items()}

    def in_range(self, pose):
        return all(c['range_min'] <= pose[n] <= c['range_max']
                   for n, c in self.calibration.items())

    # ------------------------------------------------------------- command
    def enable(self, acceleration=0, speed=65):
        for name, c in self.calibration.items():
            motor_id = c['id']
            self.write(motor_id, REG_ACCELERATION, 1, acceleration)
            self.write(motor_id, REG_GOAL_SPEED, 2, speed)
        self.hold(self.positions())
        for c in self.calibration.values():
            self.write(c['id'], REG_TORQUE_ENABLE, 1, 1)

    def disable(self):
        for c in self.calibration.values():
            self.write(c['id'], REG_TORQUE_ENABLE, 1, 0)

    def hold(self, pose):
        """Send one synchronised goal packet for all six motors."""
        if not self.in_range(pose):
            raise ValueError(f'Goal outside calibrated range: {pose}')
        parameters = [REG_GOAL_POSITION, 2]
        for name, c in self.calibration.items():
            value = int(pose[name])
            parameters.extend([c['id'], value & 255, value >> 8])
        packet = [BROADCAST_ID, len(parameters) + 2, 131] + parameters
        self.bus.write(bytes([255, 255] + packet + [(~sum(packet)) & 255]))
        self.bus.flush()

    def hold_and_check(self, pose, tolerance=None, abort_tracking=None):
        """Command a goal, wait for it to settle, verify telemetry."""
        motion = self.config['motion']
        tolerance = motion['position_tolerance_ticks'] if tolerance is None else tolerance
        abort_tracking = motion['abort_tracking_error_ticks'] if abort_tracking is None else abort_tracking
        self.hold(pose)
        deadline = time.monotonic() + 2.0
        actual = None
        while time.monotonic() < deadline:
            actual = self.positions()
            if max(abs(actual[n] - pose[n]) for n in self.calibration) <= tolerance:
                return actual
            time.sleep(.05)
        errors = {n: int(actual[n] - pose[n]) for n in self.calibration}
        worst = max(errors, key=lambda n: abs(errors[n]))
        if abs(errors[worst]) > abort_tracking:
            raise RuntimeError(f'{worst}: tracking error {errors[worst]} ticks')
        return actual
