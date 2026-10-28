"""Emulated Feetech bus for testing so101_arm.py and the control loop offline.

Models registers, torque, position goals and telemetry. Positions move a few
counts per poll so tracking checks and clamps behave like the real arm.
"""
import struct
import time


class FakeBus:
    def __init__(self, positions=None, calibration=None, latency=0.0,
                 stop_responding=None, hot=None, stuck=None):
        self.positions = dict(positions or {})
        self.calibration = calibration or {}
        self.registers = {c['id']: {} for c in self.calibration.values()}
        self.goals = dict(self.positions)
        self.torque = {i: 0 for i in self.registers}
        self.latency = latency
        self.stop_responding = stop_responding or set()
        self.hot = hot or set()
        self.stuck = stuck or set()
        self.writes = []
        self._pending = bytearray()
        self.closed = False

    # serial.Serial surface ------------------------------------------------
    def reset_input_buffer(self):
        self._pending.clear()

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def read(self, length):
        if self.latency:
            time.sleep(self.latency)
        data = bytes(self._pending[:length])
        del self._pending[:length]
        return data

    def write(self, packet):
        packet = bytearray(packet)
        if len(packet) < 6:
            return
        motor_id = packet[2]
        command = packet[4]
        length = packet[3]
        # length counts the instruction and the checksum on top of the parameters
        parameters = list(packet[5:5 + length - 2])
        if motor_id == 254:
            self._broadcast(command, parameters)
            return
        if motor_id in self.stop_responding:
            return
        self._advance()
        if command == 2:
            address, size = parameters[0], parameters[1]
            if address == 56 and size == 15:
                payload = self.health_block(motor_id)
                self._respond(motor_id, payload)
                return
            value = self._read_register(motor_id, address, size)
            self._respond(motor_id, (int(value) & ((1 << (8 * size)) - 1)).to_bytes(size, 'little'))
        elif command == 3:
            address = parameters[0]
            value = int.from_bytes(bytes(parameters[1:]), 'little')
            size = len(parameters) - 1
            self._write_register(motor_id, address, size, value)
            self._respond(motor_id, b'')

    # emulation ------------------------------------------------------------
    def _respond(self, motor_id, payload):
        body = [motor_id, len(payload) + 2, 0, *payload]
        self._pending.extend(bytes([255, 255] + body + [(~sum(body)) & 255]))

    def _read_register(self, motor_id, address, size):
        names = {c['id']: n for n, c in self.calibration.items()}
        name = names.get(motor_id)
        if address == 56:
            return int(self.positions.get(name, 2048))
        if address == 40:
            return self.torque.get(motor_id, 0)
        if address == 3:
            return 777
        if address == 5:
            return motor_id
        if address == 9:
            return int(self.calibration[name]['range_min'])
        if address == 11:
            return int(self.calibration[name]['range_max'])
        if address == 33:
            return 0
        if address == 31:
            offset = self.calibration[name]['homing_offset']
            return abs(offset) | (2048 if offset < 0 else 0)
        return self.registers[motor_id].get(address, 0)

    def _write_register(self, motor_id, address, size, value):
        self.writes.append((motor_id, address, value))
        if address == 40:
            self.torque[motor_id] = value
        elif address == 42:
            for name, c in self.calibration.items():
                if c['id'] == motor_id:
                    if name not in self.stuck:
                        self.goals[name] = value
        else:
            self.registers[motor_id][address] = value

    def _broadcast(self, command, parameters):
        if command != 131 or parameters[0] != 42:
            return
        self._advance()
        pairs = parameters[2:]
        for index in range(0, len(pairs), 3):
            motor_id = pairs[index]
            value = pairs[index + 1] | (pairs[index + 2] << 8)
            for name, c in self.calibration.items():
                if c['id'] == motor_id and name not in self.stuck:
                    self.goals[name] = value

    def _advance(self, step=8):
        """Move toward goals while torque is on."""
        for name, c in self.calibration.items():
            if not self.torque.get(c['id'], 0):
                continue
            current = self.positions.get(name, 2048)
            goal = self.goals.get(name, current)
            if abs(goal - current) <= step:
                self.positions[name] = goal
            else:
                self.positions[name] = current + step * (1 if goal > current else -1)

    def health_block(self, motor_id):
        """15-byte telemetry block from address 56, Feetech layout."""
        names = {c['id']: n for n, c in self.calibration.items()}
        name = names[motor_id]
        position = int(self.positions.get(name, 2048)).to_bytes(2, 'little')
        speed = (0).to_bytes(2, 'little')
        load = (0).to_bytes(2, 'little')
        voltage = 120
        temperature = 60 if name in self.hot else 30
        # position, speed, load, voltage, temperature, current, status, reserved...
        block = position + speed + load + bytes(
            [voltage, temperature, 0, 0, 0, 0, 0, 0, 0])
        return block

    def _respond_health(self, motor_id):
        data = self.health_block(motor_id)
        body = [motor_id, len(data) + 2, 0, *data]
        self._pending.extend(bytes([255, 255] + body + [(~sum(body)) & 255]))
