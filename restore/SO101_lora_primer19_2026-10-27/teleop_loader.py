"""Load the supplied teleop bundle together with its own verified helpers."""
import importlib
from pathlib import Path
import sys
import types

EXPORT = Path(__file__).resolve().parent / 'SO101_teleop_export_2026-10-23'


def load_teleop():
    name = '_so101_recording_teleop'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(EXPORT)]
        sys.modules[name] = package
    return importlib.import_module(f'{name}.teleop_so101')


def resolve_controller(configured, serial_number, override=None):
    """Identify each controller independently; never pick an arbitrary USB port."""
    import serial.tools.list_ports
    if override:
        if not Path(override).exists():
            raise RuntimeError(f'Serial port does not exist: {override}')
        return override
    if Path(configured).exists():
        return configured
    matches = []
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if serial_number in str(port.serial_number or '') or serial_number in port.device:
            matches.append(port.device)
    # macOS can omit the serial_number field but retains it in the device name.
    if not matches:
        for port in Path('/dev').glob(f'tty.usbmodem*{serial_number}*'):
            matches.append(str(port))
    # cu.* and tty.* can be aliases of the same macOS controller.
    by_controller = {}
    for port in sorted(set(matches)):
        by_controller.setdefault(port.replace('/dev/cu.', '/dev/tty.'), port)
    matches = list(by_controller.values())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one controller {serial_number}, found {matches}. '
                           f'Available ports: {[p.device for p in ports]}. '
                           'Pass --leader-port and --follower-port explicitly.')
    return matches[0]
