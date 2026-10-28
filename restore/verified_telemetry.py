"""Confirm anomalous servo temperatures using separate full telemetry reads."""
import time


def read_health(motor_id, exchange, limit=45):
    def block():
        data = exchange(motor_id, 2, [56, 15], 21)
        if len(data) != 15:
            raise RuntimeError(f'ID {motor_id}: invalid telemetry block length')
        result = {
            'position': int.from_bytes(data[0:2], 'little'),
            'load': int.from_bytes(data[4:6], 'little') & 1023,
            'voltage': data[6],
            'temp': data[7],
            'status': data[9],
        }
        if result['status'] != 0:
            raise RuntimeError(f'ID {motor_id}: hardware status {result["status"]}')
        return result

    first = block()
    if first['temp'] < limit:
        return first
    confirmations = []
    for _ in range(3):
        # The caller sends no new movement goals during confirmation.
        time.sleep(.02)
        confirmations.append(block())
    temperatures = [row['temp'] for row in confirmations]
    if max(temperatures) >= limit or max(temperatures) - min(temperatures) > 2:
        raise RuntimeError(
            f'ID {motor_id}: temperature anomaly not cleared: '
            f'{first["temp"]}, confirmations={temperatures}'
        )
    if any(not 105 <= row['voltage'] <= 135 or row['load'] >= 850
           for row in [first, *confirmations]):
        raise RuntimeError(f'ID {motor_id}: unsafe voltage or load during confirmation')
    result = confirmations[-1]
    result['confirmed_temperature_outlier'] = first['temp']
    return result
