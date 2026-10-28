"""VLASH remote_inference wire format, without importing VLASH or LeRobot.

Matches commit 093d70b1fecff37b868fd124a6ae8b5691df2a39:
https://github.com/mit-han-lab/vlash/tree/093d70b1fecff37b868fd124a6ae8b5691df2a39/vlash/transport
The server uses pickle for CPU observation tensors and CUDA action tensors.
"""
import io
import pickle

import numpy as np

CHUNK_SIZE = 2 * 1024 * 1024


def load_schema():
    """Built-in services.proto descriptor; no reflection or protoc needed."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    proto = descriptor_pb2.FileDescriptorProto(
        name='vlash_transport.proto', package='transport', syntax='proto3')
    enum = proto.enum_type.add(name='TransferState')
    for number, name in enumerate(('UNKNOWN', 'BEGIN', 'MIDDLE', 'END')):
        enum.value.add(name=f'TRANSFER_{name}', number=number)
    observation = proto.message_type.add(name='Observation')
    observation.field.add(name='transfer_state', number=1, type=14,
                          type_name='.transport.TransferState')
    observation.field.add(name='data', number=2, type=12)
    proto.message_type.add(name='ActionChunk').field.add(name='data', number=1, type=12)
    proto.message_type.add(name='Empty')
    service = proto.service.add(name='RemoteInference')
    for name, request, response, streaming in (
            ('LaunchInference', 'Observation', 'Empty', True),
            ('GetActionChunk', 'Empty', 'ActionChunk', False),
            ('Ready', 'Empty', 'Empty', False)):
        service.method.add(name=name, input_type=f'.transport.{request}',
                           output_type=f'.transport.{response}', client_streaming=streaming)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    return pool, [('transport.RemoteInference', m.name) for m in service.method]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            'VLASH transport requires PyTorch. Install requirements-policy.txt '
            'in the Python environment running run_policy.py.') from error
    return torch


def encode_observation(observation, prompt):
    """OpenCV BGR uint8 HWC -> RGB float32 BCHW; state -> float32 [1,6]."""
    torch = _torch()
    state = np.asarray(observation['observation.state'], dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError('observation.state must contain six finite values')
    batch = {'observation.state': torch.from_numpy(state.copy()).unsqueeze(0),
             'task': [prompt], 'robot_type': 'so101_follower'}
    for key, value in observation.items():
        if not key.startswith('observation.images.'):
            continue
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f'{key}: expected a uint8 HWC BGR camera frame')
        rgb = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1))
        batch[key] = torch.from_numpy(rgb).to(torch.float32).div_(255).unsqueeze(0)
    return pickle.dumps(batch, protocol=4)


def observation_chunks(payload, message_class):
    """Match VLASH send_bytes_in_chunks, including END for a single chunk."""
    for offset in range(0, len(payload), CHUNK_SIZE):
        end = offset + CHUNK_SIZE
        state = 3 if end >= len(payload) else (1 if offset == 0 else 2)
        yield message_class(transfer_state=state, data=payload[offset:end])


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Plain pickle.loads would restore CUDA storage on the Mac and fail.
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: _torch().load(io.BytesIO(data), map_location='cpu',
                                              weights_only=True)
        return super().find_class(module, name)


def decode_actions(payload):
    # This is the server's pickle protocol; connect only to a trusted server.
    if not payload:
        raise ValueError('VLASH returned an empty action payload')
    actions = _CPUUnpickler(io.BytesIO(payload)).load()
    torch = _torch()
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().to(device='cpu', dtype=torch.float32).numpy()
    return validate_actions(actions)


def validate_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 6:
        raise ValueError(f'Expected a nonempty action chunk [N,6], got {actions.shape}')
    if not np.isfinite(actions).all():
        raise ValueError('Action chunk contains NaN or infinity')
    return actions
