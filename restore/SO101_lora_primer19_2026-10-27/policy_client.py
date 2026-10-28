"""VLASH client: Ready -> LaunchInference(stream) -> GetActionChunk.

The transport.RemoteInference schema is built in for servers without reflection.
Explicit .proto / descriptor sets and reflected unary inference remain supported.
SO-101 units: five arm joints in [-100,100], gripper in [0,100].
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

JOINT_ORDER = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']
ARM_JOINTS = JOINT_ORDER[:5]


def _is_repeated(field):
    """protobuf >= 5 exposes is_repeated; older versions use label."""
    repeated = getattr(field, 'is_repeated', None)
    if repeated is not None:
        return bool(repeated)
    return field.label == field.LABEL_REPEATED


def load_calibration(path=None):
    return json.loads(Path(path or ROOT / 'so101_follower_calibration.json').read_text())


# --------------------------------------------------------------- unit mapping

# lerobot normalisation for the SO-101: body joints are RANGE_M100_100 over the
# calibrated range, the gripper is RANGE_0_100. Cross-checked against a public
# pi05 SO-101 checkpoint, whose per-joint state ranges (e.g. wrist_flex up to
# 99.1) only fit the range-normalised reading.
def counts_to_lerobot(counts, calibration):
    """Encoder counts -> [5 x -100..100, gripper 0..100] (lerobot SO-101)."""
    values = []
    for name in ARM_JOINTS:
        c = calibration[name]
        span = c['range_max'] - c['range_min']
        values.append((counts[name] - c['range_min']) / span * 200.0 - 100.0)
    c = calibration['gripper']
    span = c['range_max'] - c['range_min']
    values.append((counts['gripper'] - c['range_min']) / span * 100.0)
    return np.asarray(values, dtype=np.float32)


def lerobot_to_counts(values, calibration):
    """[5 x -100..100, gripper 0..100] -> counts clamped to the calibrated limits."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size != 6:
        raise ValueError(f'Expected 6 action values, got {values.size}')
    if not np.isfinite(values).all():
        raise ValueError('Action values must be finite')
    counts = {}
    for name, value in zip(ARM_JOINTS, values[:5]):
        c = calibration[name]
        span = c['range_max'] - c['range_min']
        raw = (float(value) + 100.0) / 200.0 * span + c['range_min']
        counts[name] = int(round(min(c['range_max'], max(c['range_min'], raw))))
    c = calibration['gripper']
    span = c['range_max'] - c['range_min']
    raw = float(values[5]) / 100.0 * span + c['range_min']
    counts['gripper'] = int(round(min(c['range_max'], max(c['range_min'], raw))))
    return counts


def _encode(value):
    """Encode a frame to bytes: PNG when it is an image, UTF-8 for strings."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode('utf-8')
    array = np.asarray(value)
    if array.ndim in (2, 3) and array.dtype == np.uint8:
        import cv2
        ok, buffer = cv2.imencode('.png', array)
        if ok:
            return buffer.tobytes()
    return array.tobytes()


def _fill_image_message(message, value):
    """Fill an image sub-message: bytes + optional width/height/encoding."""
    fields = {f.name: f for f in message.DESCRIPTOR.fields}
    for name in ('data', 'image', 'bytes', 'pixels', 'encoded'):
        if name in fields:
            setattr(message, name, _encode(value))
            break
    else:
        array = np.asarray(value)
        for name in ('width', 'height'):
            if name in fields and array.ndim >= 2:
                setattr(message, name, int(array.shape[1] if name == 'width' else array.shape[0]))
        if 'channels' in fields:
            setattr(message, 'channels', int(array.shape[2]) if array.ndim == 3 else 1)
        return message
    array = np.asarray(value)
    if array.ndim >= 2:
        if 'width' in fields:
            setattr(message, 'width', int(array.shape[1]))
        if 'height' in fields:
            setattr(message, 'height', int(array.shape[0]))
    if 'encoding' in fields:
        setattr(message, 'encoding', 'png')
    return message


# --------------------------------------------------------------- descriptors

def _load_reflection(channel, timeout):
    """Return (pool, [(service_full_name, method_name)]) via server reflection."""
    from google.protobuf import descriptor_pb2, descriptor_pool
    from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

    stub = reflection_pb2_grpc.ServerReflectionStub(channel)
    services = []
    for response in stub.ServerReflectionInfo(
            iter([reflection_pb2.ServerReflectionRequest(list_services='')]),
            timeout=timeout):
        for service in response.list_services_response.service:
            if not service.name.startswith('grpc.reflection'):
                services.append(service.name)

    pool = descriptor_pool.DescriptorPool()
    seen = set()
    methods = []
    warnings = []
    for service in services:
        request = reflection_pb2.ServerReflectionRequest(file_containing_symbol=service)
        for response in stub.ServerReflectionInfo(iter([request]), timeout=timeout):
            for blob in response.file_descriptor_response.file_descriptor_proto:
                proto = descriptor_pb2.FileDescriptorProto()
                proto.ParseFromString(blob)
                key = (proto.name, proto.package)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    for dependency in proto.dependency:
                        _ensure_dependency(pool, stub, dependency, timeout, seen)
                    pool.Add(proto)
                except Exception as problem:  # noqa: BLE001 - one bad file must not abort discovery
                    warnings.append(f'skipped {proto.name}: {problem}')
                    continue
                for svc in proto.service:
                    full = f'{proto.package}.{svc.name}' if proto.package else svc.name
                    for method in svc.method:
                        methods.append((full, method.name))
    if warnings:
        print('\n'.join(warnings))
    return pool, methods


def _load_descriptor_set(paths):
    """Read a binary FileDescriptorSet produced by protoc --descriptor_set_out."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    pool = descriptor_pool.DescriptorPool()
    methods = []
    files = []
    for path in paths:
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(Path(path).read_bytes())
        files.extend(descriptor_set.file)
    for proto in files:  # protoc emits dependencies first
        try:
            pool.Add(proto)
        except Exception as problem:  # noqa: BLE001 - duplicates are harmless
            if 'already' not in str(problem).lower():
                raise
    for proto in files:
        for svc in proto.service:
            full = f'{proto.package}.{svc.name}' if proto.package else svc.name
            methods.extend((full, m.name) for m in svc.method)
    return pool, methods


def _load_proto_files(proto_files, proto_path=None):
    """Compile .proto files with grpc_tools and return (pool, methods)."""
    import os
    import tempfile

    from google.protobuf import descriptor_pb2, descriptor_pool
    from grpc_tools import protoc
    import grpc_tools

    files = [str(Path(p).resolve()) for p in proto_files]
    includes = list(proto_path or [])
    includes += sorted({str(Path(p).parent) for p in files})
    includes.append(str(Path(grpc_tools.__file__).parent / '_proto'))  # well-known types
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'descriptor.pb')
        args = ['protoc', *[f'-I{i}' for i in includes],
                f'--descriptor_set_out={out}', '--include_imports', *files]
        if protoc.main(args) != 0:
            raise RuntimeError(f'protoc failed for {files}; check imports / -I paths')
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(Path(out).read_bytes())

    pool = descriptor_pool.DescriptorPool()
    methods = []
    for proto in descriptor_set.file:  # dependency order is guaranteed by protoc
        pool.Add(proto)
    for proto in descriptor_set.file:
        for svc in proto.service:
            full = f'{proto.package}.{svc.name}' if proto.package else svc.name
            methods.extend((full, m.name) for m in svc.method)
    return pool, methods


def _ensure_dependency(pool, stub, name, timeout, seen):
    from google.protobuf import descriptor_pb2
    from grpc_reflection.v1alpha import reflection_pb2

    if name in seen or name.startswith('google/protobuf/'):
        return
    seen.add(name)
    request = reflection_pb2.ServerReflectionRequest(file_by_filename=name)
    for response in stub.ServerReflectionInfo(iter([request]), timeout=timeout):
        for blob in response.file_descriptor_response.file_descriptor_proto:
            proto = descriptor_pb2.FileDescriptorProto()
            proto.ParseFromString(blob)
            for dependency in proto.dependency:
                _ensure_dependency(pool, stub, dependency, timeout, seen)
            pool.Add(proto)


def _message_factory(pool):
    try:
        from google.protobuf import message_factory
        return message_factory.GetMessageClass
    except ImportError:  # protobuf < 4.22
        from google.protobuf import message_factory as factory
        return lambda descriptor: factory.MessageFactory(pool).GetPrototype(descriptor)


def describe(methods, pool):
    """Human-readable schema of every RPC, used by --dump-descriptor."""
    factory = _message_factory(pool)
    lines = []
    for service, method in methods:
        rpc = pool.FindMethodByName(f'{service}.{method}')
        if rpc is None:
            continue
        lines.append(f'{service}/{method}')
        for label, message in (('request', rpc.input_type), ('response', rpc.output_type)):
            lines.append(f'  {label}: {message.full_name}')
            for field in message.fields:
                type_name = (field.message_type.full_name if field.message_type
                             else field.type)
                lines.append(f'    {field.number:>2} {field.name:<28} {type_name} '
                             f'{"repeated" if _is_repeated(field) else ""}')
    return '\n'.join(lines)


# --------------------------------------------------------------------- client

class PolicyClient:
    def __init__(self, host, port=50051, timeout=60.0, max_message_size=16 * 1024 * 1024,
                 method=None):
        import grpc
        self._grpc = grpc
        self.address = f'{host}:{port}'
        self.timeout = timeout
        self.channel = grpc.insecure_channel(
            self.address,
            options=[('grpc.max_receive_message_length', max_message_size),
                     ('grpc.max_send_message_length', max_message_size)])
        self.pool = None
        self.methods = []
        self.rpc = None
        self.method_name = method
        self._factory = None
        self._vlash_rpcs = None

    def close(self):
        self.channel.close()

    def connect(self, proto_files=None, proto_path=None, descriptor_set=None):
        """Load explicit schema, reflection, or built-in VLASH on UNIMPLEMENTED."""
        if descriptor_set:
            pool, methods = _load_descriptor_set(descriptor_set)
            if not methods:
                raise RuntimeError(f'No services in {descriptor_set}')
        elif proto_files:
            pool, methods = _load_proto_files(proto_files, proto_path)
            if not methods:
                raise RuntimeError(f'No services defined in {proto_files}')
        else:
            try:
                pool, methods = _load_reflection(self.channel, self.timeout)
            except self._grpc.RpcError as error:
                if error.code() != self._grpc.StatusCode.UNIMPLEMENTED:
                    raise RuntimeError(
                        f'Cannot discover {self.address}: {error.code()} {error.details()}. '
                        'Check the SSH tunnel and wait for "Server started on port".') from error
                from vlash_transport import load_schema
                pool, methods = load_schema()
                print('Reflection unavailable; using built-in transport.RemoteInference schema')
            if not methods:
                raise RuntimeError(f'{self.address} exposes no RPCs through reflection')
        self.pool = pool
        self.methods = methods
        self._factory = _message_factory(pool)
        vlash_services = [s for s, m in methods if m == 'LaunchInference'
                          and (s, 'GetActionChunk') in methods and (s, 'Ready') in methods]
        candidates = [(s, m) for s, m in methods
                      if any(k in m.lower() for k in ('infer', 'predict', 'act', 'policy', 'step'))]
        if self.method_name:
            service, _, name = self.method_name.strip('/').rpartition('/')
            rpc = None
            if service:
                try:
                    rpc = pool.FindMethodByName(f'{service}.{name}')
                except KeyError:
                    rpc = None
            if rpc is None:
                raise RuntimeError(f'Method {self.method_name} not in schema; have {methods}')
            self.rpc = rpc
        else:
            if vlash_services:
                candidates = [(vlash_services[0], 'LaunchInference')]
            if not candidates:
                raise RuntimeError(f'No inference-like RPC; schema has {methods}. Use --method')
            service, name = candidates[0]
            self.rpc = pool.FindMethodByName(f'{service}.{name}')
        self.method_name = f'/{self.rpc.containing_service.full_name}/{self.rpc.name}'
        service = self.rpc.containing_service.full_name
        if service in vlash_services:
            if self.rpc.name != 'LaunchInference':
                raise RuntimeError('For VLASH use --method /transport.RemoteInference/LaunchInference '
                                   'or omit --method; GetActionChunk is called automatically')
            launch = self.rpc
            get = pool.FindMethodByName(f'{service}.GetActionChunk')
            ready = pool.FindMethodByName(f'{service}.Ready')
            if (not launch.client_streaming or launch.server_streaming
                    or get.client_streaming or get.server_streaming
                    or ready.client_streaming or ready.server_streaming):
                raise RuntimeError('Unexpected VLASH RPC streaming signatures')
            self._vlash_rpcs = (launch, get, ready)
            self.ready()
        elif self.rpc.client_streaming or self.rpc.server_streaming:
            raise RuntimeError(f'Unsupported streaming RPC: {self.method_name}')
        return self

    def _unary(self, rpc, request, timeout=None):
        return self.channel.unary_unary(
            f'/{rpc.containing_service.full_name}/{rpc.name}',
            request_serializer=lambda m: m.SerializeToString(),
            response_deserializer=self._factory(rpc.output_type).FromString,
        )(request, timeout=self.timeout if timeout is None else timeout)

    def ready(self):
        """Check the actual service, rather than just an open TCP tunnel."""
        if self._vlash_rpcs:
            rpc = self._vlash_rpcs[2]
            try:
                self._unary(rpc, self._factory(rpc.input_type)())
            except self._grpc.RpcError as error:
                raise RuntimeError(
                    f'{self.address} Ready failed: {error.code()} {error.details()}. '
                    'Check the forwarded port and wait for "Server started on port".') from error

    def _infer_vlash(self, observation, prompt):
        from vlash_transport import decode_actions, encode_observation, observation_chunks

        launch, get, _ = self._vlash_rpcs
        payload = encode_observation(observation, prompt)
        try:
            self.channel.stream_unary(
                self.method_name,
                request_serializer=lambda m: m.SerializeToString(),
                response_deserializer=self._factory(launch.output_type).FromString,
            )(observation_chunks(payload, self._factory(launch.input_type)), timeout=self.timeout)
        except self._grpc.RpcError as error:
            raise RuntimeError(f'VLASH LaunchInference: {error.code()} {error.details()}') from error
        try:
            reply = self._unary(get, self._factory(get.input_type)())
        except self._grpc.RpcError as error:
            raise RuntimeError(f'VLASH GetActionChunk: {error.code()} {error.details()}') from error
        return decode_actions(reply.data)

    @property
    def request_fields(self):
        return {f.name: f for f in self.rpc.input_type.fields}

    @property
    def response_fields(self):
        return {f.name: f for f in self.rpc.output_type.fields}

    def transport(self):
        """Select VLASH streaming pickle, structured protobuf, or msgpack."""
        if self._vlash_rpcs:
            return 'vlash'
        names = set(self.request_fields)
        if names & {'observation', 'state', 'prompt', 'request'}:
            return 'proto'
        blobs = [f for f in self.rpc.input_type.fields if f.type == f.TYPE_BYTES]
        if len(self.rpc.input_type.fields) == 1 and blobs:
            return 'msgpack'
        return 'proto'

    def _infer_msgpack(self, observation, prompt):
        """Fallback: one opaque bytes field carrying msgpack (openpi convention)."""
        import msgpack
        field = self.rpc.input_type.fields[0]
        request = self._factory(self.rpc.input_type)()
        setattr(request, field.name, msgpack.packb(
            {'observation': observation, 'prompt': prompt}, use_bin_type=True))
        reply = self.channel.unary_unary(
            self.method_name,
            request_serializer=lambda m: m.SerializeToString(),
            response_deserializer=self._factory(self.rpc.output_type).FromString,
        )(request, timeout=self.timeout)
        fields = list(reply.DESCRIPTOR.fields)
        raw = getattr(reply, fields[0].name) if len(fields) == 1 else None
        if isinstance(raw, bytes):
            data = msgpack.unpackb(raw, raw=False)
            key = 'actions' if 'actions' in data else 'action'
            return np.atleast_2d(np.asarray(data[key], dtype=np.float32))
        values = [getattr(reply, f.name) for f in fields]
        return np.asarray(values, dtype=np.float32).reshape(1, -1)

    # ------------------------------------------------------------ message build
    def _fill(self, message, observation, prompt):
        """Populate a request message from a flat observation dict.

        Handles both shapes seen in the wild:
          a) one nested `observation` map + `prompt`
          b) flat `observation.state` / `observation.images.x` protobuf fields
        """
        from google.protobuf import json_format

        fields = {f.name: f for f in message.DESCRIPTOR.fields}
        if 'observation' in fields and fields['observation'].message_type is not None:
            nested = fields['observation'].message_type
            self._fill_flat(message.observation, nested, observation)
            if 'prompt' in fields:
                setattr(message, 'prompt', prompt)
            return message
        self._fill_flat(message, message.DESCRIPTOR, observation, prompt)
        return message

    def _fill_flat(self, message, descriptor, observation, prompt=None):
        for field in descriptor.fields:
            name = field.name
            value = None
            if name in observation:
                value = observation[name]
            elif name == 'prompt':
                value = prompt
            elif name == 'state' and 'observation.state' in observation:
                value = observation['observation.state']
            elif name.startswith('image') or name.endswith('_rgb'):
                for key, candidate in observation.items():
                    if key.endswith(name) or name.endswith(key.split('.')[-1]):
                        value = candidate
                        break
            if value is None:
                continue
            try:
                if field.type == field.TYPE_BYTES:
                    setattr(message, name, _encode(value))
                elif field.type == field.TYPE_STRING:
                    setattr(message, name, str(value))
                elif _is_repeated(field) and field.type in (
                        field.TYPE_DOUBLE, field.TYPE_FLOAT):
                    getattr(message, name).extend(np.asarray(value).ravel().tolist())
                elif field.message_type is not None:
                    _fill_image_message(getattr(message, name), value)
                else:
                    setattr(message, name, value)
            except (TypeError, ValueError) as error:
                raise RuntimeError(f'Cannot fill field {name}: {error}') from error
        return message

    def _read_actions(self, reply):
        from google.protobuf import json_format

        actions = None
        for field in reply.DESCRIPTOR.fields:
            if field.name in ('actions', 'action') and _is_repeated(field):
                if field.type in (field.TYPE_DOUBLE, field.TYPE_FLOAT):
                    actions = np.asarray(list(getattr(reply, field.name)), dtype=np.float32)
                elif field.message_type is not None:
                    rows = []
                    for row in getattr(reply, field.name):
                        values = [v for f in row.DESCRIPTOR.fields for v in (
                            getattr(row, f.name) if not _is_repeated(f)
                            else list(getattr(row, f.name)))]
                        rows.append(np.asarray(values, dtype=np.float32))
                    actions = np.asarray(rows, dtype=np.float32)
            elif field.name == 'actions' and field.type == field.TYPE_BYTES:
                actions = np.frombuffer(getattr(reply, field.name), dtype=np.float32)
        if actions is None:
            try:
                data = json_format.MessageToDict(reply)
            except Exception:
                data = {}
            for key in ('actions', 'action'):
                if key in data:
                    actions = np.asarray(data[key], dtype=np.float32)
                    break
        if actions is None:
            names = [f.name for f in reply.DESCRIPTOR.fields]
            raise RuntimeError(f'No actions in response; fields={names}')
        if actions.ndim == 1:
            actions = actions.reshape(-1, 6) if actions.size % 6 == 0 else actions.reshape(1, -1)
        return actions

    # ---------------------------------------------------------------- inference
    def infer(self, observation, prompt):
        from vlash_transport import validate_actions

        if self._vlash_rpcs:
            return self._infer_vlash(observation, prompt)
        if self.transport() == 'msgpack':
            return validate_actions(self._infer_msgpack(observation, prompt))
        request = self._factory(self.rpc.input_type)()
        self._fill(request, observation, prompt)
        reply = self.channel.unary_unary(
            self.method_name,
            request_serializer=lambda m: m.SerializeToString(),
            response_deserializer=self._factory(self.rpc.output_type).FromString,
        )(request, timeout=self.timeout)
        return validate_actions(self._read_actions(reply))
