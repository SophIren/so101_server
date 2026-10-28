"""Fake vlash server for exercising policy_client.py without the real host."""
import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import grpc
import numpy as np
from grpc_reflection.v1alpha import reflection

import vlash_test_pb2 as pb
import vlash_test_pb2_grpc as pb_grpc

STATS = None


class TransportServicer:
    """Actual transport.RemoteInference contract; reflection is disabled."""
    def __init__(self, chunk=32):
        self.chunk = chunk
        self.calls = []
        self.events = []
        self.actions = None
        self.error = None

    def add_to_server(self, server, pool=None):
        from google.protobuf.message_factory import GetMessageClass
        from vlash_transport import load_schema
        pool = pool or load_schema()[0]
        self.Empty = GetMessageClass(pool.FindMessageTypeByName('transport.Empty'))
        self.ActionChunk = GetMessageClass(pool.FindMessageTypeByName('transport.ActionChunk'))
        Observation = GetMessageClass(pool.FindMessageTypeByName('transport.Observation'))
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            'transport.RemoteInference', {
                'Ready': grpc.unary_unary_rpc_method_handler(
                    self.Ready, request_deserializer=self.Empty.FromString,
                    response_serializer=self.Empty.SerializeToString),
                'LaunchInference': grpc.stream_unary_rpc_method_handler(
                    self.LaunchInference, request_deserializer=Observation.FromString,
                    response_serializer=self.Empty.SerializeToString),
                'GetActionChunk': grpc.unary_unary_rpc_method_handler(
                    self.GetActionChunk, request_deserializer=self.Empty.FromString,
                    response_serializer=self.ActionChunk.SerializeToString),
            }),))

    def Ready(self, request, context):
        self.events.append('Ready')
        return self.Empty()

    def LaunchInference(self, request_iterator, context):
        import torch
        self.events.append('LaunchInference')
        parts = []
        states = []
        ended = False
        for item in request_iterator:
            states.append(item.transfer_state)
            if len(item.data) > 2 * 1024 * 1024:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'observation part exceeds 2 MiB')
            if item.transfer_state == 1:
                parts = [item.data]
            elif item.transfer_state in (2, 3):
                parts.append(item.data)
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'unknown transfer state')
            if item.transfer_state == 3:
                ended = True
                break
        if not ended:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'missing TRANSFER_END')
        batch = pickle.loads(b''.join(parts))
        state = batch['observation.state']
        if not isinstance(state, torch.Tensor) or tuple(state.shape) != (1, 6):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'expected torch state [1,6]')
        images = {k: v for k, v in batch.items() if k.startswith('observation.images.')}
        for image in images.values():
            if (not isinstance(image, torch.Tensor) or image.dtype != torch.float32
                    or image.ndim != 4 or image.shape[:2] != (1, 3)
                    or image.min() < 0 or image.max() > 1):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'expected float32 RGB BCHW [0,1]')
        self.calls.append({'batch': batch, 'states': states, 'sizes': list(map(len, parts))})
        self.actions = state + torch.arange(1, self.chunk + 1).reshape(-1, 1) * 0.1
        if STATS is not None:
            STATS.write_text(json.dumps({
                'calls': len(self.calls), 'task': batch['task'],
                'state': state.tolist(), 'images': {k: list(v.shape) for k, v in images.items()},
                'transfer_states': states,
            }, indent=2) + '\n')
        return self.Empty()

    def GetActionChunk(self, request, context):
        self.events.append('GetActionChunk')
        if self.error:
            context.abort(grpc.StatusCode.INTERNAL, self.error)
        if self.actions is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'LaunchInference must run first')
        return self.ActionChunk(data=pickle.dumps(self.actions))


class Servicer(pb_grpc.RemoteInferenceServicer):
    def __init__(self, chunk=50):
        self.chunk = chunk
        self.calls = []

    def Infer(self, request, context):
        images = []
        for name in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'):
            blob = getattr(request.observation, name).data
            frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR) if blob else None
            images.append(None if frame is None else list(frame.shape))
        self.calls.append({'prompt': request.prompt,
                           'state': list(request.observation.state),
                           'images': images})
        if STATS is not None:
            STATS.write_text(json.dumps({'calls': len(self.calls),
                                         'last': self.calls[-1]}, indent=2) + '\n')
        start = np.asarray(request.observation.state, dtype=np.float32)
        return pb.InferResponse(
            actions=[pb.Action(values=list(start + 0.1 * (i + 1))) for i in range(self.chunk)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--stats', type=Path, help='write received requests here')
    parser.add_argument('--protocol', choices=['transport', 'legacy'], default='transport')
    args = parser.parse_args()
    STATS = args.stats

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    if args.protocol == 'transport':
        TransportServicer().add_to_server(server)
    else:
        pb_grpc.add_RemoteInferenceServicer_to_server(Servicer(), server)
        reflection.enable_server_reflection(
            (pb.DESCRIPTOR.services_by_name['RemoteInference'].full_name, reflection.SERVICE_NAME),
            server)
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    print(f'mock vlash on {args.port}', flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
