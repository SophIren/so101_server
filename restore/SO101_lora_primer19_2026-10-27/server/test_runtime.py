"""Run in the restored server environment; no checkpoint or hardware required."""
from concurrent.futures import ThreadPoolExecutor
import pickle
from types import SimpleNamespace
import unittest

import grpc
import torch

from vlash.serve import RemoteInferenceServicer
from vlash.so101_policy_loader import check_normalization
from vlash.transport import services_pb2 as pb, services_pb2_grpc as rpc


class RuntimeTests(unittest.TestCase):
    def test_normalization_rejects_missing_or_nonfinite_statistics(self):
        def buffers():
            return SimpleNamespace(mean=torch.zeros(6), std=torch.ones(6))
        policy = SimpleNamespace(
            normalize_inputs=SimpleNamespace(buffer_observation_state=buffers()),
            normalize_targets=SimpleNamespace(buffer_action=buffers()),
            unnormalize_outputs=SimpleNamespace(buffer_action=buffers()))
        check_normalization(policy)
        policy.unnormalize_outputs.buffer_action.std[0] = float('inf')
        with self.assertRaises(ValueError):
            check_normalization(policy)
        policy.unnormalize_outputs.buffer_action.std[0] = 0
        with self.assertRaises(ValueError):
            check_normalization(policy)

    def test_backported_rpc_returns_cpu_actions_and_propagates_model_errors(self):
        class FakePolicy:
            bad = False

            def predict_action_chunk(self, batch):
                if self.bad:
                    return torch.full((1, 32, 6), float('nan'))
                return batch['observation.state'][:, None, :].expand(1, 32, 6)

        policy = FakePolicy()
        servicer = RemoteInferenceServicer(policy, torch.device('cpu'))
        server = grpc.server(ThreadPoolExecutor(max_workers=4))
        rpc.add_RemoteInferenceServicer_to_server(servicer, server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()
        channel = grpc.insecure_channel(f'127.0.0.1:{port}')
        stub = rpc.RemoteInferenceStub(channel)
        try:
            stub.Ready(pb.Empty(), timeout=3)
            batch = {'observation.state': torch.ones(1, 6)}
            payload = pickle.dumps(batch)
            stub.LaunchInference(iter([pb.Observation(transfer_state=3, data=payload)]), timeout=3)
            result = pickle.loads(stub.GetActionChunk(pb.Empty(), timeout=3).data)
            self.assertEqual(tuple(result.shape), (32, 6))
            self.assertEqual(result.device.type, 'cpu')
            self.assertTrue(torch.equal(result, torch.ones(32, 6)))
            policy.bad = True
            stub.LaunchInference(iter([pb.Observation(transfer_state=3, data=payload)]), timeout=3)
            with self.assertRaises(grpc.RpcError) as failure:
                stub.GetActionChunk(pb.Empty(), timeout=3)
            self.assertEqual(failure.exception.code(), grpc.StatusCode.INTERNAL)
            self.assertIn('invalid actions', failure.exception.details())
        finally:
            channel.close()
            server.stop(0).wait(3)
            servicer.shutdown()


if __name__ == '__main__':
    unittest.main()
