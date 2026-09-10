"""Device-boundary regressions: python -m unittest tests.test_devices -v."""

import unittest

import torch


class TestDevices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("requires two CUDA devices")
        import deep_select
        from deep_select import deep_select_cuda

        cls.api = deep_select
        cls.backend = deep_select_cuda

    def test_rejects_mixed_devices(self):
        tensors = {
            "input": torch.zeros((1, 256), device="cuda:0"),
            "end": torch.full((1,), 256, dtype=torch.int32, device="cuda:0"),
            "output_value": torch.empty((1, 8), device="cuda:0"),
            "output_index": torch.empty((1, 8), dtype=torch.int32, device="cuda:0"),
            "output_idx_offset": torch.zeros((1,), dtype=torch.int32, device="cuda:0"),
        }
        for name in ("end", "output_value", "output_index", "output_idx_offset"):
            with self.subTest(tensor=name):
                args = dict(tensors)
                args[name] = args[name].to("cuda:1")
                with self.assertRaisesRegex(RuntimeError, f"`{name}` must be on the same device as `input`"):
                    self.backend.topk(
                        args["input"], 8, None, args["end"], False, False,
                        args["output_value"], args["output_index"], args["output_idx_offset"],
                        2147483647, float("-inf"), True, True,
                    )

    def test_noncurrent_device_and_stream(self):
        supported = [i for i in range(torch.cuda.device_count())
                     if torch.cuda.get_device_capability(i) in ((10, 0), (10, 3))]
        if not supported:
            self.skipTest("requires an SM100 or SM103 device")
        target = supported[0]
        other = next(i for i in range(torch.cuda.device_count()) if i != target)
        stream = torch.cuda.Stream(device=target)
        with torch.cuda.stream(stream):
            x = torch.arange(256, device=f"cuda:{target}", dtype=torch.float32).reshape(1, 256)
            # Warm up lazy setup before capture.
            self.api.topk(x, 8, sorted=True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with torch.cuda.device(other):
                values, indices = self.api.topk(x, 8, sorted=True)
                self.assertEqual(torch.cuda.current_device(), other)
        # Capture proves the launch uses target's current stream: graph replay
        # must recompute the result after changing input on that same stream.
        with torch.cuda.stream(stream):
            x.neg_()
            graph.replay()
        stream.synchronize()
        expected = torch.topk(x, 8)
        torch.testing.assert_close(values, expected.values)
        torch.testing.assert_close(indices.to(torch.int64), expected.indices)


if __name__ == "__main__":
    unittest.main()
