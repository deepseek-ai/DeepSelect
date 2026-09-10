"""Run with python -m unittest tests.test_memory_boundaries on SM100/SM103.

Also run under compute-sanitizer --tool memcheck for allocation-boundary checks.
"""
import unittest

import torch


class MemoryBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("requires an SM100/SM103 GPU")
        import deep_select
        cls.topk = staticmethod(deep_select.topk)

    def test_output_canaries(self):
        for dtype in (torch.float32, torch.bfloat16):
            for index_dtype in (torch.int32, torch.int64):
                for mode in ("unsorted", "index", "value"):
                    if dtype == torch.bfloat16 and mode == "value":
                        continue
                    for k in (1, 3, 7, 9, 17, 1025):
                        for shortcut in (False, True):
                            with self.subTest(dtype=dtype, index_dtype=index_dtype, mode=mode, k=k, shortcut=shortcut):
                                # Rounded rows also provide the TMA input padding.
                                x = torch.arange(2048, device="cuda", dtype=torch.float32).repeat(2, 1).to(dtype)
                                width = (k + 31) // 32 * 32 + 32
                                backing = torch.full((2, width), -123, device="cuda", dtype=index_dtype)
                                output = backing[:, :k]
                                end_value = min(k, 5) if shortcut else 2048
                                end = torch.full((2,), end_value, device="cuda", dtype=torch.int32)
                                values, indices = self.topk(x, k, end=end, indices_type=index_dtype,
                                                           output_idx=output, sorted=mode == "value",
                                                           sorted_index=mode == "index")
                                torch.cuda.synchronize()
                                self.assertTrue(torch.all(backing[:, k:] == -123).item())
                                count = min(k, end_value)
                                selected = indices[:, :count].long()
                                self.assertTrue(torch.all((selected >= 0) & (selected < end_value)).item())
                                torch.testing.assert_close(values[:, :count], x.gather(1, selected))
                                expected = x[:, :end_value].topk(count).values.sort(dim=1).values
                                torch.testing.assert_close(values[:, :count].sort(dim=1).values, expected)

    def test_minimal_output_storage(self):
        x = torch.ones((2, 256), device="cuda")
        # Exactly 36 bytes: row zero at byte 0, row one at byte 32.
        output = torch.empty(9, device="cuda", dtype=torch.int32).as_strided((2, 1), (8, 1))
        end = torch.ones(2, device="cuda", dtype=torch.int32)
        _, indices = self.topk(x, 1, end=end, output_idx=output, indices_type=torch.int32)
        torch.cuda.synchronize()
        torch.testing.assert_close(indices, torch.zeros_like(indices))

    def test_input_requires_final_row_padding(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                stride = 1024 // dtype.itemsize
                width = stride + 1
                row_stride = 2 * stride
                x = torch.ones(row_stride + width, device="cuda", dtype=dtype).as_strided((2, width), (row_stride, 1))
                with self.assertRaisesRegex(RuntimeError, "input backing storage"):
                    self.topk(x, 1)
                padded = torch.ones((2, row_stride), device="cuda", dtype=dtype)[:, :width]
                values, _ = self.topk(padded, 1)
                torch.testing.assert_close(values, torch.ones_like(values))

    def test_misaligned_views(self):
        x = torch.ones((2, 512), device="cuda")
        with self.assertRaisesRegex(RuntimeError, "input.data_ptr"):
            self.topk(x[:, 1:257], 1)
        backing = torch.empty((2, 16), device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "output_index.data_ptr"):
            self.topk(x, 1, output_idx=backing[:, 1:2], indices_type=torch.int32)

    def test_output_rows_must_not_overlap(self):
        x = torch.ones((2, 256), device="cuda")
        output = torch.empty((1, 8), device="cuda", dtype=torch.int32)[:, :1].expand(2, 1)
        with self.assertRaisesRegex(RuntimeError, "output_index rows must not overlap"):
            self.topk(x, 1, output_idx=output, indices_type=torch.int32)


if __name__ == "__main__":
    unittest.main()
