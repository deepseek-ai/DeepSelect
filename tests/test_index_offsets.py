"""Run with python -m unittest discover -s tests -p test_index_offsets.py -v."""

import unittest

import torch


class IndexOffsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("the extension requires SM100 or SM103")
        import deep_select
        cls.topk = staticmethod(deep_select.topk)

    def test_sorted_offsets_and_padding(self):
        # Full backing rows meet input alignment and padding requirements.
        vocab_size, topk = 8192, 8
        offsets = torch.tensor([2**31 - 1, -(2**31), -1, 0, 1], dtype=torch.int32, device="cuda")
        inputs = torch.arange(vocab_size, dtype=torch.float32, device="cuda").repeat(len(offsets), 1)
        # Exercise selection, exact-size shortcut, partial shortcut and empty rows.
        for end_value in (vocab_size, topk, 3, 0):
            for fill in (2**31 - 1, -(2**31), -1):
                with self.subTest(end=end_value, fill=fill):
                    end = torch.full_like(offsets, end_value)
                    values, indices = self.topk(
                        inputs, topk, sorted=True, end=end,
                        output_idx_offset=offsets, idx_oob_fill_value=fill,
                        indices_type=torch.int64,
                    )
                    count = min(end_value, topk)
                    selected = torch.arange(end_value - 1, end_value - count - 1, -1, device="cuda")
                    expected_indices = torch.full_like(indices, fill)
                    expected_indices[:, :count] = selected + offsets.to(torch.int64)[:, None]
                    expected_values = torch.full_like(values, float("-inf"))
                    expected_values[:, :count] = selected.to(torch.float32)
                    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
                    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)

    def test_int32_padding_with_representable_offsets(self):
        inputs = torch.zeros((2, 256), device="cuda")
        offsets = torch.tensor([-1, 1], dtype=torch.int32, device="cuda")
        end = torch.ones_like(offsets)
        for fill in (2**31 - 1, -(2**31)):
            with self.subTest(fill=fill):
                _, indices = self.topk(
                    inputs, 8, sorted=True, end=end,
                    output_idx_offset=offsets, idx_oob_fill_value=fill,
                    indices_type=torch.int32,
                )
                expected = torch.full_like(indices, fill)
                expected[:, 0] = offsets
                torch.testing.assert_close(indices, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
