from unittest.mock import MagicMock, patch

import pytest
import torch
from deep_select import interface
from tests.kernelkit.platform import Platform, get_current_platform


class TestInterfaceValidation:
    """Tests for parameter validation in deep_select.interface.topk"""

    def test_topk_range_validation(self):
        x = torch.randn(4, 128)
        # topk <= 0
        with pytest.raises(ValueError, match=r"topk must be an integer in \[1, 4096\]"):
            interface.topk(x, 0)

        with pytest.raises(ValueError, match=r"topk must be an integer in \[1, 4096\]"):
            interface.topk(x, -5)

        # topk > 4096
        with pytest.raises(ValueError, match=r"topk must be an integer in \[1, 4096\]"):
            interface.topk(x, 4097)

        # topk non-integer
        with pytest.raises(ValueError, match=r"topk must be an integer in \[1, 4096\]"):
            interface.topk(x, 3.5)  # type: ignore

    def test_input_dimension_validation(self):
        # 1D input
        x_1d = torch.randn(128)
        with pytest.raises(ValueError, match="input must be a 2D tensor"):
            interface.topk(x_1d, 10)

        # 3D input
        x_3d = torch.randn(2, 4, 128)
        with pytest.raises(ValueError, match="input must be a 2D tensor"):
            interface.topk(x_3d, 10)

        # non-tensor input
        with pytest.raises(ValueError, match="input must be a 2D tensor"):
            interface.topk([[1.0, 2.0]], 1)  # type: ignore

    def test_indices_type_validation(self):
        x = torch.randn(4, 128)
        with pytest.raises(TypeError, match="indices_type must be torch.int32 or torch.int64"):
            interface.topk(x, 10, indices_type=torch.float32)  # type: ignore

    def test_unsupported_begin_and_hint(self):
        x = torch.randn(4, 128)
        begin = torch.zeros(4, dtype=torch.int32)
        hint = torch.zeros(4, dtype=torch.int32)

        with pytest.raises(NotImplementedError, match="`begin` is not supported now"):
            interface.topk(x, 10, begin=begin)

        with pytest.raises(NotImplementedError, match="`hint` is not supported now"):
            interface.topk(x, 10, hint=hint)

    def test_output_idx_validation(self):
        x = torch.randn(4, 128)

        # Mismatched shape
        wrong_shape_idx = torch.empty((4, 20), dtype=torch.int32)
        with pytest.raises(ValueError, match="output_idx shape must be"):
            interface.topk(x, 10, output_idx=wrong_shape_idx)

        # Mismatched dtype
        wrong_dtype_idx = torch.empty((4, 10), dtype=torch.int64)
        with pytest.raises(TypeError, match="output_idx dtype must be"):
            interface.topk(x, 10, indices_type=torch.int32, output_idx=wrong_dtype_idx)

    def test_end_validation(self):
        x = torch.randn(4, 128)

        # Wrong shape
        wrong_shape_end = torch.empty((2,), dtype=torch.int32)
        with pytest.raises(ValueError, match="end shape must be"):
            interface.topk(x, 10, end=wrong_shape_end)

        # Wrong dtype
        wrong_dtype_end = torch.empty((4,), dtype=torch.int64)
        with pytest.raises(TypeError, match="end dtype must be torch.int32"):
            interface.topk(x, 10, end=wrong_dtype_end)

    def test_output_idx_offset_validation(self):
        x = torch.randn(4, 128)

        # Wrong shape
        wrong_shape_offset = torch.empty((2,), dtype=torch.int32)
        with pytest.raises(ValueError, match="output_idx_offset shape must be"):
            interface.topk(x, 10, output_idx_offset=wrong_shape_offset)

        # Wrong dtype
        wrong_dtype_offset = torch.empty((4,), dtype=torch.float32)
        with pytest.raises(TypeError, match="output_idx_offset dtype must be torch.int32"):
            interface.topk(x, 10, output_idx_offset=wrong_dtype_offset)

    def test_mock_backend_invocation(self):
        mock_backend = MagicMock()
        mock_backend.get_alignment_requirement.return_value = (32, 32)
        mock_backend.topk.return_value = None

        with patch.object(interface, "get_backend", return_value=mock_backend):
            # Clear lru_cache for get_stride_requirement so mock backend is consulted
            interface.get_stride_requirement.cache_clear()

            x = torch.randn(4, 128)
            val, idx = interface.topk(x, 10, return_value=True)

            assert val is not None
            assert val.shape == (4, 10)
            assert idx is not None
            assert idx.shape == (4, 10)
            assert idx.dtype == torch.int64
            assert mock_backend.topk.called


class TestPlatformDetection:
    """Tests for platform fallback when lspci is not found"""

    def test_get_current_platform_fallback(self):
        get_current_platform.cache_clear()
        with patch("os.path.exists", return_value=False), \
             patch("subprocess.check_output", side_effect=FileNotFoundError("lspci not found")), \
             patch("torch.cuda.is_available", return_value=False):
            plat = get_current_platform()
            assert plat == Platform.CPU_ONLY

        get_current_platform.cache_clear()
        with patch("os.path.exists", return_value=False), \
             patch("subprocess.check_output", side_effect=FileNotFoundError("lspci not found")), \
             patch("torch.cuda.is_available", return_value=True):
            plat = get_current_platform()
            assert plat == Platform.CUDA
