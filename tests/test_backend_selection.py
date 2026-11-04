"""Exercise backend discovery without requiring either native extension."""

import importlib
import os
import uuid
from pathlib import Path

import pytest
import torch  # noqa: F401  # Load before patching the platform probe.


@pytest.fixture
def load_interface(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "deep_select" / "interface.py"

    def load(*, ascend_device, cuda=None, npu=None):
        package = "backend_test_" + uuid.uuid4().hex
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text("")
        (root / "interface.py").write_text(source.read_text())
        for name, body in (("cuda", cuda), ("npu", npu)):
            if body is not None:
                (root / f"deep_select_{name}.py").write_text(body)
        monkeypatch.syspath_prepend(str(tmp_path))
        exists = os.path.exists

        def platform_exists(path):
            return ascend_device if str(path) == "/dev/davinci_manager" else exists(path)

        with monkeypatch.context() as scoped:
            scoped.setattr(os.path, "exists", platform_exists)
            return importlib.import_module(f"{package}.interface")

    return load


@pytest.mark.parametrize(
    "ascend_device,cuda,npu,expected",
    [
        (False, "name = 'cuda'", None, "cuda"),
        (True, None, "name = 'npu'", "npu"),
        (True, "name = 'cuda'", None, "cuda"),
        (False, None, "name = 'npu'", "npu"),
        (False, "name = 'cuda'", "name = 'npu'", "cuda"),
        (True, "name = 'cuda'", "name = 'npu'", "npu"),
    ],
)
def test_selects_installed_backend(load_interface, ascend_device, cuda, npu, expected):
    interface = load_interface(ascend_device=ascend_device, cuda=cuda, npu=npu)
    assert interface._backend.name == expected


def test_missing_backend_reports_build_target(load_interface):
    with pytest.raises(ImportError, match="DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA"):
        load_interface(ascend_device=False)


def test_broken_preferred_backend_is_not_silently_skipped(load_interface):
    with pytest.raises(ImportError) as error:
        load_interface(
            ascend_device=True,
            cuda="name = 'cuda'",
            npu="raise ImportError('broken native dependency')",
        )
    assert "deep_select_npu" in str(error.value)
    assert "broken native dependency" in str(error.value.__cause__)
