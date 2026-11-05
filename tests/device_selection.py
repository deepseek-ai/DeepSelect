"""Device placement checks for the Top-K launch configuration.

Requires at least two CUDA devices: the point is that the caller's current
device differs from the input's device.
"""

import torch

import deep_select


def require_two_devices():
    if torch.cuda.device_count() < 2:
        raise SystemExit("SKIP: needs at least two CUDA devices")


@torch.inference_mode()
def check_foreign_input(dtype, sorted_value, name):
    torch.cuda.set_device(0)
    host = torch.arange(4096, dtype=torch.float32).reshape(1, 4096).to(dtype)
    x = host.to("cuda:1")
    value, index = deep_select.topk(x, 8, sorted=sorted_value, return_value=True)
    # Wait on both devices: a launch on the wrong device is reported asynchronously.
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(0)
    assert torch.cuda.current_device() == 0, name
    assert index.device == x.device and value.device == x.device, name
    picked = host[0].gather(0, index[0].cpu().to(torch.int64))
    assert torch.equal(value[0].cpu(), picked), name
    expected = torch.topk(host, 8, dim=1).values
    assert torch.equal(torch.sort(picked, descending=True).values, expected[0]), name
    assert index[0].cpu().unique().numel() == 8, name
    print(f"{name}: PASS", flush=True)


@torch.inference_mode()
def check_mixed_output_rejected(name):
    torch.cuda.set_device(0)
    x = torch.arange(4096, dtype=torch.float32).reshape(1, 4096).cuda(0)
    output_index = torch.empty((1, 8), dtype=torch.int64, device="cuda:1")
    try:
        deep_select.topk(x, 8, sorted=False, return_value=False, output_idx=output_index)
    except RuntimeError as error:
        assert "same CUDA device" in str(error), (name, error)
        print(f"{name}: PASS ({error})", flush=True)
        return
    raise AssertionError(f"{name}: mixed-device output was accepted")


require_two_devices()
check_foreign_input(torch.float32, True, "fp32-foreign-device")
check_foreign_input(torch.bfloat16, False, "bf16-foreign-device")
check_mixed_output_rejected("mixed-device-output")
