from __future__ import annotations

import os

GPU_DEVICE_ID = "7"


def force_gpu_7() -> str:
    """Reserve only physical GPU 7 for Buili GPU-side processes."""
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE_ID
    return GPU_DEVICE_ID


def assert_gpu_7() -> None:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != GPU_DEVICE_ID:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {GPU_DEVICE_ID}, got {actual!r}")


def gpu_policy() -> dict[str, str]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "required_device": GPU_DEVICE_ID,
        "policy": "single_gpu_only",
    }
