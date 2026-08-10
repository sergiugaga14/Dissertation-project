"""Device selection so the same code runs on a GPU laptop and a CPU server."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available torch device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """bfloat16 on modern GPUs, float16 on older ones, float32 on CPU."""
    device = device or get_device()
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def describe_device() -> str:
    """Human-readable summary of the active device, for logging at run start."""
    device = get_device()
    lines = [f"device: {device.type}", f"dtype:  {get_dtype(device)}"]

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        lines += [
            f"gpu:    {props.name}",
            f"memory: {total_gb:.1f} GiB",
            f"count:  {torch.cuda.device_count()}",
            f"capability: sm_{props.major}{props.minor}",
        ]
    else:
        lines.append(f"threads: {torch.get_num_threads()}")

    lines.append(f"torch:  {torch.__version__}")
    return "\n".join(lines)
