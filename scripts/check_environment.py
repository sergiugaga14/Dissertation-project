# %% [markdown]
# # Environment check
#
# Run cell-by-cell in the Interactive Window (Shift+Enter), or as a script:
#     uv run python scripts/check_environment.py

# %%
import sys
import time

import numpy as np
import torch

from dissertation_project.device import describe_device, get_device, get_dtype

print(sys.executable)
print(describe_device())

# %% [markdown]
# ## Device-agnostic tensor work
#
# Move tensors and models with `.to(device)` rather than `.cuda()`, so the same
# code runs on the CPU server and the GPU laptop.

# %%
device = get_device()
dtype = get_dtype(device)

a = torch.randn(1024, 1024, device=device, dtype=dtype)
b = torch.randn(1024, 1024, device=device, dtype=dtype)
c = a @ b

print(c.shape, c.device, c.dtype)


# %% [markdown]
# ## Quick matmul benchmark

# %%
def benchmark(n: int = 4096, repeats: int = 10) -> float:
    x = torch.randn(n, n, device=device, dtype=dtype)
    y = torch.randn(n, n, device=device, dtype=dtype)

    for _ in range(3):  # warm-up: first call pays kernel autotuning cost
        x @ y
    if device.type == "cuda":
        torch.cuda.synchronize()  # CUDA is async; sync or the timing is meaningless

    start = time.perf_counter()
    for _ in range(repeats):
        x @ y
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return repeats * 2 * n**3 / elapsed / 1e12


print(f"{benchmark():.2f} TFLOP/s on {device.type}")

# %% [markdown]
# ## Reproducibility

# %%
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(torch.randn(3))
