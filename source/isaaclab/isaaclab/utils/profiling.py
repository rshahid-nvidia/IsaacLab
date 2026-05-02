# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small profiling helpers used by optional performance instrumentation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch


def nvtx_range_push(message: str) -> None:
    """Push an NVTX range when CUDA NVTX support is available."""

    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_push(message)


def nvtx_range_pop() -> None:
    """Pop an NVTX range when CUDA NVTX support is available."""

    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Context manager variant of :func:`nvtx_range_push` / :func:`nvtx_range_pop`."""

    nvtx_range_push(message)
    try:
        yield
    finally:
        nvtx_range_pop()
