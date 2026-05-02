# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers for CUDA graph based environment paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from isaaclab.utils.cuda_graph import CudaGraphCaptureError
from isaaclab.utils.reset import ResetSelection

__all__ = [
    "CudaGraphCaptureError",
    "CudaGraphReplayGuard",
    "ResetContext",
    "ResetGraphPhase",
]


@dataclass(frozen=True)
class ResetContext:
    """Reset inputs shared by fused reset cores and optional CUDA graph launchers."""

    env_ids: torch.Tensor | None
    reset_mask_wp: wp.array

    @property
    def selection(self) -> ResetSelection:
        """Return the normalized reset selector represented by this context."""

        return ResetSelection(env_ids=self.env_ids, env_mask=self.reset_mask_wp)

    def with_env_ids(self, env_ids: torch.Tensor) -> ResetContext:
        """Return a copy with concrete environment ids materialized."""

        return type(self)(env_ids=env_ids, reset_mask_wp=self.reset_mask_wp)


@dataclass(frozen=True)
class ResetGraphPhase:
    """One ordered CUDA graph phase in a split reset replay sequence.

    ``between_hook`` runs on the replay stream after this phase launches and before the next phase launches. It returns
    the possibly-updated reset context plus whether replay should continue to later phases.
    """

    graph_attr: str
    name: str
    launch_fn: Callable[[ResetContext], None]
    prepare_capture: Callable[[], None] | None = None
    between_hook: Callable[[ResetContext], tuple[ResetContext, bool]] | None = None


@dataclass(frozen=True)
class CudaGraphTensorSignature:
    """Tensor properties that a captured CUDA graph depends on."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> CudaGraphTensorSignature:
        """Capture the replay-relevant identity and metadata of ``tensor``."""

        return cls(
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )

    def check(self, tensor: torch.Tensor) -> str | None:
        """Return a short mismatch reason, or ``None`` if ``tensor`` still matches."""

        current = type(self).capture(tensor)
        if current.data_ptr != self.data_ptr:
            return "storage changed"
        if current != self:
            return "metadata changed"
        return None


@dataclass(frozen=True)
class CudaGraphWarpArraySignature:
    """Warp array properties that a captured CUDA graph depends on."""

    data_ptr: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def capture(cls, array: wp.array) -> CudaGraphWarpArraySignature:
        """Capture the replay-relevant identity and metadata of ``array``."""

        return cls(
            data_ptr=int(array.ptr),
            shape=tuple(array.shape),
            strides=tuple(array.strides),
            dtype=str(array.dtype),
            device=str(array.device),
        )

    def check(self, array: wp.array) -> str | None:
        """Return a short mismatch reason, or ``None`` if ``array`` still matches."""

        current = type(self).capture(array)
        if current.data_ptr != self.data_ptr:
            return "storage changed"
        if current != self:
            return "metadata changed"
        return None


CudaGraphCaptureTarget = torch.Tensor | wp.array
CudaGraphCaptureSignature = CudaGraphTensorSignature | CudaGraphWarpArraySignature


def _capture_graph_target(target: CudaGraphCaptureTarget) -> CudaGraphCaptureSignature:
    if isinstance(target, torch.Tensor):
        return CudaGraphTensorSignature.capture(target)
    if isinstance(target, wp.array):
        return CudaGraphWarpArraySignature.capture(target)
    raise TypeError(f"Unsupported CUDA graph replay target type: {type(target).__name__}")


class CudaGraphReplayGuard:
    """Validate that replay-time tensors and constants still match capture-time assumptions.

    The guard is intentionally small: task code owns the list of tensors and scalar values that matter for its graph,
    while this helper gives every task the same conservative checking semantics and error strings.
    """

    def __init__(self, *, tensors: Mapping[str, CudaGraphCaptureTarget], values: Mapping[str, Any] | None = None):
        self._tensor_signatures = {name: _capture_graph_target(tensor) for name, tensor in tensors.items()}
        self._values = dict(values or {})

    def check(
        self, *, tensors: Mapping[str, CudaGraphCaptureTarget], values: Mapping[str, Any] | None = None
    ) -> str | None:
        """Return the first violated replay assumption, or ``None`` if replay is safe."""

        values = values or {}
        for name, signature in self._tensor_signatures.items():
            tensor = tensors.get(name)
            if tensor is None:
                return f"{name} missing"
            mismatch = signature.check(tensor)
            if mismatch is not None:
                return f"{name} {mismatch}"
        for name in tensors:
            if name not in self._tensor_signatures:
                return f"{name} was not captured"

        for name, expected in self._values.items():
            if name not in values:
                return f"{name} missing"
            current = values[name]
            if current != expected:
                return f"{name} changed from {expected} to {current}"
        for name in values:
            if name not in self._values:
                return f"{name} was not captured"

        return None
