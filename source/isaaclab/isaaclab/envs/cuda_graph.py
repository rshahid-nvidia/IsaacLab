# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers for CUDA graph based environment paths."""

from __future__ import annotations

import contextlib
import ctypes
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch
import warp as wp


logger = logging.getLogger(__name__)

try:
    _cudart = ctypes.CDLL("libcudart.so.12")
except OSError:
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        _cudart = None

if _cudart is None:
    logger.warning(
        "CUDA runtime library was not found; relaxed CUDA graph capture is unavailable and reset graph capture will "
        "fall back to Warp's strict ScopedCapture."
    )


@dataclass(frozen=True)
class ResetContext:
    """Reset inputs shared by fused reset cores and optional CUDA graph launchers."""

    env_ids: torch.Tensor | None
    reset_mask_wp: wp.array


@dataclass(frozen=True)
class CudaGraphTensorSignature:
    """Tensor properties that a captured CUDA graph depends on."""

    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> "CudaGraphTensorSignature":
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
    def capture(cls, array: wp.array) -> "CudaGraphWarpArraySignature":
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


def capture_cuda_graph_relaxed(device: str, launch_fn: Callable[[], None]):
    """Capture ``launch_fn`` into a CUDA graph using relaxed stream-capture mode when possible."""

    if _cudart is None:
        with wp.ScopedCapture() as capture:
            launch_fn()
        return capture.graph

    raw_handle = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreateWithFlags(ctypes.byref(raw_handle), ctypes.c_uint(0x01))
    if ret != 0:
        raise RuntimeError(f"cudaStreamCreateWithFlags(cudaStreamNonBlocking) failed with code {ret}")
    stream_handle = raw_handle.value
    fresh_stream = wp.Stream(device, cuda_stream=stream_handle, owner=False)

    ret = _cudart.cudaStreamBeginCapture(ctypes.c_void_p(stream_handle), ctypes.c_int(2))
    if ret != 0:
        _cudart.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
        raise RuntimeError(f"cudaStreamBeginCapture(cudaStreamCaptureModeRelaxed) failed with code {ret}")

    try:
        wp.capture_begin(stream=fresh_stream, external=True)
    except Exception:
        raw_graph = ctypes.c_void_p()
        _cudart.cudaStreamEndCapture(ctypes.c_void_p(stream_handle), ctypes.byref(raw_graph))
        if raw_graph.value:
            _cudart.cudaGraphDestroy(raw_graph)
        _cudart.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
        raise

    graph = None
    error: Exception | None = None
    with wp.ScopedStream(fresh_stream, sync_enter=False):
        try:
            launch_fn()
        except Exception as exc:
            error = exc

    if error is None:
        try:
            graph = wp.capture_end(stream=fresh_stream)
        except Exception as exc:
            error = exc
    else:
        with contextlib.suppress(Exception):
            wp.capture_end(stream=fresh_stream)

    raw_graph = ctypes.c_void_p()
    end_ret = _cudart.cudaStreamEndCapture(ctypes.c_void_p(stream_handle), ctypes.byref(raw_graph))
    _cudart.cudaStreamDestroy(ctypes.c_void_p(stream_handle))

    if error is not None:
        if raw_graph.value:
            _cudart.cudaGraphDestroy(raw_graph)
        raise error
    if end_ret != 0 or not raw_graph.value:
        raise RuntimeError(f"cudaStreamEndCapture failed with code {end_ret}")

    graph.graph = raw_graph
    graph.graph_exec = None
    return graph


def launch_cuda_graph_on_current_torch_stream(device: str, graph) -> wp.Stream:
    """Launch ``graph`` on the current PyTorch CUDA stream and return the Warp stream wrapper."""

    torch_stream = torch.cuda.current_stream(torch.device(device))
    replay_stream = wp.Stream(device, cuda_stream=torch_stream.cuda_stream, owner=False)
    wp.capture_launch(graph, stream=replay_stream)
    return replay_stream
