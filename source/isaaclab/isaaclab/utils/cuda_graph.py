# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level CUDA graph helpers shared by profiling-sensitive paths."""

from __future__ import annotations

import contextlib
import ctypes
from collections.abc import Callable

import torch
import warp as wp

CUDA_RUNTIME_SONAMES = ("libcudart.so.12", "libcudart.so")
CUDA_STREAM_NON_BLOCKING = 0x01
CUDA_STREAM_CAPTURE_MODE_RELAXED = 2


class CudaGraphCaptureError(RuntimeError):
    """Raised when a low-level CUDA graph capture operation fails."""


def _load_cudart():
    for soname in CUDA_RUNTIME_SONAMES:
        try:
            return ctypes.CDLL(soname), soname
        except OSError:
            pass
    return None, None


_cudart, _cudart_soname = _load_cudart()


def relaxed_cuda_graph_capture_available() -> bool:
    """Return whether relaxed CUDA stream capture can use the CUDA runtime library."""

    return _cudart is not None


def cuda_runtime_soname() -> str | None:
    """Return the CUDA runtime soname used for relaxed capture, if available."""

    return _cudart_soname


def _destroy_raw_graph(raw_graph: ctypes.c_void_p) -> None:
    if _cudart is not None and raw_graph.value:
        _cudart.cudaGraphDestroy(raw_graph)


def _create_nonblocking_stream() -> int:
    if _cudart is None:
        raise CudaGraphCaptureError(
            "CUDA runtime library is not available for relaxed graph capture "
            f"(tried {', '.join(CUDA_RUNTIME_SONAMES)})."
        )

    raw_handle = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreateWithFlags(ctypes.byref(raw_handle), ctypes.c_uint(CUDA_STREAM_NON_BLOCKING))
    if ret != 0 or raw_handle.value is None:
        raise CudaGraphCaptureError(f"cudaStreamCreateWithFlags(cudaStreamNonBlocking) failed with code {ret}")
    return int(raw_handle.value)


def capture_cuda_graph_relaxed(
    device: str,
    launch_fn: Callable[[], None],
    *,
    fallback_to_scoped_capture: bool = False,
):
    """Capture ``launch_fn`` into a CUDA graph using relaxed stream-capture mode.

    This is the shared implementation of the RTX-compatible capture pattern used by Newton physics and reset graph
    paths. It relies on Warp's ``wp.capture_begin(external=True)`` contract: CUDA stream capture is started directly
    through cudart, then Warp is told to register the active external capture so Warp control-flow helpers and graph
    bookkeeping see a valid capture. If that interop changes upstream, this helper is the only place that should need
    adjustment.
    """

    if _cudart is None:
        if fallback_to_scoped_capture:
            with wp.ScopedCapture() as capture:
                launch_fn()
            return capture.graph
        raise CudaGraphCaptureError(
            "relaxed CUDA graph capture unavailable because libcudart could not be loaded "
            f"(tried {', '.join(CUDA_RUNTIME_SONAMES)})"
        )

    stream_handle = _create_nonblocking_stream()
    fresh_stream = wp.Stream(device, cuda_stream=stream_handle, owner=False)

    ret = _cudart.cudaStreamBeginCapture(ctypes.c_void_p(stream_handle), ctypes.c_int(CUDA_STREAM_CAPTURE_MODE_RELAXED))
    if ret != 0:
        _cudart.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
        raise CudaGraphCaptureError(f"cudaStreamBeginCapture(cudaStreamCaptureModeRelaxed) failed with code {ret}")

    try:
        wp.capture_begin(stream=fresh_stream, external=True)
    except Exception as exc:
        raw_graph = ctypes.c_void_p()
        with contextlib.suppress(Exception):
            _cudart.cudaStreamEndCapture(ctypes.c_void_p(stream_handle), ctypes.byref(raw_graph))
        _destroy_raw_graph(raw_graph)
        _cudart.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
        raise CudaGraphCaptureError(f"wp.capture_begin(external=True) failed: {exc}") from exc

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
        _destroy_raw_graph(raw_graph)
        raise CudaGraphCaptureError(f"CUDA graph capture body failed: {error}") from error
    if end_ret != 0 or not raw_graph.value:
        raise CudaGraphCaptureError(f"cudaStreamEndCapture failed with code {end_ret}")
    if graph is None:
        _destroy_raw_graph(raw_graph)
        raise CudaGraphCaptureError("Warp capture did not return a graph")

    # Warp's external capture object is only bookkeeping for device.captures. The raw CUDA graph returned by
    # cudaStreamEndCapture is the graph that must be instantiated/launched.
    graph.graph = raw_graph
    graph.graph_exec = None
    return graph


def launch_cuda_graph_on_current_torch_stream(device: str, graph) -> wp.Stream:
    """Launch ``graph`` on the current PyTorch CUDA stream and return the Warp stream wrapper."""

    torch_stream = torch.cuda.current_stream(torch.device(device))
    replay_stream = wp.Stream(device, cuda_stream=torch_stream.cuda_stream, owner=False)
    wp.capture_launch(graph, stream=replay_stream)
    return replay_stream
