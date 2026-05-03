# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp

import isaaclab.utils.cuda_graph as cuda_graph_utils
from isaaclab.envs.cuda_graph import CudaGraphReplayGuard
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_unique


def test_cuda_graph_replay_guard_accepts_unchanged_tensors_and_values():
    tensor = torch.zeros(4, 3)
    guard = CudaGraphReplayGuard(tensors={"tensor": tensor}, values={"count": 4, "scale": 0.5})

    assert guard.check(tensors={"tensor": tensor}, values={"count": 4, "scale": 0.5}) is None


def test_cuda_graph_replay_guard_detects_storage_change():
    tensor = torch.zeros(4, 3)
    guard = CudaGraphReplayGuard(tensors={"tensor": tensor})

    assert guard.check(tensors={"tensor": tensor.clone()}) == "tensor storage changed"


def test_cuda_graph_replay_guard_detects_metadata_change_with_same_storage():
    tensor = torch.zeros(4, 3)
    guard = CudaGraphReplayGuard(tensors={"tensor": tensor})

    assert guard.check(tensors={"tensor": tensor.transpose(0, 1)}) == "tensor metadata changed"


def test_cuda_graph_replay_guard_detects_value_change():
    tensor = torch.zeros(4)
    guard = CudaGraphReplayGuard(tensors={"tensor": tensor}, values={"num_envs": 4})

    assert guard.check(tensors={"tensor": tensor}, values={"num_envs": 5}) == "num_envs changed from 4 to 5"


def test_cuda_graph_replay_guard_detects_missing_and_unexpected_keys():
    tensor = torch.zeros(4)
    guard = CudaGraphReplayGuard(tensors={"tensor": tensor}, values={"scale": 1.0})

    assert guard.check(tensors={}, values={"scale": 1.0}) == "tensor missing"
    assert guard.check(tensors={"tensor": tensor, "extra": tensor}, values={"scale": 1.0}) == "extra was not captured"
    assert guard.check(tensors={"tensor": tensor}, values={}) == "scale missing"
    assert guard.check(tensors={"tensor": tensor}, values={"scale": 1.0, "extra": 2.0}) == "extra was not captured"


def test_cuda_graph_replay_guard_accepts_unchanged_warp_arrays():
    array = wp.zeros((4, 3), dtype=wp.float32, device="cpu")
    guard = CudaGraphReplayGuard(tensors={"array": array})

    assert guard.check(tensors={"array": array}) is None


def test_cuda_graph_replay_guard_detects_warp_array_storage_change():
    array = wp.zeros((4, 3), dtype=wp.float32, device="cpu")
    guard = CudaGraphReplayGuard(tensors={"array": array})

    assert guard.check(tensors={"array": wp.zeros((4, 3), dtype=wp.float32, device="cpu")}) == "array storage changed"


def test_relaxed_cuda_graph_capture_requires_cudart_without_explicit_fallback(monkeypatch):
    monkeypatch.setattr(cuda_graph_utils, "_cudart", None)
    monkeypatch.setattr(cuda_graph_utils, "_cudart_soname", None)

    with pytest.raises(cuda_graph_utils.CudaGraphCaptureError, match="libcudart"):
        cuda_graph_utils.capture_cuda_graph_relaxed("cuda:0", lambda: None)


def test_relaxed_cuda_graph_capture_replays_torch_ops_on_capture_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for relaxed CUDA graph replay.")
    if not cuda_graph_utils.relaxed_cuda_graph_capture_available():
        pytest.skip("CUDA runtime library is required for relaxed CUDA graph capture.")

    device = "cuda:0"
    values = torch.zeros(4, device=device)
    values.add_(1.0)
    values.zero_()
    torch.cuda.synchronize()

    graph = cuda_graph_utils.capture_cuda_graph_relaxed(device, lambda: values.add_(1.0))
    torch.cuda.synchronize()

    values.zero_()
    cuda_graph_utils.launch_cuda_graph_on_current_torch_stream(device, graph)
    torch.cuda.synchronize()
    torch.testing.assert_close(values, torch.ones_like(values))

    cuda_graph_utils.launch_cuda_graph_on_current_torch_stream(device, graph)
    torch.cuda.synchronize()
    torch.testing.assert_close(values, torch.full_like(values, 2.0))


def test_relaxed_cuda_graph_capture_replays_quat_from_matrix():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for relaxed CUDA graph replay.")
    if not cuda_graph_utils.relaxed_cuda_graph_capture_available():
        pytest.skip("CUDA runtime library is required for relaxed CUDA graph capture.")

    device = "cuda:0"
    rotations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.38268343, 0.0, 0.0, 0.9238795],
            [0.0, -0.70710677, 0.0, 0.70710677],
            [0.0, 0.0, 0.9238795, 0.38268343],
        ],
        device=device,
    )
    matrices = matrix_from_quat(rotations).contiguous()
    output = torch.empty_like(rotations)

    graph = cuda_graph_utils.capture_cuda_graph_relaxed(
        device,
        lambda: output.copy_(quat_from_matrix(matrices)),
    )

    new_rotations = torch.tensor(
        [
            [0.0, 0.0, 0.70710677, 0.70710677],
            [-0.38268343, 0.0, 0.0, 0.9238795],
            [0.0, 0.25881904, 0.0, 0.9659258],
            [0.5, 0.5, 0.5, 0.5],
        ],
        device=device,
    )
    matrices.copy_(matrix_from_quat(new_rotations))
    output.zero_()

    cuda_graph_utils.launch_cuda_graph_on_current_torch_stream(device, graph)
    torch.cuda.synchronize()

    torch.testing.assert_close(quat_unique(output), quat_unique(new_rotations), rtol=1e-5, atol=1e-5)
