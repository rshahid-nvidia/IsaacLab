# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.cuda_graph import CudaGraphReplayGuard


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
