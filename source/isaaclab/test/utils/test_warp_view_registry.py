# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp

from isaaclab.envs.cuda_graph import CudaGraphReplayGuard
from isaaclab.utils.warp_view_registry import CudaGraphTargetRegistry, WarpViewRegistry

wp.init()


class _Owner:
    pass


def test_warp_view_registry_refreshes_checks_and_tracks_torch_backed_views():
    owner = _Owner()
    state = {"tensor": torch.zeros(2, 3, dtype=torch.float32)}
    registry = WarpViewRegistry(owner, device="cpu")

    registry.register_torch_view(
        "points",
        lambda: state["tensor"],
        warp_attr="_points_wp",
        warp_dtype=wp.vec3f,
        torch_dtype=torch.float32,
        expected_shape=(2, 3),
        groups=("step", "graph"),
    )

    assert registry.check_compatibility(groups=("step",)) is None
    registry.refresh(groups=("step",))
    assert hasattr(owner, "_points_wp")

    graph_tensors = registry.graph_tensors(groups=("graph",))
    assert graph_tensors["points"] is state["tensor"]
    assert graph_tensors["points_wp"] is owner._points_wp
    assert CudaGraphReplayGuard(tensors=graph_tensors).check(tensors=graph_tensors) is None

    state["tensor"] = torch.zeros(3, 3, dtype=torch.float32)
    assert registry.check_compatibility(groups=("step",)) == "points shape changed from (2, 3) to (3, 3)"


def test_warp_view_registry_tracks_existing_backend_views():
    owner = _Owner()
    array = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    tensor = wp.to_torch(array)
    registry = WarpViewRegistry(owner, device="cpu")

    registry.register_existing_view(
        "backend.buffer",
        lambda: tensor,
        lambda: array,
        torch_dtype=torch.float32,
        expected_shape=(2, 3),
        groups=("state", "graph"),
        warp_attr="_backend_buffer_wp",
        warp_name="backend.buffer_wp",
    )

    registry.refresh(groups=("state",))
    assert owner._backend_buffer_wp is array
    graph_tensors = registry.graph_tensors(groups=("graph",))
    assert list(graph_tensors) == ["backend.buffer_wp"]
    assert graph_tensors["backend.buffer_wp"] is array


def test_cuda_graph_target_registry_uses_latest_getter_value_and_rejects_duplicates():
    state = {"tensor": torch.zeros(1)}
    registry = CudaGraphTargetRegistry()
    registry.register("tensor", lambda: state["tensor"])

    first = registry.tensors()
    assert first["tensor"] is state["tensor"]

    state["tensor"] = torch.ones(1)
    second = registry.tensors()
    assert second["tensor"] is state["tensor"]
    assert second["tensor"].item() == 1.0

    with pytest.raises(ValueError, match="already registered"):
        registry.register("tensor", lambda: state["tensor"])
