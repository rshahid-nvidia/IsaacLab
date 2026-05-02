# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for maintaining Torch-backed Warp views.

Fused Warp task paths usually need the same tensor in three places: a Torch tensor for regular task code, a Warp view
for kernels, and a replay-guard entry when the kernels are captured in a CUDA graph. This registry keeps those bindings
declared once so refresh, compatibility checks, and graph-guard target lists cannot drift independently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

GraphTarget = torch.Tensor | wp.array
ShapeSpec = tuple[int, ...] | Callable[[], tuple[int, ...]]


@dataclass(frozen=True)
class _WarpViewSpec:
    name: str
    groups: frozenset[str]
    tensor_getter: Callable[[], torch.Tensor] | None
    torch_dtype: torch.dtype | None
    expected_shape: ShapeSpec | None
    warp_attr: str | None
    warp_dtype: Any | None
    warp_getter: Callable[[], wp.array] | None
    track_torch: bool
    track_warp: bool
    warp_name: str | None


def _resolve_shape(shape: ShapeSpec | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if callable(shape):
        return tuple(shape())
    return tuple(shape)


class CudaGraphTargetRegistry:
    """Small name-to-getter registry for CUDA graph replay-guard targets."""

    def __init__(self) -> None:
        self._targets: dict[str, Callable[[], GraphTarget]] = {}

    def register(self, name: str, getter: Callable[[], GraphTarget]) -> None:
        """Track one replay-guard target by name."""

        if name in self._targets:
            raise ValueError(f"CUDA graph target {name!r} is already registered.")
        self._targets[name] = getter

    def update(self, targets: Mapping[str, Callable[[], GraphTarget]]) -> None:
        """Track several replay-guard targets."""

        for name, getter in targets.items():
            self.register(name, getter)

    def tensors(self) -> dict[str, GraphTarget]:
        """Return the current replay-guard target mapping."""

        return {name: getter() for name, getter in self._targets.items()}


class WarpViewRegistry:
    """Declare, refresh, validate, and track Warp views from one source of truth."""

    def __init__(self, owner: object, *, device: str | torch.device) -> None:
        self._owner = owner
        self._device = torch.device(device)
        self._specs: list[_WarpViewSpec] = []

    def register_torch_view(
        self,
        name: str,
        tensor_getter: Callable[[], torch.Tensor],
        *,
        warp_attr: str,
        warp_dtype: Any,
        torch_dtype: torch.dtype,
        expected_shape: ShapeSpec,
        groups: Iterable[str],
        track_torch: bool = True,
        track_warp: bool = True,
        warp_name: str | None = None,
    ) -> None:
        """Register a Warp view created with :func:`warp.from_torch`."""

        self._register(
            name,
            tensor_getter=tensor_getter,
            torch_dtype=torch_dtype,
            expected_shape=expected_shape,
            warp_attr=warp_attr,
            warp_dtype=warp_dtype,
            warp_getter=None,
            groups=groups,
            track_torch=track_torch,
            track_warp=track_warp,
            warp_name=warp_name,
        )

    def register_existing_view(
        self,
        name: str,
        tensor_getter: Callable[[], torch.Tensor],
        warp_getter: Callable[[], wp.array],
        *,
        torch_dtype: torch.dtype,
        expected_shape: ShapeSpec,
        groups: Iterable[str],
        warp_attr: str | None = None,
        track_torch: bool = False,
        track_warp: bool = True,
        warp_name: str | None = None,
    ) -> None:
        """Register an existing backend-owned Warp view, such as a ProxyArray ``.warp`` view."""

        self._register(
            name,
            tensor_getter=tensor_getter,
            torch_dtype=torch_dtype,
            expected_shape=expected_shape,
            warp_attr=warp_attr,
            warp_dtype=None,
            warp_getter=warp_getter,
            groups=groups,
            track_torch=track_torch,
            track_warp=track_warp,
            warp_name=warp_name,
        )

    def _register(
        self,
        name: str,
        *,
        tensor_getter: Callable[[], torch.Tensor] | None,
        torch_dtype: torch.dtype | None,
        expected_shape: ShapeSpec | None,
        warp_attr: str | None,
        warp_dtype: Any | None,
        warp_getter: Callable[[], wp.array] | None,
        groups: Iterable[str],
        track_torch: bool,
        track_warp: bool,
        warp_name: str | None,
    ) -> None:
        group_set = frozenset(groups)
        if not group_set:
            raise ValueError(f"Warp view {name!r} must belong to at least one group.")
        if track_warp and warp_attr is None and warp_getter is None:
            raise ValueError(f"Warp view {name!r} cannot track a Warp target without a view source.")
        self._specs.append(
            _WarpViewSpec(
                name=name,
                groups=group_set,
                tensor_getter=tensor_getter,
                torch_dtype=torch_dtype,
                expected_shape=expected_shape,
                warp_attr=warp_attr,
                warp_dtype=warp_dtype,
                warp_getter=warp_getter,
                track_torch=track_torch,
                track_warp=track_warp,
                warp_name=warp_name,
            )
        )

    def refresh(self, *, groups: Iterable[str] | None = None) -> None:
        """Refresh registered Warp view attributes for the selected groups."""

        for spec in self._iter_specs(groups):
            if spec.warp_attr is None:
                continue
            if spec.warp_getter is not None:
                view = spec.warp_getter()
            else:
                assert spec.tensor_getter is not None and spec.warp_dtype is not None
                view = wp.from_torch(spec.tensor_getter(), dtype=spec.warp_dtype)
            setattr(self._owner, spec.warp_attr, view)

    def check_compatibility(self, *, groups: Iterable[str] | None = None) -> str | None:
        """Return the first tensor compatibility issue, or ``None`` if all selected views are usable."""

        for spec in self._iter_specs(groups):
            if spec.tensor_getter is None:
                continue
            tensor = spec.tensor_getter()
            if not torch.is_tensor(tensor):
                return f"{spec.name} is not a torch.Tensor"
            expected_shape = _resolve_shape(spec.expected_shape)
            if expected_shape is not None and tuple(tensor.shape) != expected_shape:
                return f"{spec.name} shape changed from {expected_shape} to {tuple(tensor.shape)}"
            if spec.torch_dtype is not None and tensor.dtype != spec.torch_dtype:
                return f"{spec.name} dtype changed from {spec.torch_dtype} to {tensor.dtype}"
            if tensor.device != self._device:
                return f"{spec.name} device changed from {self._device} to {tensor.device}"
        return None

    def graph_tensors(self, *, groups: Iterable[str] | None = None) -> dict[str, GraphTarget]:
        """Return replay-guard targets for the selected groups."""

        tensors: dict[str, GraphTarget] = {}
        for spec in self._iter_specs(groups):
            if spec.track_torch:
                if spec.tensor_getter is None:
                    raise RuntimeError(f"Warp view {spec.name!r} has no Torch tensor getter to track.")
                tensors[spec.name] = spec.tensor_getter()
            if spec.track_warp:
                tensors[spec.warp_name or f"{spec.name}_wp"] = self._warp_view(spec)
        return tensors

    def _warp_view(self, spec: _WarpViewSpec) -> wp.array:
        if spec.warp_attr is not None and hasattr(self._owner, spec.warp_attr):
            return getattr(self._owner, spec.warp_attr)
        if spec.warp_getter is not None:
            return spec.warp_getter()
        if spec.tensor_getter is not None and spec.warp_dtype is not None:
            return wp.from_torch(spec.tensor_getter(), dtype=spec.warp_dtype)
        raise RuntimeError(f"Warp view {spec.name!r} has no view source.")

    def _iter_specs(self, groups: Iterable[str] | None):
        if groups is None:
            yield from self._specs
            return

        group_set = set(groups)
        for spec in self._specs:
            if spec.groups & group_set:
                yield spec
