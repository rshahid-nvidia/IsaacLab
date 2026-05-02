# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for keeping reset selector semantics consistent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import warp as wp


ResetEnvIds = Sequence[int] | torch.Tensor | wp.array | slice | None


def is_full_reset_selector(env_ids: ResetEnvIds, env_mask: wp.array | None = None) -> bool:
    """Return whether the selector denotes an all-environment reset without a mask."""

    return env_mask is None and (env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)))


@dataclass(frozen=True)
class ResetSelection:
    """Normalized reset selectors with one precedence rule.

    Public reset APIs still accept ``env_ids`` and ``env_mask`` for compatibility. Internally, this helper makes the
    selector rule explicit: when ``env_mask`` is provided, it is the source of truth. Concrete environment ids are
    materialized lazily only for residual APIs that still require indexed resets.
    """

    env_ids: ResetEnvIds = None
    env_mask: wp.array | None = None

    def graph_kwargs(self) -> dict[str, ResetEnvIds | wp.array | None]:
        """Return keyword arguments for graphable/reset-after-graph APIs."""

        return {"env_ids": self.env_ids, "env_mask": self.env_mask}

    def materialize_env_ids(
        self,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.int32,
        full_selector: ResetEnvIds = None,
    ) -> ResetEnvIds:
        """Return concrete ids for env-id-only residual APIs.

        If a mask is present it takes precedence over ``env_ids``. A full unmasked reset returns ``full_selector`` so
        callers can preserve the legacy convention expected by their downstream API, for example ``None`` for a classic
        full reset or ``slice(None)`` for indexed-kernel full reset.
        """

        if self.env_mask is not None:
            return wp.to_torch(self.env_mask).nonzero(as_tuple=False).squeeze(-1).to(device=device, dtype=dtype)
        if self.env_ids is None:
            return full_selector
        if isinstance(self.env_ids, slice):
            return self.env_ids
        if isinstance(self.env_ids, wp.array):
            return wp.to_torch(self.env_ids).to(device=device, dtype=dtype)
        if torch.is_tensor(self.env_ids):
            return self.env_ids.to(device=device, dtype=dtype)
        return torch.tensor(self.env_ids, dtype=dtype, device=device)

    def residual_kwargs(self, *, full_selector: ResetEnvIds = None) -> dict[str, ResetEnvIds | wp.array | None]:
        """Return selectors for residual hooks that do not require materialized ids.

        This preserves mask-native residual paths and only substitutes ``full_selector`` for the no-arg full-reset
        convention. Use :meth:`materialize_env_ids` instead for residual APIs that are strictly env-id based.
        """

        if self.env_ids is None and self.env_mask is None:
            return {"env_ids": full_selector, "env_mask": None}
        return self.graph_kwargs()
