# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import warp as wp

from isaaclab.utils.reset import ResetSelection


def test_reset_selection_mask_takes_precedence_over_env_ids():
    env_mask = wp.array([False, True, False, True], dtype=wp.bool, device="cpu")
    selection = ResetSelection(env_ids=torch.tensor([0, 2]), env_mask=env_mask)

    env_ids = selection.materialize_env_ids(device="cpu", dtype=torch.long)

    torch.testing.assert_close(env_ids, torch.tensor([1, 3], dtype=torch.long))


def test_reset_selection_full_residual_selector_is_explicit():
    selection = ResetSelection()

    assert selection.residual_kwargs(full_selector=slice(None)) == {"env_ids": slice(None), "env_mask": None}


def test_reset_selection_preserves_mask_for_non_materializing_residuals():
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    selection = ResetSelection(env_ids=torch.tensor([1]), env_mask=env_mask)

    kwargs = selection.residual_kwargs(full_selector=slice(None))

    torch.testing.assert_close(kwargs["env_ids"], torch.tensor([1]))
    assert kwargs["env_mask"] is env_mask


def test_reset_selection_accepts_warp_array_ids():
    env_ids_wp = wp.array([0, 2], dtype=wp.int32, device="cpu")
    selection = ResetSelection(env_ids=env_ids_wp)

    env_ids = selection.materialize_env_ids(device="cpu", dtype=torch.long)

    torch.testing.assert_close(env_ids, torch.tensor([0, 2], dtype=torch.long))
