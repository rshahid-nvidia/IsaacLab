# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch reference parity checks for the fused in-hand Warp paths."""

import argparse
from collections.abc import Sequence
import sys

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.envs.cuda_graph import ResetContext
from isaaclab.utils import has_kit

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env as inhand_env_module
from isaaclab_tasks.utils import compute_kit_requirements, launch_simulation, resolve_task_config

pytestmark = pytest.mark.isaacsim_ci

_ALLEGRO_TASK = "Isaac-Repose-Cube-Allegro-Direct-v0"


@wp.kernel
def _preview_reset_randoms(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    num_dofs: wp.int32,
    goal_rot_rand: wp.array2d(dtype=wp.float32),
    object_pos_rand: wp.array2d(dtype=wp.float32),
    object_rot_rand: wp.array2d(dtype=wp.float32),
    dof_pos_rand: wp.array2d(dtype=wp.float32),
    dof_vel_rand: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    base_state = rng_state[env_id]
    offset = wp.uint32(0)
    state = base_state + offset
    goal_rot_rand[env_id, 0] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    goal_rot_rand[env_id, 1] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    object_pos_rand[env_id, 0] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    object_pos_rand[env_id, 1] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    object_pos_rand[env_id, 2] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    object_rot_rand[env_id, 0] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)
    state = base_state + offset
    object_rot_rand[env_id, 1] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    offset += wp.uint32(1)

    for dof_id in range(num_dofs):
        state = base_state + offset
        dof_pos_rand[env_id, dof_id] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
        offset += wp.uint32(1)
        state = base_state + offset
        dof_vel_rand[env_id, dof_id] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
        offset += wp.uint32(1)


@wp.kernel
def _preview_goal_reset_randoms(
    reset_goal_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    goal_rot_rand: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if not reset_goal_mask[env_id]:
        return

    base_state = rng_state[env_id]
    state = base_state
    goal_rot_rand[env_id, 0] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))
    state = base_state + wp.uint32(1)
    goal_rot_rand[env_id, 1] = wp.randf(state, wp.float32(-1.0), wp.float32(1.0))


def _make_launcher_args() -> argparse.Namespace:
    return argparse.Namespace(
        distributed=False,
        device="cuda:0",
        enable_cameras=False,
        headless=True,
        visualizer=None,
        max_visible_envs=None,
    )


def _make_cfg(num_envs: int = 6):
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], "presets=newton"]
        cfg, _ = resolve_task_config(_ALLEGRO_TASK, None)
    finally:
        sys.argv = old_argv

    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.seed = 29
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.015
    cfg.reset_dof_pos_noise = 0.2
    cfg.reset_dof_vel_noise = 0.25
    return cfg


def _tensor_env_ids(env, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(env_ids, dtype=torch.int32, device=env.device)


def _mask_from_env_ids(env, env_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if len(env_ids) > 0:
        mask[env_ids.to(dtype=torch.long)] = True
    return mask


def _clone_wp_array(array: wp.array) -> torch.Tensor:
    return wp.to_torch(array).clone()


def _copy_wp_array(array: wp.array, value: torch.Tensor) -> None:
    wp.to_torch(array).copy_(value)


def _snapshot_env_state(env) -> dict[str, torch.Tensor]:
    return {
        "episode_length_buf": env.episode_length_buf.clone(),
        "reset_terminated": env.reset_terminated.clone(),
        "reset_time_outs": env.reset_time_outs.clone(),
        "reset_buf": env.reset_buf.clone(),
        "reward_buf": env.reward_buf.clone(),
        "successes": env.successes.clone(),
        "consecutive_successes": env.consecutive_successes.clone(),
        "reset_goal_buf": env.reset_goal_buf.clone(),
        "goal_rot": env.goal_rot.clone(),
        "prev_targets": env.prev_targets.clone(),
        "cur_targets": env.cur_targets.clone(),
        "hand_dof_targets": env.hand_dof_targets.clone(),
        "last_episode_success": env._last_episode_success.clone(),
        "object_root_pose": env.object.data.root_link_pose_w.torch.clone(),
        "object_root_velocity": env.object.data.root_com_vel_w.torch.clone(),
        "hand_joint_pos": env.hand.data.joint_pos.torch.clone(),
        "hand_joint_vel": env.hand.data.joint_vel.torch.clone(),
        "reset_rng_state": _clone_wp_array(env._reset_rng_state_wp),
        "goal_reset_rng_state": _clone_wp_array(env._goal_reset_rng_state_wp),
    }


def _restore_env_state(env, state: dict[str, torch.Tensor]) -> None:
    for name in (
        "episode_length_buf",
        "reset_terminated",
        "reset_time_outs",
        "reset_buf",
        "reward_buf",
        "successes",
        "consecutive_successes",
        "reset_goal_buf",
        "goal_rot",
        "prev_targets",
        "cur_targets",
        "hand_dof_targets",
        "last_episode_success",
    ):
        getattr(env, name if name != "last_episode_success" else "_last_episode_success").copy_(state[name])

    env._write_obj_root_pose(root_pose=state["object_root_pose"])
    env._write_obj_root_vel(root_velocity=state["object_root_velocity"])
    env._set_joint_pos_target(target=state["cur_targets"])
    env._write_hand_joint_pos(position=state["hand_joint_pos"])
    env._write_hand_joint_vel(velocity=state["hand_joint_vel"])
    _copy_wp_array(env._reset_rng_state_wp, state["reset_rng_state"])
    _copy_wp_array(env._goal_reset_rng_state_wp, state["goal_reset_rng_state"])
    env.sim.forward()
    env._compute_intermediate_values()
    if env._inhand_warp_step_enabled:
        env._refresh_inhand_warp_state_inputs()


def _collect_parity_buffers(env) -> dict[str, torch.Tensor]:
    return {
        "episode_length_buf": env.episode_length_buf.clone(),
        "reset_terminated": env.reset_terminated.clone(),
        "reset_time_outs": env.reset_time_outs.clone(),
        "reset_buf": env.reset_buf.clone(),
        "successes": env.successes.clone(),
        "consecutive_successes": env.consecutive_successes.clone(),
        "reward_buf": env.reward_buf.clone(),
        "reset_goal_buf": env.reset_goal_buf.clone(),
        "goal_rot": env.goal_rot.clone(),
        "prev_targets": env.prev_targets.clone(),
        "cur_targets": env.cur_targets.clone(),
        "hand_dof_targets": env.hand_dof_targets.clone(),
        "last_episode_success": env._last_episode_success.clone(),
        "object_root_pose": env.object.data.root_link_pose_w.torch.clone(),
        "object_root_velocity": env.object.data.root_com_vel_w.torch.clone(),
        "hand_joint_pos": env.hand.data.joint_pos.torch.clone(),
        "hand_joint_vel": env.hand.data.joint_vel.torch.clone(),
    }


def _assert_buffers_close(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, expected_value in expected.items():
        if expected_value.dtype == torch.bool:
            torch.testing.assert_close(actual[name], expected_value, msg=name)
        elif expected_value.dtype.is_floating_point:
            torch.testing.assert_close(actual[name], expected_value, rtol=2e-5, atol=2e-5, msg=name)
        else:
            torch.testing.assert_close(actual[name], expected_value, msg=name)


def _preview_reset_samples(env, env_ids: torch.Tensor) -> list[torch.Tensor]:
    env_ids_long = env_ids.to(dtype=torch.long)
    env_mask = _mask_from_env_ids(env, env_ids)
    goal_rot_rand = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    object_pos_rand = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    object_rot_rand = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    dof_pos_rand = torch.zeros((env.num_envs, env.num_hand_dofs), dtype=torch.float32, device=env.device)
    dof_vel_rand = torch.zeros((env.num_envs, env.num_hand_dofs), dtype=torch.float32, device=env.device)

    wp.launch(
        _preview_reset_randoms,
        dim=env.num_envs,
        inputs=[
            wp.from_torch(env_mask, dtype=wp.bool),
            env._reset_rng_state_wp,
            env.num_hand_dofs,
            wp.from_torch(goal_rot_rand, dtype=wp.float32),
            wp.from_torch(object_pos_rand, dtype=wp.float32),
            wp.from_torch(object_rot_rand, dtype=wp.float32),
            wp.from_torch(dof_pos_rand, dtype=wp.float32),
            wp.from_torch(dof_vel_rand, dtype=wp.float32),
        ],
        device=env.device,
    )
    torch.cuda.synchronize()

    return [
        goal_rot_rand[env_ids_long],
        object_pos_rand[env_ids_long],
        object_rot_rand[env_ids_long],
        dof_pos_rand[env_ids_long],
        dof_vel_rand[env_ids_long],
    ]


def _preview_goal_reset_samples(env, reset_goal_mask: torch.Tensor) -> torch.Tensor:
    goal_rot_rand = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    wp.launch(
        _preview_goal_reset_randoms,
        dim=env.num_envs,
        inputs=[
            wp.from_torch(reset_goal_mask, dtype=wp.bool),
            env._goal_reset_rng_state_wp,
            wp.from_torch(goal_rot_rand, dtype=wp.float32),
        ],
        device=env.device,
    )
    torch.cuda.synchronize()
    return goal_rot_rand[reset_goal_mask]


def _patch_sample_uniform(monkeypatch, expected_samples: Sequence[torch.Tensor]) -> None:
    samples = list(expected_samples)

    def _sample_uniform(_lower, _upper, shape, device):
        assert samples, f"Unexpected sample_uniform call for shape {shape}."
        sample = samples.pop(0)
        assert tuple(sample.shape) == tuple(shape)
        assert str(sample.device) == str(torch.device(device))
        return sample.clone()

    monkeypatch.setattr(inhand_env_module, "sample_uniform", _sample_uniform)


def _run_torch_reset_with_previewed_rng(env, env_ids: torch.Tensor, monkeypatch) -> dict[str, torch.Tensor]:
    samples = _preview_reset_samples(env, env_ids)
    with monkeypatch.context() as mp:
        _patch_sample_uniform(mp, samples)
        env._reset_idx_torch(env_ids)
    torch.cuda.synchronize()
    return _collect_parity_buffers(env)


def _run_fused_reset(env, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    env._reset_env_mask.zero_()
    env._reset_env_mask[env_ids.to(dtype=torch.long)] = True
    env._run_inhand_fused_reset(
        ResetContext(env_ids=env_ids, reset_mask_wp=env._reset_env_mask_wp),
        use_cuda_graph=False,
    )
    torch.cuda.synchronize()
    return _collect_parity_buffers(env)


def _configure_reset_scenario(env) -> None:
    env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 3
    env.reset_buf.zero_()
    env.reset_terminated.zero_()
    env.reset_time_outs.zero_()
    env.successes[:] = torch.tensor([0.0, 2.0, 1.0, 4.0, 0.0, 3.0], device=env.device)
    env.consecutive_successes[:] = 1.25
    env.reset_goal_buf[:] = torch.tensor([True, False, True, False, True, False], device=env.device)
    env.prev_targets[:] = env.hand.data.default_joint_pos.torch + 0.05
    env.cur_targets[:] = env.hand.data.default_joint_pos.torch - 0.05
    env.hand_dof_targets[:] = env.cur_targets
    env._last_episode_success.zero_()


def _compare_reset_once(env, monkeypatch, env_ids: torch.Tensor) -> None:
    _configure_reset_scenario(env)
    baseline = _snapshot_env_state(env)
    torch_buffers = _run_torch_reset_with_previewed_rng(env, env_ids, monkeypatch)
    _restore_env_state(env, baseline)
    fused_buffers = _run_fused_reset(env, env_ids)
    _assert_buffers_close(fused_buffers, torch_buffers)


def test_inhand_warp_reset_matches_torch_reference_across_selectors(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_cfg()
    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped
        try:
            gym_env.reset()
            assert env._inhand_fused_reset_enabled

            for env_ids in (
                _tensor_env_ids(env, list(range(env.num_envs))),
                _tensor_env_ids(env, [0, 2, 5]),
            ):
                _compare_reset_once(env, monkeypatch, env_ids)

            repeat_env_ids = _tensor_env_ids(env, [1, 4])
            _compare_reset_once(env, monkeypatch, repeat_env_ids)
            _compare_reset_once(env, monkeypatch, repeat_env_ids)

            baseline = _snapshot_env_state(env)
            env.reset_buf.zero_()
            no_reset_ids = env._reset_idx_from_reset_buf()
            torch.cuda.synchronize()
            assert len(no_reset_ids) == 0
            _assert_buffers_close(_collect_parity_buffers(env), _collect_parity_buffers_from_snapshot(baseline))
        finally:
            gym_env.close()


def _collect_parity_buffers_from_snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "episode_length_buf": state["episode_length_buf"],
        "reset_terminated": state["reset_terminated"],
        "reset_time_outs": state["reset_time_outs"],
        "reset_buf": state["reset_buf"],
        "successes": state["successes"],
        "consecutive_successes": state["consecutive_successes"],
        "reward_buf": state["reward_buf"],
        "reset_goal_buf": state["reset_goal_buf"],
        "goal_rot": state["goal_rot"],
        "prev_targets": state["prev_targets"],
        "cur_targets": state["cur_targets"],
        "hand_dof_targets": state["hand_dof_targets"],
        "last_episode_success": state["last_episode_success"],
        "object_root_pose": state["object_root_pose"],
        "object_root_velocity": state["object_root_velocity"],
        "hand_joint_pos": state["hand_joint_pos"],
        "hand_joint_vel": state["hand_joint_vel"],
    }


def _configure_step_scenario(env, *, mode: str) -> None:
    env.actions = torch.linspace(
        -0.3,
        0.3,
        env.num_envs * env.single_action_space.shape[0],
        dtype=torch.float32,
        device=env.device,
    ).reshape(env.num_envs, env.single_action_space.shape[0])
    env.reset_buf[:] = torch.tensor([False, True, False, False, True, False], device=env.device)
    env.reset_goal_buf.zero_()
    env.successes[:] = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0], device=env.device)
    env.consecutive_successes[:] = 0.75
    env.episode_length_buf[:] = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device=env.device)

    env._get_dones()
    torch.cuda.synchronize()

    env.goal_rot[:] = env.object_rot
    env.object_pos[:] = env.in_hand_pos
    if mode == "fall-only":
        env.object_pos[:, 0] = env.in_hand_pos[:, 0] + env.cfg.fall_dist + 0.2
        env.cfg.success_tolerance = -1.0
    elif mode == "goal-reset":
        env.reset_goal_buf[:] = torch.tensor([True, False, True, False, False, True], device=env.device)
        env.cfg.success_tolerance = 10.0
    elif mode == "max-success":
        env.cfg.max_consecutive_success = 2
        env.cfg.success_tolerance = 10.0
        env.successes[:] = torch.tensor([0.0, 2.0, 1.0, 3.0, 0.0, 4.0], device=env.device)
    else:
        env.cfg.success_tolerance = -1.0


def _run_torch_step_reference(env, monkeypatch, *, preview_goal_samples: bool) -> dict[str, torch.Tensor]:
    expected_samples: list[torch.Tensor] = []
    if preview_goal_samples:
        rot_dist = inhand_env_module.rotation_distance(env.object_rot, env.goal_rot)
        reset_goal_mask = torch.abs(rot_dist) <= env.cfg.success_tolerance
        reset_goal_mask |= env.reset_goal_buf
        expected_samples.append(_preview_goal_reset_samples(env, reset_goal_mask))

    with monkeypatch.context() as mp:
        if expected_samples:
            _patch_sample_uniform(mp, expected_samples)
        terminated, time_outs = env._get_dones_torch()
        env.reset_terminated[:] = terminated
        env.reset_time_outs[:] = time_outs
        torch.logical_or(env.reset_terminated, env.reset_time_outs, out=env.reset_buf)
        reward = env._get_rewards_torch()
        env.reward_buf.copy_(reward)
    torch.cuda.synchronize()
    buffers = _collect_parity_buffers(env)
    buffers["terminated_return"] = terminated.clone()
    buffers["time_outs_return"] = time_outs.clone()
    buffers["reward_return"] = reward.clone()
    return buffers


def _run_fused_step(env) -> dict[str, torch.Tensor]:
    terminated, time_outs = env._get_dones()
    env.reset_terminated[:] = terminated
    env.reset_time_outs[:] = time_outs
    torch.logical_or(env.reset_terminated, env.reset_time_outs, out=env.reset_buf)
    reward = env._get_rewards_warp()
    torch.cuda.synchronize()
    buffers = _collect_parity_buffers(env)
    buffers["terminated_return"] = terminated.clone()
    buffers["time_outs_return"] = time_outs.clone()
    buffers["reward_return"] = reward.clone()
    return buffers


def test_inhand_warp_done_reward_match_torch_reference_across_scenarios(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_cfg()
    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped
        try:
            gym_env.reset()
            assert env._inhand_warp_step_enabled

            for mode, preview_goal_samples in (
                ("no-reset", False),
                ("fall-only", False),
                ("goal-reset", True),
                ("max-success", True),
            ):
                _configure_step_scenario(env, mode=mode)
                baseline = _snapshot_env_state(env)
                torch_buffers = _run_torch_step_reference(env, monkeypatch, preview_goal_samples=preview_goal_samples)
                _restore_env_state(env, baseline)
                fused_buffers = _run_fused_step(env)
                _assert_buffers_close(fused_buffers, torch_buffers)
        finally:
            gym_env.close()
