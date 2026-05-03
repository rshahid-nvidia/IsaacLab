# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness checks for the opt-in in-hand reset CUDA graph path."""

import argparse
import sys

import gymnasium as gym
import pytest
import torch
import warp as wp

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import has_kit
from isaaclab.utils.configclass import configclass
from isaaclab.utils.cuda_graph import capture_cuda_graph_relaxed, launch_cuda_graph_on_current_torch_stream
from isaaclab.utils.math import quat_conjugate, quat_mul
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import compute_kit_requirements, launch_simulation, resolve_task_config

pytestmark = pytest.mark.isaacsim_ci

_ALLEGRO_TASK = "Isaac-Repose-Cube-Allegro-Direct-v0"
_SHADOW_TASK = "Isaac-Repose-Cube-Shadow-Direct-v0"
_SHADOW_VISION_BENCHMARK_TASK = "Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0"


def _noop_event(env, env_ids):
    pass


def _record_reset_event(env, env_ids):
    env._test_reset_event_calls = getattr(env, "_test_reset_event_calls", 0) + 1
    env._test_reset_event_env_ids = torch.as_tensor(env_ids, device=env.device).clone()


def _record_ordered_reset_event(env, env_ids):
    if not hasattr(env, "_test_reset_order"):
        env._test_reset_order = []
    env._test_reset_order.append("event")
    env._test_reset_event_env_ids = torch.as_tensor(env_ids, device=env.device).clone()


def _clear_successes_reset_event(env, env_ids):
    if not hasattr(env, "_test_reset_order"):
        env._test_reset_order = []
    env._test_reset_order.append("event")
    env._test_reset_event_env_ids = torch.as_tensor(env_ids, device=env.device).clone()
    env.successes[env._test_reset_event_env_ids.to(dtype=torch.long)] = 0.0


@configclass
class _ResetEventCfg:
    reset = EventTerm(func=_noop_event, mode="reset")


@configclass
class _RecordResetEventCfg:
    reset = EventTerm(func=_record_reset_event, mode="reset")


@configclass
class _RecordOrderedResetEventCfg:
    reset = EventTerm(func=_record_ordered_reset_event, mode="reset")


@configclass
class _ClearSuccessesResetEventCfg:
    reset = EventTerm(func=_clear_successes_reset_event, mode="reset")


@configclass
class _IntervalEventCfg:
    interval = EventTerm(func=_noop_event, mode="interval", interval_range_s=(3600.0, 3600.0))


def _warp_buffer_to_torch(buffer):
    return buffer.torch if hasattr(buffer, "torch") else wp.to_torch(buffer)


def _dirty_camera_reset_state(camera, frame_value: int = 17) -> None:
    camera._frame.fill_(frame_value)
    camera._data.pos_w.fill_(-123.0)
    camera._data.quat_w_world.fill_(0.25)
    wp.to_torch(camera._is_outdated).zero_()
    wp.to_torch(camera._timestamp).fill_(9.0)
    wp.to_torch(camera._timestamp_last_update).fill_(4.0)
    wp.to_torch(camera._render_data.camera_transforms).zero_()


def _snapshot_camera_reset_state(camera) -> dict[str, torch.Tensor]:
    return {
        "frame": camera._frame.detach().clone(),
        "pos_w": camera._data.pos_w.detach().clone(),
        "quat_w_world": camera._data.quat_w_world.detach().clone(),
        "is_outdated": wp.to_torch(camera._is_outdated).detach().clone(),
        "timestamp": wp.to_torch(camera._timestamp).detach().clone(),
        "timestamp_last_update": wp.to_torch(camera._timestamp_last_update).detach().clone(),
        "camera_transforms": wp.to_torch(camera._render_data.camera_transforms).detach().clone(),
    }


def _assert_camera_reset_states_close(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name, actual_tensor in actual.items():
        torch.testing.assert_close(actual_tensor, expected[name])


def _snapshot_camera_rgb(camera) -> torch.Tensor:
    return camera.data.output["rgb"].detach().clone()


def _make_launcher_args() -> argparse.Namespace:
    return argparse.Namespace(
        distributed=False,
        device="cuda:0",
        enable_cameras=False,
        headless=True,
        visualizer=None,
        max_visible_envs=None,
    )


def _make_newton_cfg(task: str, num_envs: int = 8, presets: str = "newton"):
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], f"presets={presets}"]
        cfg, _ = resolve_task_config(task, None)
    finally:
        sys.argv = old_argv

    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.seed = 13
    cfg.reset_cuda_graph = "force"
    cfg.episode_length_s = 0.04
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0
    return cfg


def _make_default_cfg(task: str, num_envs: int = 2):
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        cfg, _ = resolve_task_config(task, None)
    finally:
        sys.argv = old_argv

    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.seed = 13
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 0.04
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0
    return cfg


def _assert_graph_reset_to_default_state(gym_env, obs, reward, time_outs, extras, expected_success_rate: float):
    env = gym_env.unwrapped

    assert env._reset_cuda_graph_enabled
    assert env._reset_cuda_graph is not None
    assert int(env._reset_count_torch.item()) == env.num_envs
    assert time_outs.all()
    assert extras["log"]["Metrics/success_rate"] == pytest.approx(expected_success_rate)

    torch.testing.assert_close(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))
    torch.testing.assert_close(env.successes, torch.zeros_like(env.successes))
    assert not env.reset_goal_buf.any()

    default_joint_pos = env.hand.data.default_joint_pos.torch
    default_joint_vel = env.hand.data.default_joint_vel.torch
    torch.testing.assert_close(env.hand.data.joint_pos.torch, default_joint_pos)
    torch.testing.assert_close(env.hand.data.joint_vel.torch, default_joint_vel)
    torch.testing.assert_close(env.prev_targets, default_joint_pos)
    torch.testing.assert_close(env.cur_targets, default_joint_pos)
    torch.testing.assert_close(env.hand_dof_targets, default_joint_pos)

    object_pose = env.object.data.root_link_pose_w.torch
    expected_object_pos = env.object.data.default_root_pose.torch[:, 0:3] + env.scene.env_origins
    torch.testing.assert_close(object_pose[:, 0:3], expected_object_pos)
    torch.testing.assert_close(
        env.object.data.root_com_vel_w.torch,
        torch.zeros_like(env.object.data.root_com_vel_w.torch),
    )

    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["policy"]).all()
    torch.testing.assert_close(torch.linalg.norm(env.goal_rot, dim=-1), torch.ones(env.num_envs, device=env.device))
    torch.testing.assert_close(torch.linalg.norm(env.object_rot, dim=-1), torch.ones(env.num_envs, device=env.device))


def _assert_intermediates_match_canonical_recompute(env):
    graph_outputs = {
        "fingertip_pos": env.fingertip_pos.clone(),
        "fingertip_rot": env.fingertip_rot.clone(),
        "fingertip_velocities": env.fingertip_velocities.clone(),
        "object_pos": env.object_pos.clone(),
        "object_rot": env.object_rot.clone(),
        "object_velocities": env.object_velocities.clone(),
        "object_linvel": env.object_linvel.clone(),
        "object_angvel": env.object_angvel.clone(),
    }

    env._compute_intermediate_values()

    for name, value in graph_outputs.items():
        torch.testing.assert_close(value, getattr(env, name), rtol=1e-5, atol=1e-5)


def _zero_actions(env):
    return torch.zeros((env.num_envs, env.single_action_space.shape[0]), device=env.device)


def _rotation_distance_reference_torch(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.linalg.norm(quat_diff[:, 0:3], ord=2, dim=-1), max=1.0))


def _compute_rewards_reference_values(env):
    goal_dist = torch.linalg.norm(env.object_pos - env.in_hand_pos, ord=2, dim=-1)
    rot_dist = _rotation_distance_reference_torch(env.object_rot, env.goal_rot)

    reward = goal_dist * env.cfg.dist_reward_scale
    reward += 1.0 / (torch.abs(rot_dist) + env.cfg.rot_eps) * env.cfg.rot_reward_scale
    reward += torch.sum(env.actions**2, dim=-1) * env.cfg.action_penalty_scale

    goal_resets = torch.where(
        torch.abs(rot_dist) <= env.cfg.success_tolerance,
        torch.ones_like(env.reset_goal_buf),
        env.reset_goal_buf,
    )
    successes = env.successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + env.cfg.reach_goal_bonus, reward)
    reward = torch.where(goal_dist >= env.cfg.fall_dist, reward + env.cfg.fall_penalty, reward)

    resets = torch.where(goal_dist >= env.cfg.fall_dist, torch.ones_like(env.reset_buf), env.reset_buf)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    consecutive_successes = torch.where(
        num_resets > 0,
        env.cfg.av_factor * finished_cons_successes / num_resets
        + (1.0 - env.cfg.av_factor) * env.consecutive_successes,
        env.consecutive_successes,
    )

    return reward, goal_resets, successes, consecutive_successes


def _get_dones_reference_torch(env):
    env._compute_intermediate_values()
    goal_dist = torch.linalg.norm(env.object_pos - env.in_hand_pos, ord=2, dim=-1)
    out_of_reach = goal_dist >= env.cfg.fall_dist

    if env.cfg.max_consecutive_success > 0:
        rot_dist = _rotation_distance_reference_torch(env.object_rot, env.goal_rot)
        env.episode_length_buf.masked_fill_(torch.abs(rot_dist) <= env.cfg.success_tolerance, 0)
        max_success_reached = env.successes >= env.cfg.max_consecutive_success

    time_out = env.episode_length_buf >= env.max_episode_length - 1
    if env.cfg.max_consecutive_success > 0:
        time_out = time_out | max_success_reached
    return out_of_reach, time_out


def _get_rewards_reference_torch(env):
    total_reward, reset_goal_buf, successes, consecutive_successes = _compute_rewards_reference_values(env)
    env.reset_goal_buf[:] = reset_goal_buf
    env.successes[:] = successes
    env.consecutive_successes[:] = consecutive_successes

    goal_env_ids = env.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(goal_env_ids) > 0:
        env.reset_goal_buf[goal_env_ids] = False
    return total_reward


def _step_until_all_envs_timeout(gym_env, max_steps: int = 8):
    env = gym_env.unwrapped
    actions = _zero_actions(env)
    for _ in range(max_steps):
        obs, reward, _, time_outs, extras = gym_env.step(actions)
        if bool(time_outs.all().item()):
            return obs, reward, time_outs, extras
    pytest.fail(f"Expected all environments to time out within {max_steps} steps.")


def test_inhand_warp_core_kernels_run_on_cpu():
    from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
        _clear_reset_stats,
        _compute_inhand_intermediate_and_dones,
        _compute_inhand_rewards_and_reset_goals,
        _finalize_inhand_rewards,
        _initialize_reset_rng,
        _prepare_inhand_reset,
        _snapshot_inhand_reset_success,
    )

    device = "cpu"
    num_envs = 3
    num_dofs = 2
    num_bodies = 3
    num_fingertips = 2
    num_actions = 2

    env_mask = torch.tensor([True, False, True], device=device)
    env_mask_wp = wp.from_torch(env_mask, dtype=wp.bool)
    env_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    env_origins_wp = wp.from_torch(env_origins, dtype=wp.vec3f)
    default_object_pose_wp = wp.array(
        [
            wp.transform((0.1, 0.2, 0.3), wp.quat_identity()),
            wp.transform((1.0, 1.0, 1.0), wp.quat_identity()),
            wp.transform((2.0, 2.0, 2.0), wp.quat_identity()),
        ],
        dtype=wp.transformf,
        device=device,
    )
    default_joint_pos = torch.tensor([[0.1, 0.2], [1.0, 1.1], [2.0, 2.1]], dtype=torch.float32, device=device)
    default_joint_vel = torch.tensor([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=torch.float32, device=device)
    lower_limits = default_joint_pos - 1.0
    upper_limits = default_joint_pos + 1.0
    successes = torch.tensor([1.0, 7.0, 2.0], dtype=torch.float32, device=device)
    last_episode_success = torch.tensor([False, True, False], dtype=torch.bool, device=device)
    reset_goal_buf = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    episode_length_buf = torch.tensor([5, 6, 7], dtype=torch.int64, device=device)
    goal_rot_wp = wp.zeros(num_envs, dtype=wp.quatf, device=device)
    object_pose_out_wp = wp.zeros(num_envs, dtype=wp.transformf, device=device)
    object_velocity_out_wp = wp.zeros(num_envs, dtype=wp.spatial_vectorf, device=device)
    joint_pos_out = torch.full((num_envs, num_dofs), -10.0, dtype=torch.float32, device=device)
    joint_vel_out = torch.full((num_envs, num_dofs), -10.0, dtype=torch.float32, device=device)
    prev_targets = torch.zeros_like(joint_pos_out)
    cur_targets = torch.zeros_like(joint_pos_out)
    hand_dof_targets = torch.zeros_like(joint_pos_out)
    reset_count_wp = wp.zeros(1, dtype=wp.int32, device=device)
    reset_success_count_wp = wp.zeros(1, dtype=wp.int32, device=device)
    rng_state_wp = wp.zeros(num_envs, dtype=wp.uint32, device=device)

    wp.launch(_initialize_reset_rng, dim=num_envs, inputs=[17, rng_state_wp], device=device)
    wp.launch(_clear_reset_stats, dim=1, inputs=[reset_count_wp, reset_success_count_wp], device=device)
    wp.launch(
        _snapshot_inhand_reset_success,
        dim=num_envs,
        inputs=[
            env_mask_wp,
            1,
            wp.from_torch(successes, dtype=wp.float32),
            wp.from_torch(last_episode_success, dtype=wp.bool),
            reset_count_wp,
            reset_success_count_wp,
        ],
        device=device,
    )
    wp.launch(
        _prepare_inhand_reset,
        dim=num_envs,
        inputs=[
            env_mask_wp,
            default_object_pose_wp,
            env_origins_wp,
            0.0,
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.from_torch(default_joint_pos, dtype=wp.float32),
            wp.from_torch(default_joint_vel, dtype=wp.float32),
            wp.from_torch(lower_limits, dtype=wp.float32),
            wp.from_torch(upper_limits, dtype=wp.float32),
            0.0,
            0.0,
            num_dofs,
            rng_state_wp,
            wp.from_torch(successes, dtype=wp.float32),
            wp.from_torch(reset_goal_buf, dtype=wp.bool),
            wp.from_torch(episode_length_buf, dtype=wp.int64),
            goal_rot_wp,
            object_pose_out_wp,
            object_velocity_out_wp,
            wp.from_torch(joint_pos_out, dtype=wp.float32),
            wp.from_torch(joint_vel_out, dtype=wp.float32),
            wp.from_torch(prev_targets, dtype=wp.float32),
            wp.from_torch(cur_targets, dtype=wp.float32),
            wp.from_torch(hand_dof_targets, dtype=wp.float32),
        ],
        device=device,
    )

    assert int(wp.to_torch(reset_count_wp)[0].item()) == 2
    assert int(wp.to_torch(reset_success_count_wp)[0].item()) == 2
    torch.testing.assert_close(last_episode_success, torch.tensor([True, True, True], dtype=torch.bool, device=device))
    torch.testing.assert_close(reset_goal_buf, torch.tensor([False, True, False], dtype=torch.bool, device=device))
    torch.testing.assert_close(episode_length_buf, torch.tensor([0, 6, 0], dtype=torch.int64, device=device))
    torch.testing.assert_close(successes, torch.tensor([0.0, 7.0, 0.0], dtype=torch.float32, device=device))
    torch.testing.assert_close(joint_pos_out[env_mask], default_joint_pos[env_mask])
    torch.testing.assert_close(joint_vel_out[env_mask], default_joint_vel[env_mask])
    torch.testing.assert_close(prev_targets[env_mask], default_joint_pos[env_mask])
    torch.testing.assert_close(cur_targets[env_mask], default_joint_pos[env_mask])
    torch.testing.assert_close(hand_dof_targets[env_mask], default_joint_pos[env_mask])

    hand_body_pose_w_wp = wp.array(
        [
            [
                wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((0.2, 0.0, 0.0), wp.quat_identity()),
            ],
            [
                wp.transform((10.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((10.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((10.2, 0.0, 0.0), wp.quat_identity()),
            ],
            [
                wp.transform((20.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((20.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform((20.2, 0.0, 0.0), wp.quat_identity()),
            ],
        ],
        dtype=wp.transformf,
        device=device,
    )
    hand_body_vel_w_wp = wp.zeros((num_envs, num_bodies), dtype=wp.spatial_vectorf, device=device)
    finger_bodies_wp = wp.array([0, 2], dtype=wp.int32, device=device)
    object_root_pose_w_wp = wp.array(
        [
            wp.transform((0.1, 0.0, 0.0), wp.quat_identity()),
            wp.transform((12.0, 0.0, 0.0), wp.quat_identity()),
            wp.transform((20.1, 0.0, 0.0), wp.quat_identity()),
        ],
        dtype=wp.transformf,
        device=device,
    )
    object_root_vel_w_wp = wp.zeros(num_envs, dtype=wp.spatial_vectorf, device=device)
    in_hand_pos_wp = wp.array(
        [wp.vec3f(0.1, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.1, 0.0, 0.0)],
        dtype=wp.vec3f,
        device=device,
    )
    goal_rot_wp = wp.array(
        [wp.quat_identity(), wp.quat_identity(), wp.quat_identity()],
        dtype=wp.quatf,
        device=device,
    )
    fingertip_pos_wp = wp.zeros((num_envs, num_fingertips), dtype=wp.vec3f, device=device)
    fingertip_rot_wp = wp.zeros((num_envs, num_fingertips), dtype=wp.quatf, device=device)
    fingertip_velocities_wp = wp.zeros((num_envs, num_fingertips), dtype=wp.spatial_vectorf, device=device)
    object_pos_wp = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    object_rot_wp = wp.zeros(num_envs, dtype=wp.quatf, device=device)
    object_velocities_wp = wp.zeros(num_envs, dtype=wp.spatial_vectorf, device=device)
    object_linvel_wp = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    object_angvel_wp = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
    reset_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    reset_time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)

    wp.launch(
        _compute_inhand_intermediate_and_dones,
        dim=num_envs,
        inputs=[
            hand_body_pose_w_wp,
            hand_body_vel_w_wp,
            finger_bodies_wp,
            env_origins_wp,
            object_root_pose_w_wp,
            object_root_vel_w_wp,
            num_fingertips,
            in_hand_pos_wp,
            goal_rot_wp,
            wp.from_torch(successes, dtype=wp.float32),
            wp.from_torch(episode_length_buf, dtype=wp.int64),
            4,
            1.0,
            2,
            0.01,
            fingertip_pos_wp,
            fingertip_rot_wp,
            fingertip_velocities_wp,
            object_pos_wp,
            object_rot_wp,
            object_velocities_wp,
            object_linvel_wp,
            object_angvel_wp,
            wp.from_torch(reset_terminated, dtype=wp.bool),
            wp.from_torch(reset_time_outs, dtype=wp.bool),
        ],
        device=device,
    )

    torch.testing.assert_close(reset_terminated, torch.tensor([False, True, False], device=device))
    torch.testing.assert_close(reset_time_outs, torch.tensor([False, True, False], device=device))
    torch.testing.assert_close(wp.to_torch(fingertip_pos_wp)[0], torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]))

    reset_goal_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    reset_count_terms_wp = wp.zeros(num_envs, dtype=wp.float32, device=device)
    finished_success_terms_wp = wp.zeros(num_envs, dtype=wp.float32, device=device)
    goal_reset_terms_wp = wp.zeros(num_envs, dtype=wp.int32, device=device)
    consecutive_successes = torch.zeros(1, dtype=torch.float32, device=device)
    goal_reset_count_wp = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _compute_inhand_rewards_and_reset_goals,
        dim=num_envs,
        inputs=[
            env_mask_wp,
            wp.from_torch(reset_goal_buf, dtype=wp.bool),
            wp.from_torch(successes, dtype=wp.float32),
            object_pos_wp,
            object_rot_wp,
            in_hand_pos_wp,
            goal_rot_wp,
            wp.from_torch(torch.zeros((num_envs, num_actions), dtype=torch.float32, device=device), dtype=wp.float32),
            num_actions,
            -10.0,
            1.0,
            0.1,
            -0.01,
            0.01,
            2.0,
            1.0,
            -5.0,
            rng_state_wp,
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.from_torch(reward, dtype=wp.float32),
            reset_count_terms_wp,
            finished_success_terms_wp,
            goal_reset_terms_wp,
        ],
        device=device,
    )
    wp.launch(
        _finalize_inhand_rewards,
        dim=1,
        inputs=[
            reset_count_terms_wp,
            finished_success_terms_wp,
            num_envs,
            0.5,
            goal_reset_terms_wp,
            wp.from_torch(consecutive_successes, dtype=wp.float32),
            goal_reset_count_wp,
        ],
        device=device,
    )

    assert torch.isfinite(reward).all()
    assert int(wp.to_torch(goal_reset_count_wp)[0].item()) >= 1
    assert not reset_goal_buf.any()


@pytest.mark.parametrize("task", [_ALLEGRO_TASK, _SHADOW_TASK])
def test_default_physx_inhand_task_constructs_resets_and_steps(task: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the public in-hand task smoke test.")

    cfg = _make_default_cfg(task, num_envs=2)
    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    if needs_kit and not has_kit():
        pytest.skip("Default PhysX in-hand tasks require an active Kit app for this smoke test.")

    with launch_simulation(cfg, launcher_args):
        gym_env = gym.make(task, cfg=cfg)
        env = gym_env.unwrapped

        try:
            assert not env._inhand_warp_step_enabled
            assert not env._inhand_fused_reset_enabled
            assert not env._reset_cuda_graph_enabled

            obs, extras = gym_env.reset()
            obs, reward, terminated, time_outs, extras = gym_env.step(_zero_actions(env))

            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(reward).all()
            assert terminated.shape == (env.num_envs,)
            assert time_outs.shape == (env.num_envs,)
        finally:
            gym_env.close()


def test_inhand_reset_seed_reseeds_reset_and_goal_warp_rngs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.03
    cfg.reset_dof_pos_noise = 0.25
    cfg.reset_dof_vel_noise = 0.1

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    def sample_reset_and_goal(gym_env, seed: int):
        env = gym_env.unwrapped
        gym_env.reset(seed=seed)
        torch.cuda.synchronize()

        reset_state = {
            "goal_rot_after_reset": env.goal_rot.clone(),
            "object_pose_after_reset": env.object.data.root_link_pose_w.torch.clone(),
            "joint_pos_after_reset": env.hand.data.joint_pos.torch.clone(),
            "reset_rng_state": wp.to_torch(env._reset_rng_state_wp).clone(),
            "goal_rng_state_before_goal_reset": wp.to_torch(env._goal_reset_rng_state_wp).clone(),
        }

        env._graph_object_pos_torch[:] = env.in_hand_pos
        env._graph_object_rot_torch[:] = env.goal_rot
        env.reset_goal_buf[:] = True
        env.reset_buf.zero_()
        env.actions = _zero_actions(env)
        env._get_rewards()
        torch.cuda.synchronize()

        reset_state["goal_rot_after_reward_goal_reset"] = env.goal_rot.clone()
        reset_state["goal_rng_state_after_goal_reset"] = wp.to_torch(env._goal_reset_rng_state_wp).clone()
        return reset_state

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)

        try:
            first = sample_reset_and_goal(gym_env, seed=123)
            second = sample_reset_and_goal(gym_env, seed=123)
            different = sample_reset_and_goal(gym_env, seed=124)

            for key, value in first.items():
                torch.testing.assert_close(value, second[key], rtol=1e-6, atol=1e-6)

            assert not torch.equal(first["reset_rng_state"], different["reset_rng_state"])
            assert not torch.equal(
                first["goal_rng_state_before_goal_reset"], different["goal_rng_state_before_goal_reset"]
            )
            assert not torch.allclose(
                first["goal_rot_after_reward_goal_reset"], different["goal_rot_after_reward_goal_reset"]
            )
        finally:
            gym_env.close()


@pytest.mark.parametrize("task", [_ALLEGRO_TASK, _SHADOW_TASK])
def test_reset_cuda_graph_resets_timeout_envs_to_default_state(task: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(task)
    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(task, cfg=cfg)

        try:
            gym_env.reset()
            obs, reward, time_outs, extras = _step_until_all_envs_timeout(gym_env)

            _assert_graph_reset_to_default_state(gym_env, obs, reward, time_outs, extras, expected_success_rate=0.0)
        finally:
            gym_env.close()


def test_reset_cuda_graph_runs_scene_reset_for_vision_sensors(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(
        _SHADOW_VISION_BENCHMARK_TASK,
        num_envs=4,
        presets="newton,newton_renderer,rgb",
    )
    cfg.tiled_camera.width = 32
    cfg.tiled_camera.height = 32

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_SHADOW_VISION_BENCHMARK_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            camera = env.scene.sensors["tiled_camera"]
            assert camera._supports_graphable_camera_reset()

            def _unexpected_camera_reset(env_ids=None, env_mask=None):
                raise AssertionError("Newton camera graphable reset should not call full Camera.reset().")

            monkeypatch.setattr(camera, "reset", _unexpected_camera_reset)

            # Run through two timeout resets. The first reset captures and replays the graph, while the second reset
            # proves the camera residual hook is driven by the explicit replay contract instead of capture-time Python
            # state mutated by Camera.reset_graphable().
            obs, reward, time_outs, extras = _step_until_all_envs_timeout(gym_env)
            obs, reward, time_outs, extras = _step_until_all_envs_timeout(gym_env)

            assert "tiled_camera" in env.scene.sensors
            assert "critic" in obs
            _assert_graph_reset_to_default_state(gym_env, obs, reward, time_outs, extras, expected_success_rate=0.0)
            assert torch.isfinite(obs["critic"]).all()
        finally:
            gym_env.close()


def test_newton_camera_reset_graphable_matches_full_reset():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Newton camera graphable reset path.")

    cfg = _make_newton_cfg(
        _SHADOW_VISION_BENCHMARK_TASK,
        num_envs=4,
        presets="newton,newton_renderer,rgb",
    )
    cfg.reset_cuda_graph = "off"
    cfg.tiled_camera.width = 32
    cfg.tiled_camera.height = 32

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_SHADOW_VISION_BENCHMARK_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            camera = env.scene.sensors["tiled_camera"]
            assert camera._supports_graphable_camera_reset()

            reset_env_ids = torch.tensor([1, 3], dtype=torch.int32, device=env.device)
            reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            reset_mask[reset_env_ids.to(dtype=torch.long)] = True
            reset_mask_wp = wp.from_torch(reset_mask, dtype=wp.bool)

            _dirty_camera_reset_state(camera)
            camera.reset(env_ids=reset_env_ids)
            torch.cuda.synchronize()
            reference = _snapshot_camera_reset_state(camera)

            _dirty_camera_reset_state(camera)
            camera.reset_graphable(env_ids=None, env_mask=reset_mask_wp)
            camera.reset_after_graph(env_ids=reset_env_ids, env_mask=None, graphable_reset_applied=True)
            torch.cuda.synchronize()
            graphable = _snapshot_camera_reset_state(camera)

            _assert_camera_reset_states_close(graphable, reference)
        finally:
            gym_env.close()


def test_newton_camera_reset_cuda_graph_replay_matches_full_reset(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Newton camera graphable reset path.")

    cfg = _make_newton_cfg(
        _SHADOW_VISION_BENCHMARK_TASK,
        num_envs=4,
        presets="newton,newton_renderer,rgb",
    )
    cfg.reset_cuda_graph = "off"
    cfg.tiled_camera.width = 32
    cfg.tiled_camera.height = 32

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_SHADOW_VISION_BENCHMARK_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            camera = env.scene.sensors["tiled_camera"]
            camera_reset_supported = camera._supports_graphable_camera_reset()

            reset_env_ids = torch.tensor([1, 3], dtype=torch.int32, device=env.device)
            reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            reset_mask[reset_env_ids.to(dtype=torch.long)] = True
            reset_mask_wp = wp.from_torch(reset_mask, dtype=wp.bool)

            # Force renderer ray allocation before capture; this mirrors the production guard for camera graphability.
            _ = _snapshot_camera_rgb(camera)

            _dirty_camera_reset_state(camera)
            camera.reset(env_ids=reset_env_ids)
            torch.cuda.synchronize()
            reference_state = _snapshot_camera_reset_state(camera)
            reference_rgb = _snapshot_camera_rgb(camera)

            _dirty_camera_reset_state(camera)
            graph = capture_cuda_graph_relaxed(
                env.device,
                lambda: camera.reset_graphable(env_ids=None, env_mask=reset_mask_wp),
            )

            camera_reset_calls = []
            original_camera_reset = camera.reset

            def _record_camera_reset(env_ids=None, env_mask=None):
                camera_reset_calls.append((env_ids, env_mask))
                return original_camera_reset(env_ids=env_ids, env_mask=env_mask)

            monkeypatch.setattr(camera, "reset", _record_camera_reset)

            _dirty_camera_reset_state(camera)
            replay_stream = launch_cuda_graph_on_current_torch_stream(env.device, graph)
            with wp.ScopedStream(replay_stream, sync_enter=False):
                camera.reset_after_graph(env_ids=reset_env_ids, env_mask=None, graphable_reset_applied=True)
            torch.cuda.synchronize()

            graph_state = _snapshot_camera_reset_state(camera)
            graph_rgb = _snapshot_camera_rgb(camera)

            _assert_camera_reset_states_close(graph_state, reference_state)
            torch.testing.assert_close(graph_rgb, reference_rgb)
            if camera_reset_supported:
                assert camera_reset_calls == []
            else:
                assert len(camera_reset_calls) == 1
                assert camera_reset_calls[0][1] is None
        finally:
            gym_env.close()


def test_reset_cuda_graph_tracks_newton_camera_graph_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Newton camera graphable reset path.")

    cfg = _make_newton_cfg(
        _SHADOW_VISION_BENCHMARK_TASK,
        num_envs=4,
        presets="newton,newton_renderer,rgb",
    )
    cfg.tiled_camera.width = 32
    cfg.tiled_camera.height = 32

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_SHADOW_VISION_BENCHMARK_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            camera = env.scene.sensors["tiled_camera"]
            assert camera._supports_graphable_camera_reset()
            tensors = env._reset_cuda_graph_tensors()
            expected = {
                "scene.sensor.tiled_camera.is_outdated",
                "scene.sensor.tiled_camera.timestamp",
                "scene.sensor.tiled_camera.timestamp_last_update",
                "scene.sensor.tiled_camera.frame",
                "scene.sensor.tiled_camera.pos_w",
                "scene.sensor.tiled_camera.quat_w_world",
                "scene.sensor.tiled_camera.view.body_q",
                "scene.sensor.tiled_camera.view.site_body",
                "scene.sensor.tiled_camera.view.site_local",
                "scene.sensor.tiled_camera.view.pos_buf",
                "scene.sensor.tiled_camera.view.quat_buf",
                "scene.sensor.tiled_camera.renderer.camera_transforms",
            }
            assert expected <= set(tensors)
        finally:
            gym_env.close()


def test_reset_cuda_graph_updates_goal_markers_when_visualization_can_observe(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            calls = []

            def _record_goal_visualize(goal_pos, goal_rot):
                calls.append((goal_pos.detach().clone(), goal_rot.detach().clone()))

            monkeypatch.setattr(env, "_should_sync_goal_markers", lambda: True)
            monkeypatch.setattr(env.goal_markers, "visualize", _record_goal_visualize)

            reset_env_ids = torch.tensor([1, 3], dtype=torch.int32, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids.to(dtype=torch.long)] = True

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            assert len(calls) == 1
            torch.testing.assert_close(calls[0][0], env.goal_pos + env.scene.env_origins)
            torch.testing.assert_close(calls[0][1], env.goal_rot)
        finally:
            gym_env.close()


def test_inhand_torch_reward_path_skips_goal_marker_when_unobserved(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand Torch reward path smoke test.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.success_tolerance = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._compute_intermediate_values()
            env._inhand_warp_step_enabled = False
            env.actions.zero_()
            env.reset_buf.zero_()
            env.reset_goal_buf.zero_()
            env.successes.zero_()

            marker_calls = []

            def _record_visualize(translations=None, orientations=None, scales=None, marker_indices=None):
                marker_calls.append((translations, orientations))

            monkeypatch.setattr(env, "_should_sync_goal_markers", lambda: False)
            monkeypatch.setattr(env.goal_markers, "visualize", _record_visualize)

            reward = env._get_rewards()
            torch.cuda.synchronize()

            assert torch.isfinite(reward).all()
            assert not env.reset_goal_buf.any()
            assert marker_calls == []
            torch.testing.assert_close(
                torch.linalg.norm(env.goal_rot, dim=-1), torch.ones(env.num_envs, device=env.device)
            )
        finally:
            gym_env.close()


def test_reset_cuda_graph_supports_max_success_and_noop_noise_reset():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK)
    cfg.episode_length_s = 10.0
    cfg.max_consecutive_success = 1
    cfg.action_noise_model = NoiseModelCfg(noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"))
    cfg.observation_noise_model = NoiseModelCfg(noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"))

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.successes[:] = 1.0
            actions = _zero_actions(env)

            obs, reward, _, time_outs, extras = gym_env.step(actions)

            _assert_graph_reset_to_default_state(gym_env, obs, reward, time_outs, extras, expected_success_rate=1.0)
        finally:
            gym_env.close()


def test_reset_cuda_graph_auto_disables_reset_events():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=2)
    cfg.reset_cuda_graph = "auto"
    cfg.events = _RecordResetEventCfg()

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        try:
            env = gym_env.unwrapped
            assert not env._reset_cuda_graph_enabled
            assert "reset events are configured" in env._reset_cuda_graph_disable_reason

            gym_env.reset()
            env._test_reset_event_calls = 0
            reset_env_ids = torch.tensor([0], dtype=torch.int32, device=env.device)
            env._reset_idx(reset_env_ids)

            assert env._test_reset_event_calls == 1
            torch.testing.assert_close(env._test_reset_event_env_ids, reset_env_ids)
            torch.testing.assert_close(
                env.hand.data.joint_pos.torch[reset_env_ids.to(torch.long)],
                env.hand.data.default_joint_pos.torch[reset_env_ids.to(torch.long)],
            )
        finally:
            gym_env.close()


def test_reset_cuda_graph_allows_non_reset_events():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.events = _IntervalEventCfg()

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)

        try:
            assert gym_env.unwrapped._reset_cuda_graph_enabled
            gym_env.reset()
            obs, reward, time_outs, extras = _step_until_all_envs_timeout(gym_env)
            _assert_graph_reset_to_default_state(gym_env, obs, reward, time_outs, extras, expected_success_rate=0.0)
        finally:
            gym_env.close()


def test_reset_cuda_graph_partial_reset_preserves_unmasked_envs_and_intermediates():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=6)
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.02
    cfg.reset_dof_pos_noise = 0.2
    cfg.reset_dof_vel_noise = 0.3

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            action_values = torch.linspace(
                -0.75,
                0.75,
                env.num_envs * env.single_action_space.shape[0],
                dtype=torch.float32,
                device=env.device,
            )
            gym_env.step(action_values.reshape(env.num_envs, env.single_action_space.shape[0]))

            reset_env_ids = torch.tensor([0, 2, 5], dtype=torch.int32, device=env.device)
            keep_env_ids = torch.tensor([1, 3, 4], dtype=torch.long, device=env.device)
            reset_env_ids_long = reset_env_ids.to(dtype=torch.long)

            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids_long] = True
            env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 11
            env.successes[:] = torch.tensor([2.0, 4.0, 0.0, 5.0, 6.0, 1.0], device=env.device)
            env.reset_goal_buf[:] = True

            keep_state = {
                "episode_length_buf": env.episode_length_buf[keep_env_ids].clone(),
                "successes": env.successes[keep_env_ids].clone(),
                "reset_goal_buf": env.reset_goal_buf[keep_env_ids].clone(),
                "goal_rot": env.goal_rot[keep_env_ids].clone(),
                "joint_pos": env.hand.data.joint_pos.torch[keep_env_ids].clone(),
                "joint_vel": env.hand.data.joint_vel.torch[keep_env_ids].clone(),
                "prev_targets": env.prev_targets[keep_env_ids].clone(),
                "cur_targets": env.cur_targets[keep_env_ids].clone(),
                "hand_dof_targets": env.hand_dof_targets[keep_env_ids].clone(),
                "object_pose": env.object.data.root_link_pose_w.torch[keep_env_ids].clone(),
                "object_vel": env.object.data.root_com_vel_w.torch[keep_env_ids].clone(),
            }

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            assert int(env._reset_count_torch.item()) == len(reset_env_ids)
            assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(2.0 / 3.0)
            torch.testing.assert_close(
                env._last_episode_success[reset_env_ids_long],
                torch.tensor([True, False, True], device=env.device),
            )
            torch.testing.assert_close(env.episode_length_buf[reset_env_ids_long], torch.zeros_like(reset_env_ids_long))
            torch.testing.assert_close(
                env.successes[reset_env_ids_long], torch.zeros(len(reset_env_ids), device=env.device)
            )
            assert not env.reset_goal_buf[reset_env_ids_long].any()

            default_joint_pos = env.hand.data.default_joint_pos.torch[reset_env_ids_long]
            default_joint_vel = env.hand.data.default_joint_vel.torch[reset_env_ids_long]
            lower_limits = env.hand_dof_lower_limits[reset_env_ids_long]
            upper_limits = env.hand_dof_upper_limits[reset_env_ids_long]
            reset_joint_pos = env.hand.data.joint_pos.torch[reset_env_ids_long]
            reset_joint_vel = env.hand.data.joint_vel.torch[reset_env_ids_long]
            delta_max = upper_limits - default_joint_pos
            delta_min = lower_limits - default_joint_pos
            rand_delta_min = delta_min - 0.5 * (delta_max - delta_min)
            rand_delta_max = delta_min + 0.5 * (delta_max - delta_min)
            expected_joint_pos_low = default_joint_pos + cfg.reset_dof_pos_noise * rand_delta_min
            expected_joint_pos_high = default_joint_pos + cfg.reset_dof_pos_noise * rand_delta_max
            assert torch.all(reset_joint_pos >= torch.minimum(expected_joint_pos_low, expected_joint_pos_high) - 1e-6)
            assert torch.all(reset_joint_pos <= torch.maximum(expected_joint_pos_low, expected_joint_pos_high) + 1e-6)
            assert torch.all(torch.abs(reset_joint_vel - default_joint_vel) <= cfg.reset_dof_vel_noise + 1e-6)
            torch.testing.assert_close(env.prev_targets[reset_env_ids_long], reset_joint_pos)
            torch.testing.assert_close(env.cur_targets[reset_env_ids_long], reset_joint_pos)
            torch.testing.assert_close(env.hand_dof_targets[reset_env_ids_long], reset_joint_pos)

            expected_object_pos = (
                env.object.data.default_root_pose.torch[reset_env_ids_long, 0:3]
                + env.scene.env_origins[reset_env_ids_long]
            )
            reset_object_pose = env.object.data.root_link_pose_w.torch[reset_env_ids_long]
            assert torch.all(torch.abs(reset_object_pose[:, 0:3] - expected_object_pos) <= 0.02 + 1e-6)
            torch.testing.assert_close(
                env.object.data.root_com_vel_w.torch[reset_env_ids_long],
                torch.zeros_like(env.object.data.root_com_vel_w.torch[reset_env_ids_long]),
            )
            torch.testing.assert_close(
                torch.linalg.norm(reset_object_pose[:, 3:7], dim=-1),
                torch.ones(len(reset_env_ids), device=env.device),
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                torch.linalg.norm(env.goal_rot[reset_env_ids_long], dim=-1),
                torch.ones(len(reset_env_ids), device=env.device),
                rtol=1e-5,
                atol=1e-5,
            )

            torch.testing.assert_close(env.episode_length_buf[keep_env_ids], keep_state["episode_length_buf"])
            torch.testing.assert_close(env.successes[keep_env_ids], keep_state["successes"])
            torch.testing.assert_close(env.reset_goal_buf[keep_env_ids], keep_state["reset_goal_buf"])
            torch.testing.assert_close(env.goal_rot[keep_env_ids], keep_state["goal_rot"])
            torch.testing.assert_close(env.hand.data.joint_pos.torch[keep_env_ids], keep_state["joint_pos"])
            torch.testing.assert_close(env.hand.data.joint_vel.torch[keep_env_ids], keep_state["joint_vel"])
            torch.testing.assert_close(env.prev_targets[keep_env_ids], keep_state["prev_targets"])
            torch.testing.assert_close(env.cur_targets[keep_env_ids], keep_state["cur_targets"])
            torch.testing.assert_close(env.hand_dof_targets[keep_env_ids], keep_state["hand_dof_targets"])
            torch.testing.assert_close(env.object.data.root_link_pose_w.torch[keep_env_ids], keep_state["object_pose"])
            torch.testing.assert_close(env.object.data.root_com_vel_w.torch[keep_env_ids], keep_state["object_vel"])

            _assert_intermediates_match_canonical_recompute(env)

            graph_object = env._reset_cuda_graph
            second_reset_env_ids = torch.tensor([1, 4], dtype=torch.int32, device=env.device)
            second_reset_env_ids_long = second_reset_env_ids.to(dtype=torch.long)
            second_keep_env_ids = torch.tensor([0, 2, 3, 5], dtype=torch.long, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[second_reset_env_ids_long] = True
            env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 31
            env.successes[:] = torch.tensor([7.0, 0.0, 8.0, 9.0, 1.0, 10.0], device=env.device)
            env.reset_goal_buf[:] = True

            second_keep_state = {
                "episode_length_buf": env.episode_length_buf[second_keep_env_ids].clone(),
                "successes": env.successes[second_keep_env_ids].clone(),
                "reset_goal_buf": env.reset_goal_buf[second_keep_env_ids].clone(),
                "joint_pos": env.hand.data.joint_pos.torch[second_keep_env_ids].clone(),
                "joint_vel": env.hand.data.joint_vel.torch[second_keep_env_ids].clone(),
                "object_pose": env.object.data.root_link_pose_w.torch[second_keep_env_ids].clone(),
                "object_vel": env.object.data.root_com_vel_w.torch[second_keep_env_ids].clone(),
            }

            assert env._reset_idx_cuda_graph(second_reset_env_ids)
            torch.cuda.synchronize()

            assert env._reset_cuda_graph is graph_object
            assert int(env._reset_count_torch.item()) == len(second_reset_env_ids)
            assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(0.5)
            torch.testing.assert_close(
                env._last_episode_success[second_reset_env_ids_long],
                torch.tensor([False, True], device=env.device),
            )
            torch.testing.assert_close(
                env.episode_length_buf[second_reset_env_ids_long],
                torch.zeros_like(second_reset_env_ids_long),
            )
            torch.testing.assert_close(
                env.successes[second_reset_env_ids_long],
                torch.zeros(len(second_reset_env_ids), device=env.device),
            )
            assert not env.reset_goal_buf[second_reset_env_ids_long].any()
            torch.testing.assert_close(
                env.episode_length_buf[second_keep_env_ids],
                second_keep_state["episode_length_buf"],
            )
            torch.testing.assert_close(env.successes[second_keep_env_ids], second_keep_state["successes"])
            torch.testing.assert_close(env.reset_goal_buf[second_keep_env_ids], second_keep_state["reset_goal_buf"])
            torch.testing.assert_close(
                env.hand.data.joint_pos.torch[second_keep_env_ids], second_keep_state["joint_pos"]
            )
            torch.testing.assert_close(
                env.hand.data.joint_vel.torch[second_keep_env_ids], second_keep_state["joint_vel"]
            )
            torch.testing.assert_close(
                env.object.data.root_link_pose_w.torch[second_keep_env_ids],
                second_keep_state["object_pose"],
            )
            torch.testing.assert_close(
                env.object.data.root_com_vel_w.torch[second_keep_env_ids],
                second_keep_state["object_vel"],
            )
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


def test_reset_cuda_graph_task_apply_graph_preserves_lazy_acceleration_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._reset_idx_cuda_graph(torch.tensor([0], dtype=torch.int32, device=env.device))
            torch.cuda.synchronize()
            assert env._reset_cuda_graph is not None

            # Force the lazy acceleration properties into the hard case: stale Python timestamps before replay.
            env.hand.data._joint_acc.timestamp = -1.0
            env.object.data._body_com_acc_w.timestamp = -1.0

            reset_env_ids = torch.tensor([1, 3], dtype=torch.int32, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids.to(torch.long)] = True
            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            assert env.hand.data._joint_acc.timestamp == env.hand.data._sim_timestamp
            assert env.object.data._body_com_acc_w.timestamp == env.object.data._sim_timestamp
            torch.testing.assert_close(
                env.hand.data.joint_acc.torch[reset_env_ids.to(torch.long)],
                torch.zeros_like(env.hand.data.joint_acc.torch[reset_env_ids.to(torch.long)]),
            )
            torch.testing.assert_close(
                env.object.data.body_com_acc_w.torch[reset_env_ids.to(torch.long)],
                torch.zeros_like(env.object.data.body_com_acc_w.torch[reset_env_ids.to(torch.long)]),
            )
        finally:
            gym_env.close()


def test_reset_cuda_graph_explicit_env_ids_are_single_source_of_truth():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            stale_mask_env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
            explicit_env_ids = torch.tensor([2], dtype=torch.int32, device=env.device)
            explicit_env_ids_long = explicit_env_ids.to(dtype=torch.long)

            env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 10
            env.successes[:] = torch.arange(env.num_envs, dtype=torch.float32, device=env.device) + 1.0
            env.reset_goal_buf[:] = True

            stale_state = {
                "episode_length_buf": env.episode_length_buf[stale_mask_env_ids].clone(),
                "successes": env.successes[stale_mask_env_ids].clone(),
                "reset_goal_buf": env.reset_goal_buf[stale_mask_env_ids].clone(),
            }

            env.reset_buf.zero_()
            env.reset_buf[stale_mask_env_ids] = True

            assert env._reset_idx_cuda_graph(explicit_env_ids)
            torch.cuda.synchronize()

            assert int(env._reset_count_torch.item()) == 1
            torch.testing.assert_close(env.reset_buf, torch.tensor([False, False, True, False], device=env.device))
            torch.testing.assert_close(
                env.episode_length_buf[explicit_env_ids_long],
                torch.zeros_like(env.episode_length_buf[explicit_env_ids_long]),
            )
            torch.testing.assert_close(
                env.successes[explicit_env_ids_long],
                torch.zeros_like(env.successes[explicit_env_ids_long]),
            )
            assert not env.reset_goal_buf[explicit_env_ids_long].any()
            torch.testing.assert_close(env.episode_length_buf[stale_mask_env_ids], stale_state["episode_length_buf"])
            torch.testing.assert_close(env.successes[stale_mask_env_ids], stale_state["successes"])
            torch.testing.assert_close(env.reset_goal_buf[stale_mask_env_ids], stale_state["reset_goal_buf"])
        finally:
            gym_env.close()


def test_reset_cuda_graph_empty_mask_is_noop():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.02
    cfg.reset_dof_pos_noise = 0.2
    cfg.reset_dof_vel_noise = 0.3

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            gym_env.step(_zero_actions(env))
            before = {
                "episode_length_buf": env.episode_length_buf.clone(),
                "successes": env.successes.clone(),
                "reset_goal_buf": env.reset_goal_buf.clone(),
                "goal_rot": env.goal_rot.clone(),
                "joint_pos": env.hand.data.joint_pos.torch.clone(),
                "joint_vel": env.hand.data.joint_vel.torch.clone(),
                "prev_targets": env.prev_targets.clone(),
                "cur_targets": env.cur_targets.clone(),
                "hand_dof_targets": env.hand_dof_targets.clone(),
                "object_pose": env.object.data.root_link_pose_w.torch.clone(),
                "object_vel": env.object.data.root_com_vel_w.torch.clone(),
            }

            env.reset_buf.zero_()
            empty_env_ids = torch.empty(0, dtype=torch.int32, device=env.device)

            assert env._reset_idx_cuda_graph(empty_env_ids)
            torch.cuda.synchronize()

            assert int(env._reset_count_torch.item()) == 0
            for name, value in before.items():
                current = {
                    "episode_length_buf": env.episode_length_buf,
                    "successes": env.successes,
                    "reset_goal_buf": env.reset_goal_buf,
                    "goal_rot": env.goal_rot,
                    "joint_pos": env.hand.data.joint_pos.torch,
                    "joint_vel": env.hand.data.joint_vel.torch,
                    "prev_targets": env.prev_targets,
                    "cur_targets": env.cur_targets,
                    "hand_dof_targets": env.hand_dof_targets,
                    "object_pose": env.object.data.root_link_pose_w.torch,
                    "object_vel": env.object.data.root_com_vel_w.torch,
                }[name]
                torch.testing.assert_close(current, value)
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


def test_reset_cuda_graph_no_reset_step_entrypoint_is_noop_after_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.02
    cfg.reset_dof_pos_noise = 0.2
    cfg.reset_dof_vel_noise = 0.3

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            reset_env_ids = torch.tensor([0, 2], dtype=torch.int32, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids.to(dtype=torch.long)] = True
            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()
            graphs = (env._reset_common_cuda_graph, env._reset_cuda_graph)

            env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.int64, device=env.device) + 3
            env.successes[:] = torch.arange(env.num_envs, dtype=torch.float32, device=env.device) + 1.0
            env.reset_goal_buf[:] = torch.tensor([True, False, True, False], device=env.device)
            env.goal_rot[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            before = {
                "episode_length_buf": env.episode_length_buf.clone(),
                "successes": env.successes.clone(),
                "reset_goal_buf": env.reset_goal_buf.clone(),
                "goal_rot": env.goal_rot.clone(),
                "joint_pos": env.hand.data.joint_pos.torch.clone(),
                "joint_vel": env.hand.data.joint_vel.torch.clone(),
                "prev_targets": env.prev_targets.clone(),
                "cur_targets": env.cur_targets.clone(),
                "hand_dof_targets": env.hand_dof_targets.clone(),
                "object_pose": env.object.data.root_link_pose_w.torch.clone(),
                "object_vel": env.object.data.root_com_vel_w.torch.clone(),
            }

            env.reset_buf.zero_()
            materialized_env_ids = env._reset_idx_from_reset_buf()
            torch.cuda.synchronize()

            assert materialized_env_ids.numel() == 0
            assert env._reset_common_cuda_graph is graphs[0]
            assert env._reset_cuda_graph is graphs[1]
            for name, value in before.items():
                current = {
                    "episode_length_buf": env.episode_length_buf,
                    "successes": env.successes,
                    "reset_goal_buf": env.reset_goal_buf,
                    "goal_rot": env.goal_rot,
                    "joint_pos": env.hand.data.joint_pos.torch,
                    "joint_vel": env.hand.data.joint_vel.torch,
                    "prev_targets": env.prev_targets,
                    "cur_targets": env.cur_targets,
                    "hand_dof_targets": env.hand_dof_targets,
                    "object_pose": env.object.data.root_link_pose_w.torch,
                    "object_vel": env.object.data.root_com_vel_w.torch,
                }[name]
                torch.testing.assert_close(current, value)
        finally:
            gym_env.close()


def test_reset_cuda_graph_repeated_randomized_reset_advances_rng_for_same_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.02
    cfg.reset_dof_pos_noise = 0.2
    cfg.reset_dof_vel_noise = 0.3

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            reset_env_ids = torch.tensor([2], dtype=torch.long, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids] = True

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()
            first = torch.cat(
                (
                    env.goal_rot[reset_env_ids].flatten(),
                    env.hand.data.joint_pos.torch[reset_env_ids].flatten(),
                    env.hand.data.joint_vel.torch[reset_env_ids].flatten(),
                    env.object.data.root_link_pose_w.torch[reset_env_ids].flatten(),
                )
            )
            graph_object = env._reset_cuda_graph

            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids] = True
            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()
            second = torch.cat(
                (
                    env.goal_rot[reset_env_ids].flatten(),
                    env.hand.data.joint_pos.torch[reset_env_ids].flatten(),
                    env.hand.data.joint_vel.torch[reset_env_ids].flatten(),
                    env.object.data.root_link_pose_w.torch[reset_env_ids].flatten(),
                )
            )

            assert env._reset_cuda_graph is graph_object
            assert not torch.allclose(first, second)
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


@pytest.mark.parametrize("task", [_ALLEGRO_TASK, _SHADOW_TASK])
def test_reset_cuda_graph_replays_many_masks_without_cross_env_writes(task: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(task, num_envs=7)
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.015
    cfg.reset_dof_pos_noise = 0.1
    cfg.reset_dof_vel_noise = 0.2
    cfg.success_count_threshold = 3

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(task, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            graph_object = None
            patterns = ([0], [6], [1, 3, 5], list(range(env.num_envs)), [2, 4])
            for iteration, reset_ids_cpu in enumerate(patterns):
                reset_env_ids = torch.tensor(reset_ids_cpu, dtype=torch.int32, device=env.device)
                reset_env_ids_long = reset_env_ids.to(dtype=torch.long)
                keep_env_ids = torch.tensor(
                    [i for i in range(env.num_envs) if i not in reset_ids_cpu], dtype=torch.long, device=env.device
                )

                env.reset_buf.zero_()
                env.reset_buf[reset_env_ids_long] = True
                env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 100
                env.successes[:] = torch.arange(env.num_envs, dtype=torch.float32, device=env.device) + iteration
                env.reset_goal_buf[:] = True

                keep_state = {
                    "episode_length_buf": env.episode_length_buf[keep_env_ids].clone(),
                    "successes": env.successes[keep_env_ids].clone(),
                    "reset_goal_buf": env.reset_goal_buf[keep_env_ids].clone(),
                    "joint_pos": env.hand.data.joint_pos.torch[keep_env_ids].clone(),
                    "joint_vel": env.hand.data.joint_vel.torch[keep_env_ids].clone(),
                    "object_pose": env.object.data.root_link_pose_w.torch[keep_env_ids].clone(),
                    "object_vel": env.object.data.root_com_vel_w.torch[keep_env_ids].clone(),
                }

                side_stream = torch.cuda.Stream()
                with torch.cuda.stream(side_stream):
                    assert env._reset_idx_cuda_graph(reset_env_ids)
                side_stream.synchronize()

                if graph_object is None:
                    graph_object = env._reset_cuda_graph
                else:
                    assert env._reset_cuda_graph is graph_object

                expected_success = (
                    torch.arange(env.num_envs, device=env.device)[reset_env_ids_long] + iteration
                ) >= cfg.success_count_threshold
                torch.testing.assert_close(env._last_episode_success[reset_env_ids_long], expected_success)
                assert int(env._reset_count_torch.item()) == len(reset_ids_cpu)
                assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(
                    expected_success.float().mean().item()
                )
                torch.testing.assert_close(
                    env.episode_length_buf[reset_env_ids_long],
                    torch.zeros_like(reset_env_ids_long),
                )
                torch.testing.assert_close(
                    env.successes[reset_env_ids_long],
                    torch.zeros(len(reset_ids_cpu), device=env.device),
                )
                assert not env.reset_goal_buf[reset_env_ids_long].any()

                if len(keep_env_ids) > 0:
                    torch.testing.assert_close(env.episode_length_buf[keep_env_ids], keep_state["episode_length_buf"])
                    torch.testing.assert_close(env.successes[keep_env_ids], keep_state["successes"])
                    torch.testing.assert_close(env.reset_goal_buf[keep_env_ids], keep_state["reset_goal_buf"])
                    torch.testing.assert_close(env.hand.data.joint_pos.torch[keep_env_ids], keep_state["joint_pos"])
                    torch.testing.assert_close(env.hand.data.joint_vel.torch[keep_env_ids], keep_state["joint_vel"])
                    torch.testing.assert_close(
                        env.object.data.root_link_pose_w.torch[keep_env_ids],
                        keep_state["object_pose"],
                    )
                    torch.testing.assert_close(
                        env.object.data.root_com_vel_w.torch[keep_env_ids],
                        keep_state["object_vel"],
                    )

                _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


def test_reset_cuda_graph_base_reset_uses_exact_env_ids(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=5)
    cfg.episode_length_s = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            reset_env_ids = torch.tensor([1, 4], dtype=torch.int32, device=env.device)
            reset_env_ids_long = reset_env_ids.to(dtype=torch.long)
            env_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env_mask[reset_env_ids_long] = True

            actuator_reset_env_ids = []
            for actuator in env.hand.actuators.values():
                original_reset = actuator.reset

                def _record_actuator_reset(env_ids, original_reset=original_reset):
                    actuator_reset_env_ids.append(env_ids.detach().clone())
                    return original_reset(env_ids)

                monkeypatch.setattr(actuator, "reset", _record_actuator_reset)

            composer_snapshots = []
            for asset in (env.hand, env.object):
                body_ids, _ = asset.find_bodies(".*")
                forces = torch.arange(
                    asset.num_instances * len(body_ids) * 3, dtype=torch.float32, device=env.device
                ).reshape(asset.num_instances, len(body_ids), 3)
                torques = -forces - 1.0
                for composer in (asset.instantaneous_wrench_composer, asset.permanent_wrench_composer):
                    composer.set_forces_and_torques_index(forces=forces, torques=torques, body_ids=body_ids)
                    torch.cuda.synchronize()
                    composer_snapshots.append(
                        (
                            composer,
                            {
                                name: _warp_buffer_to_torch(getattr(composer, name)).clone()
                                for name in (
                                    "_global_force_w",
                                    "_global_torque_w",
                                    "_global_force_at_com_w",
                                    "_local_force_b",
                                    "_local_torque_b",
                                    "_out_force_b",
                                    "_out_torque_b",
                                )
                            },
                        )
                    )

            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids_long] = True

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            assert actuator_reset_env_ids
            for observed_env_ids in actuator_reset_env_ids:
                torch.testing.assert_close(observed_env_ids, reset_env_ids)

            for composer, snapshot in composer_snapshots:
                for name, before in snapshot.items():
                    current = _warp_buffer_to_torch(getattr(composer, name))
                    torch.testing.assert_close(current[env_mask], torch.zeros_like(current[env_mask]))
                    torch.testing.assert_close(current[~env_mask], before[~env_mask])
        finally:
            gym_env.close()


def test_reset_cuda_graph_accepts_noncontiguous_env_ids():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=6)
    cfg.episode_length_s = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env_ids_storage = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
            reset_env_ids = env_ids_storage[::2]
            assert not reset_env_ids.is_contiguous()
            keep_env_ids = env_ids_storage[1::2]

            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids] = True
            env.episode_length_buf[:] = torch.arange(env.num_envs, dtype=torch.long, device=env.device) + 5
            env.successes[:] = 3.0
            keep_episode_lengths = env.episode_length_buf[keep_env_ids].clone()
            keep_successes = env.successes[keep_env_ids].clone()

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            torch.testing.assert_close(env.episode_length_buf[reset_env_ids], torch.zeros_like(reset_env_ids))
            torch.testing.assert_close(
                env.successes[reset_env_ids], torch.zeros_like(reset_env_ids, dtype=torch.float32)
            )
            torch.testing.assert_close(env.episode_length_buf[keep_env_ids], keep_episode_lengths)
            torch.testing.assert_close(env.successes[keep_env_ids], keep_successes)
            torch.testing.assert_close(env._last_episode_success[reset_env_ids], torch.ones_like(reset_env_ids).bool())
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "reset_buf",
        "episode_length_buf",
        "successes",
        "last_episode_success",
        "reset_goal_buf",
        "goal_rot",
        "prev_targets",
        "cur_targets",
        "hand_dof_targets",
        "scene_wrench_buffer",
        "reset_rng_state",
        "reset_count",
        "finger_bodies",
        "hand_body_pose",
        "object_root_velocity",
        "hand_all_joint_mask",
        "graph_object_pos",
        "reset_position_noise",
    ],
)
def test_reset_cuda_graph_auto_recaptures_on_compatible_graph_assumption_change(mutation: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "auto"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.reset_buf[:] = True
            env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
            assert env._reset_idx_cuda_graph(env_ids)
            old_graph = env._reset_cuda_graph

            if mutation == "reset_buf":
                env.reset_buf = env.reset_buf.clone()
            elif mutation == "episode_length_buf":
                env.episode_length_buf = env.episode_length_buf.clone()
            elif mutation == "successes":
                env.successes = env.successes.clone()
            elif mutation == "last_episode_success":
                env._last_episode_success = env._last_episode_success.clone()
            elif mutation == "reset_goal_buf":
                env.reset_goal_buf = env.reset_goal_buf.clone()
            elif mutation == "goal_rot":
                env.goal_rot = env.goal_rot.clone()
            elif mutation == "prev_targets":
                env.prev_targets = env.prev_targets.clone()
            elif mutation == "cur_targets":
                env.cur_targets = env.cur_targets.clone()
            elif mutation == "hand_dof_targets":
                env.hand_dof_targets = env.hand_dof_targets.clone()
            elif mutation == "scene_wrench_buffer":
                composer = env.hand.instantaneous_wrench_composer
                composer._out_force_b = wp.clone(composer._out_force_b)
            elif mutation == "reset_rng_state":
                env._reset_rng_state_wp = wp.clone(env._reset_rng_state_wp)
            elif mutation == "reset_count":
                env._reset_count_wp = wp.clone(env._reset_count_wp)
            elif mutation == "finger_bodies":
                env._finger_bodies_wp = wp.clone(env._finger_bodies_wp)
            elif mutation == "hand_body_pose":
                env._hand_body_pose_w_wp = wp.clone(env._hand_body_pose_w_wp)
            elif mutation == "object_root_velocity":
                env._object_root_vel_w_wp = wp.clone(env._object_root_vel_w_wp)
            elif mutation == "hand_all_joint_mask":
                env.hand._ALL_JOINT_MASK = wp.clone(env.hand._ALL_JOINT_MASK)
            elif mutation == "graph_object_pos":
                env._graph_object_pos_wp = wp.clone(env._graph_object_pos_wp)
            elif mutation == "reset_position_noise":
                env.cfg.reset_position_noise += 0.01
            else:
                raise AssertionError(f"Unhandled mutation {mutation}")

            env.reset_buf[:] = True

            assert env._reset_idx_cuda_graph(env_ids)
            torch.cuda.synchronize()

            assert env._reset_cuda_graph_enabled
            assert env._reset_cuda_graph_disable_reason == ""
            assert env._reset_cuda_graph is not old_graph
            torch.testing.assert_close(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))
            torch.testing.assert_close(env.successes, torch.zeros_like(env.successes))
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("reset_buf_metadata", "reset_buf shape changed"),
        ("prev_targets_metadata", "prev_targets shape changed"),
    ],
)
def test_reset_cuda_graph_auto_disables_on_incompatible_graph_assumption_change(mutation: str, reason: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "auto"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            if mutation == "reset_buf_metadata":
                env.reset_buf = env.reset_buf.view(2, 2)
            elif mutation == "prev_targets_metadata":
                env.prev_targets = env.prev_targets.transpose(0, 1)
            else:
                raise AssertionError(f"Unhandled mutation {mutation}")

            env.reset_buf[:] = True
            env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)

            assert not env._reset_idx_cuda_graph(env_ids)
            assert not env._reset_cuda_graph_enabled
            assert reason in env._reset_cuda_graph_disable_reason
        finally:
            gym_env.close()


def test_reset_cuda_graph_auto_disables_when_recapture_fails(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "auto"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
            env.reset_buf[:] = True
            assert env._reset_idx_cuda_graph(env_ids)

            env.prev_targets = env.prev_targets.clone()

            def _fail_setup_reset_cuda_graph_buffers():
                raise RuntimeError("synthetic recapture failure")

            monkeypatch.setattr(env, "_setup_reset_cuda_graph_buffers", _fail_setup_reset_cuda_graph_buffers)
            env.reset_buf[:] = True

            assert not env._reset_idx_cuda_graph(env_ids)
            assert not env._reset_cuda_graph_enabled
            assert "recapture after prev_targets storage changed failed: synthetic recapture failure" in (
                env._reset_cuda_graph_disable_reason
            )
        finally:
            gym_env.close()


def test_reset_cuda_graph_force_raises_when_recapture_fails(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
            env.reset_buf[:] = True
            assert env._reset_idx_cuda_graph(env_ids)

            env.prev_targets = env.prev_targets.clone()

            def _fail_setup_reset_cuda_graph_buffers():
                raise RuntimeError("synthetic recapture failure")

            monkeypatch.setattr(env, "_setup_reset_cuda_graph_buffers", _fail_setup_reset_cuda_graph_buffers)
            env.reset_buf[:] = True

            with pytest.raises(
                RuntimeError,
                match="recapture after prev_targets storage changed failed: synthetic recapture failure",
            ):
                env._reset_idx_cuda_graph(env_ids)
            assert not env._reset_cuda_graph_enabled
        finally:
            gym_env.close()


def test_inhand_reset_graph_guard_targets_are_registry_backed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._setup_reset_cuda_graph_buffers()
            tensors = env._reset_cuda_graph_tensors()

            expected = {
                "prev_targets",
                "prev_targets_wp",
                "hand.default_joint_pos",
                "hand.default_joint_pos_wp",
                "hand.body_link_pose_w",
                "reset_mask_wp",
                "reset_rng_state",
                "reset_object_pose",
                "hand.joint_pos_target",
            }
            assert expected <= set(tensors)
        finally:
            gym_env.close()


def test_reset_cuda_graph_force_rejects_incompatible_graph_assumption_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.prev_targets = env.prev_targets.transpose(0, 1)
            env.reset_buf[:] = True
            env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)

            with pytest.raises(RuntimeError, match="prev_targets shape changed"):
                env._reset_idx_cuda_graph(env_ids)
        finally:
            gym_env.close()


def test_reset_cuda_graph_invalid_mode_is_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=2)
    cfg.reset_cuda_graph = "definitely-not-a-mode"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        with pytest.raises(ValueError, match="Unsupported reset_cuda_graph"):
            gym.make(_ALLEGRO_TASK, cfg=cfg)


def test_reset_cuda_graph_step_recaptures_reset_mask_storage_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "auto"
    cfg.episode_length_s = 0.04
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.reset_buf = env.reset_buf.clone()

            gym_env.step(_zero_actions(env))
            torch.cuda.synchronize()

            assert env._reset_cuda_graph_enabled
            assert env._reset_cuda_graph_disable_reason == ""
            assert env._reset_cuda_graph is not None
            assert env._inhand_fused_reset_enabled
        finally:
            gym_env.close()


def test_reset_cuda_graph_runs_stateful_noise_reset_outside_graph(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "auto"
    cfg.episode_length_s = 0.04
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0
    cfg.action_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"),
        bias_noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
    )

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()

            reset_calls = []
            original_noise_reset = env._action_noise_model.reset

            def _record_noise_reset(env_ids=None):
                if env_ids is not None:
                    reset_calls.append(torch.as_tensor(env_ids, device=env.device).clone())
                return original_noise_reset(env_ids)

            monkeypatch.setattr(env._action_noise_model, "reset", _record_noise_reset)

            assert env._reset_cuda_graph_enabled
            obs, reward, _, time_outs, extras = gym_env.step(_zero_actions(env))

            assert time_outs.all()
            assert env._reset_cuda_graph is not None
            assert len(reset_calls) == 1
            torch.testing.assert_close(reset_calls[0], torch.arange(env.num_envs, dtype=torch.int32, device=env.device))
            assert extras["log"]["Metrics/success_rate"] == pytest.approx(0.0)
            torch.testing.assert_close(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))
            torch.testing.assert_close(env.successes, torch.zeros_like(env.successes))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["policy"]).all()
        finally:
            gym_env.close()


def test_reset_cuda_graph_step_uses_mask_before_materializing_env_ids(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "force"
    cfg.episode_length_s = 0.04
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            calls = []
            scene_after_graph_calls = []
            original_try_cuda_graph_reset = env._try_reset_idx_cuda_graph
            original_materialize = env._materialize_reset_context_env_ids
            original_scene_reset_after_graph = env.scene.reset_after_graph

            def _record_try_cuda_graph_reset(env_ids=None):
                calls.append(("graph_reset", env_ids is None))
                return original_try_cuda_graph_reset(env_ids)

            def _record_materialize(ctx):
                calls.append(("materialize", ctx.env_ids is None))
                return original_materialize(ctx)

            def _record_scene_reset_after_graph(env_ids=None, env_mask=None, *, graphable_reset_applied=False):
                scene_after_graph_calls.append((env_ids, env_mask, graphable_reset_applied))
                return original_scene_reset_after_graph(
                    env_ids=env_ids, env_mask=env_mask, graphable_reset_applied=graphable_reset_applied
                )

            monkeypatch.setattr(env, "_try_reset_idx_cuda_graph", _record_try_cuda_graph_reset)
            monkeypatch.setattr(env, "_materialize_reset_context_env_ids", _record_materialize)
            monkeypatch.setattr(env.scene, "reset_after_graph", _record_scene_reset_after_graph)

            gym_env.step(_zero_actions(env))
            torch.cuda.synchronize()

            assert calls[0] == ("graph_reset", True)
            assert calls[1] == ("materialize", True)
            assert len(scene_after_graph_calls) == 1
            assert scene_after_graph_calls[0][1] is None
            assert scene_after_graph_calls[0][2] is True
            assert env._reset_cuda_graph_enabled
            torch.testing.assert_close(env.episode_length_buf, torch.zeros_like(env.episode_length_buf))
        finally:
            gym_env.close()


def test_reset_cuda_graph_orders_common_residual_before_task_apply_graph(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "force"
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0
    cfg.action_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"),
        bias_noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
    )

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()

            order = []
            scene_after_graph_calls = []
            original_launch_phase = env._launch_reset_cuda_graph_phase
            original_common_after = env._reset_idx_common_after_graph
            original_scene_reset_after_graph = env.scene.reset_after_graph
            original_apply_after = env._apply_inhand_reset_to_sim_after_graph
            original_noise_reset = env._action_noise_model.reset
            original_task_prepare = env._launch_inhand_task_reset_prepare
            original_apply_graphable = env._launch_inhand_reset_to_sim_graphable

            def _record_launch_phase(graph_attr, phase_name, launch_fn, *, prepare_capture=None):
                if graph_attr == "_reset_common_cuda_graph":
                    order.append("common_graph")
                elif graph_attr == "_reset_cuda_graph":
                    assert phase_name == "task/apply reset"
                    order.append("task_apply_graph")
                else:
                    order.append("unknown_graph")
                return original_launch_phase(graph_attr, phase_name, launch_fn, prepare_capture=prepare_capture)

            def _record_common_after(ctx, *, reset_episode_lengths=True):
                order.append("common_after")
                return original_common_after(ctx, reset_episode_lengths=reset_episode_lengths)

            def _record_scene_reset_after_graph(env_ids=None, env_mask=None, *, graphable_reset_applied=False):
                scene_after_graph_calls.append((env_ids, env_mask, graphable_reset_applied))
                return original_scene_reset_after_graph(
                    env_ids=env_ids, env_mask=env_mask, graphable_reset_applied=graphable_reset_applied
                )

            def _record_noise_reset(env_ids=None):
                order.append("noise")
                return original_noise_reset(env_ids)

            def _record_apply_after():
                order.append("apply_after")
                return original_apply_after()

            def _record_task_prepare(ctx):
                order.append("task_prepare")
                return original_task_prepare(ctx)

            def _record_apply_graphable(ctx):
                order.append("apply_graphable")
                return original_apply_graphable(ctx)

            monkeypatch.setattr(env, "_launch_reset_cuda_graph_phase", _record_launch_phase)
            monkeypatch.setattr(env, "_reset_idx_common_after_graph", _record_common_after)
            monkeypatch.setattr(env.scene, "reset_after_graph", _record_scene_reset_after_graph)
            monkeypatch.setattr(env._action_noise_model, "reset", _record_noise_reset)
            monkeypatch.setattr(env, "_apply_inhand_reset_to_sim_after_graph", _record_apply_after)
            monkeypatch.setattr(env, "_launch_inhand_task_reset_prepare", _record_task_prepare)
            monkeypatch.setattr(env, "_launch_inhand_reset_to_sim_graphable", _record_apply_graphable)

            reset_env_ids = torch.tensor([0, 2], dtype=torch.int32, device=env.device)
            env.reset_buf.zero_()
            env.reset_buf[reset_env_ids.to(torch.long)] = True

            assert env._reset_idx_cuda_graph(reset_env_ids)
            torch.cuda.synchronize()

            assert env._reset_common_cuda_graph is not None
            assert env._reset_cuda_graph is not None
            assert env._reset_common_cuda_graph is not env._reset_cuda_graph
            assert order == [
                "common_graph",
                "common_after",
                "noise",
                "task_apply_graph",
                "task_prepare",
                "apply_graphable",
                "apply_after",
            ]
            assert len(scene_after_graph_calls) == 1
            torch.testing.assert_close(scene_after_graph_calls[0][0], reset_env_ids)
            assert scene_after_graph_calls[0][1] is None
            assert scene_after_graph_calls[0][2] is True
            _assert_intermediates_match_canonical_recompute(env)
        finally:
            gym_env.close()


def test_reset_cuda_graph_force_supports_stateful_noise_reset_outside_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand reset CUDA graph path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.action_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"),
        bias_noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
    )

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            assert env._reset_cuda_graph_enabled
            gym_env.reset()
            obs, reward, _, time_outs, _ = gym_env.step(_zero_actions(env))

            assert time_outs.all()
            assert env._reset_cuda_graph is not None
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["policy"]).all()
        finally:
            gym_env.close()


@pytest.mark.parametrize("task", [_ALLEGRO_TASK, _SHADOW_TASK])
def test_inhand_warp_dones_match_torch_reference_with_max_success(task: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(task, num_envs=6)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.max_consecutive_success = 2
    cfg.success_tolerance = 0.2

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(task, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()

            env._compute_intermediate_values()
            env.goal_rot[0:2] = env.object_rot[0:2]
            env.successes[:] = torch.tensor([0.0, 3.0, 1.0, 2.0, 0.0, 4.0], device=env.device)
            env.episode_length_buf[:] = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.long, device=env.device)

            episode_before = env.episode_length_buf.clone()
            torch_terminated, torch_time_outs = _get_dones_reference_torch(env)
            torch_episode_after = env.episode_length_buf.clone()
            torch_intermediates = {
                "fingertip_pos": env.fingertip_pos.clone(),
                "fingertip_rot": env.fingertip_rot.clone(),
                "fingertip_velocities": env.fingertip_velocities.clone(),
                "object_pos": env.object_pos.clone(),
                "object_rot": env.object_rot.clone(),
                "object_velocities": env.object_velocities.clone(),
                "object_linvel": env.object_linvel.clone(),
                "object_angvel": env.object_angvel.clone(),
            }

            env.episode_length_buf[:] = episode_before
            warp_terminated, warp_time_outs = env._get_dones()
            torch.cuda.synchronize()

            torch.testing.assert_close(warp_terminated, torch_terminated)
            torch.testing.assert_close(warp_time_outs, torch_time_outs)
            torch.testing.assert_close(env.episode_length_buf, torch_episode_after)
            for name, expected in torch_intermediates.items():
                torch.testing.assert_close(getattr(env, name), expected, rtol=1e-5, atol=1e-5)
        finally:
            gym_env.close()


def test_inhand_warp_rewards_match_torch_reference_without_goal_resets():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=5)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.success_tolerance = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._get_dones()
            torch.cuda.synchronize()

            actions = torch.linspace(
                -0.4,
                0.4,
                env.num_envs * env.single_action_space.shape[0],
                dtype=torch.float32,
                device=env.device,
            ).reshape(env.num_envs, env.single_action_space.shape[0])
            env.actions = actions.clone()
            env.reset_buf[:] = torch.tensor([False, True, False, True, False], device=env.device)
            env.reset_goal_buf.zero_()
            env.successes[:] = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0], device=env.device)
            env.consecutive_successes[:] = 1.25

            initial = {
                "successes": env.successes.clone(),
                "reset_goal_buf": env.reset_goal_buf.clone(),
                "consecutive_successes": env.consecutive_successes.clone(),
            }

            torch_reward = _get_rewards_reference_torch(env).clone()
            torch_successes = env.successes.clone()
            torch_reset_goal_buf = env.reset_goal_buf.clone()
            torch_consecutive_successes = env.consecutive_successes.clone()

            env.successes[:] = initial["successes"]
            env.reset_goal_buf[:] = initial["reset_goal_buf"]
            env.consecutive_successes[:] = initial["consecutive_successes"]
            warp_reward = env._get_rewards_warp().clone()
            torch.cuda.synchronize()

            torch.testing.assert_close(warp_reward, torch_reward, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(env.successes, torch_successes, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(env.reset_goal_buf, torch_reset_goal_buf)
            torch.testing.assert_close(env.consecutive_successes, torch_consecutive_successes, rtol=1e-5, atol=1e-5)
        finally:
            gym_env.close()


def test_inhand_warp_rewards_reset_goals_without_nonzero_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.success_tolerance = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._get_dones()
            torch.cuda.synchronize()

            old_goal_rot = env.goal_rot.clone()
            env.reset_goal_buf.zero_()
            env.successes.zero_()
            env.consecutive_successes.zero_()
            env.reset_buf.zero_()
            env.actions.zero_()

            reward = env._get_rewards_warp()
            torch.cuda.synchronize()

            assert torch.isfinite(reward).all()
            torch.testing.assert_close(env.successes, torch.ones_like(env.successes))
            assert not env.reset_goal_buf.any()
            torch.testing.assert_close(
                torch.linalg.norm(env.goal_rot, dim=-1), torch.ones(env.num_envs, device=env.device)
            )
            assert not torch.allclose(env.goal_rot, old_goal_rot)
        finally:
            gym_env.close()


def test_inhand_warp_rewards_updates_goal_marker_when_visual_state_is_needed(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.success_tolerance = 10.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._get_dones()
            torch.cuda.synchronize()

            marker_calls = []

            def _record_visualize(translations=None, orientations=None, scales=None, marker_indices=None):
                marker_calls.append((translations.detach().clone(), orientations.detach().clone()))

            monkeypatch.setattr(env.goal_markers, "visualize", _record_visualize)
            monkeypatch.setattr(env, "_should_sync_goal_markers", lambda: True)
            env.reset_goal_buf.zero_()
            env.successes.zero_()
            env.actions.zero_()

            env._get_rewards()
            torch.cuda.synchronize()

            assert int(env._reward_goal_reset_count_torch.item()) == env.num_envs
            assert len(marker_calls) == 1
            assert not env.reset_goal_buf.any()
            torch.testing.assert_close(marker_calls[0][0], env.goal_pos + env.scene.env_origins)
            torch.testing.assert_close(marker_calls[0][1], env.goal_rot)
        finally:
            gym_env.close()


def test_inhand_warp_rewards_existing_goal_reset_mask_matches_torch_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=5)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.success_tolerance = -1.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._get_dones()
            torch.cuda.synchronize()

            env.actions.zero_()
            env.reset_buf[:] = torch.tensor([False, True, False, True, False], device=env.device)
            env.reset_goal_buf[:] = torch.tensor([True, False, True, False, False], device=env.device)
            env.successes[:] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=env.device)
            env.consecutive_successes[:] = 0.5

            initial = {
                "successes": env.successes.clone(),
                "reset_goal_buf": env.reset_goal_buf.clone(),
                "consecutive_successes": env.consecutive_successes.clone(),
                "goal_rot": env.goal_rot.clone(),
            }
            torch_reward = _get_rewards_reference_torch(env).clone()
            torch_successes = env.successes.clone()
            torch_reset_goal_buf = env.reset_goal_buf.clone()
            torch_consecutive_successes = env.consecutive_successes.clone()

            env.successes[:] = initial["successes"]
            env.reset_goal_buf[:] = initial["reset_goal_buf"]
            env.consecutive_successes[:] = initial["consecutive_successes"]
            env.goal_rot[:] = initial["goal_rot"]
            warp_reward = env._get_rewards_warp().clone()
            torch.cuda.synchronize()

            torch.testing.assert_close(warp_reward, torch_reward, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(env.successes, torch_successes, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(env.reset_goal_buf, torch_reset_goal_buf)
            torch.testing.assert_close(env.consecutive_successes, torch_consecutive_successes, rtol=1e-5, atol=1e-5)
            assert int(env._reward_goal_reset_count_torch.item()) == 2
        finally:
            gym_env.close()


def test_inhand_warp_dones_rewraps_after_storage_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.reset_terminated = env.reset_terminated.clone()

            terminated, time_outs = env._get_dones()
            torch.cuda.synchronize()

            assert terminated is env.reset_terminated
            assert time_outs is env.reset_time_outs
        finally:
            gym_env.close()


def test_inhand_warp_dones_errors_after_metadata_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.reset_terminated = env.reset_terminated.view(2, 2)

            with pytest.raises(
                RuntimeError,
                match="done computation requires the fused Warp path.*reset_terminated shape changed",
            ):
                env._get_dones()
        finally:
            gym_env.close()


def test_inhand_fused_reset_rewraps_after_reset_storage_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"

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
            env.prev_targets = env.prev_targets.clone()

            reset_env_ids = torch.tensor([0, 2], dtype=torch.int32, device=env.device)
            env._reset_idx(reset_env_ids)
            torch.cuda.synchronize()

            torch.testing.assert_close(env.successes[reset_env_ids.long()], torch.zeros(2, device=env.device))
        finally:
            gym_env.close()


def test_inhand_fused_reset_errors_after_reset_metadata_change():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env.prev_targets = env.prev_targets.transpose(0, 1)

            reset_env_ids = torch.tensor([0, 2], dtype=torch.int32, device=env.device)
            with pytest.raises(
                RuntimeError,
                match="reset requires the fused Warp reset path.*prev_targets shape changed",
            ):
                env._reset_idx(reset_env_ids)
        finally:
            gym_env.close()


def test_inhand_reset_idx_uses_fused_warp_body_when_graph_is_off(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=5)
    cfg.reset_cuda_graph = "off"
    cfg.episode_length_s = 10.0
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            assert env._reset_cuda_graph is None

            fused_reset_modes = []
            original_fused_reset = env._run_inhand_fused_reset

            def _record_fused_reset(ctx, *, use_cuda_graph):
                fused_reset_modes.append(use_cuda_graph)
                return original_fused_reset(ctx, use_cuda_graph=use_cuda_graph)

            monkeypatch.setattr(env, "_run_inhand_fused_reset", _record_fused_reset)
            reset_env_ids = torch.tensor([0, 3], dtype=torch.int32, device=env.device)
            env.successes[:] = torch.tensor([2.0, 0.0, 3.0, 4.0, 5.0], device=env.device)

            env._reset_idx(reset_env_ids)
            torch.cuda.synchronize()

            assert fused_reset_modes == [False]
            assert int(env._reset_count_torch.item()) == 2
            assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0)
            torch.testing.assert_close(
                env.hand.data.joint_pos.torch[reset_env_ids.to(torch.long)],
                env.hand.data.default_joint_pos.torch[reset_env_ids.to(torch.long)],
            )
            torch.testing.assert_close(
                env.object.data.root_com_vel_w.torch[reset_env_ids.to(torch.long)],
                torch.zeros_like(env.object.data.root_com_vel_w.torch[reset_env_ids.to(torch.long)]),
            )
        finally:
            gym_env.close()


def test_inhand_reset_idx_snapshots_success_before_reset_events():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=4)
    cfg.reset_cuda_graph = "off"
    cfg.events = _ClearSuccessesResetEventCfg()
    cfg.success_count_threshold = 2
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            reset_env_ids = torch.tensor([0, 2], dtype=torch.int32, device=env.device)
            env.successes[:] = torch.tensor([3.0, 0.0, 4.0, 0.0], device=env.device)

            env._reset_idx(reset_env_ids)
            torch.cuda.synchronize()

            torch.testing.assert_close(env._test_reset_event_env_ids, reset_env_ids)
            torch.testing.assert_close(
                env._last_episode_success[reset_env_ids.to(dtype=torch.long)],
                torch.ones(len(reset_env_ids), dtype=torch.bool, device=env.device),
            )
            assert env.extras["log"]["Metrics/success_rate"] == pytest.approx(1.0)
            torch.testing.assert_close(
                env.successes[reset_env_ids.to(dtype=torch.long)],
                torch.zeros(len(reset_env_ids), device=env.device),
            )
        finally:
            gym_env.close()


def test_inhand_reset_idx_runs_base_reset_before_fused_warp_body(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the in-hand fused Warp reset path.")

    cfg = _make_newton_cfg(_ALLEGRO_TASK, num_envs=3)
    cfg.reset_cuda_graph = "off"
    cfg.events = _RecordOrderedResetEventCfg()
    cfg.action_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0, operation="add"),
        bias_noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
    )
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0

    launcher_args = _make_launcher_args()
    needs_kit, _, _ = compute_kit_requirements(cfg, launcher_args)
    assert not needs_kit

    with launch_simulation(cfg, launcher_args):
        assert not has_kit()
        gym_env = gym.make(_ALLEGRO_TASK, cfg=cfg)
        env = gym_env.unwrapped

        try:
            gym_env.reset()
            env._test_reset_order = []
            env._test_reset_event_env_ids = None

            fused_reset_modes = []
            original_fused_reset = env._run_inhand_fused_reset
            original_prepare = env._launch_inhand_task_reset_prepare
            original_scene_reset_after_graph = env.scene.reset_after_graph
            original_noise_reset = env._action_noise_model.reset

            def _record_fused_reset(ctx, *, use_cuda_graph):
                fused_reset_modes.append(use_cuda_graph)
                return original_fused_reset(ctx, use_cuda_graph=use_cuda_graph)

            def _record_prepare(ctx):
                env._test_reset_order.append("prepare")
                return original_prepare(ctx)

            def _record_scene_reset_after_graph(env_ids=None, env_mask=None, *, graphable_reset_applied=False):
                env._test_reset_order.append("scene")
                return original_scene_reset_after_graph(
                    env_ids=env_ids, env_mask=env_mask, graphable_reset_applied=graphable_reset_applied
                )

            def _record_noise_reset(env_ids=None):
                env._test_reset_order.append("noise")
                return original_noise_reset(env_ids)

            monkeypatch.setattr(env, "_run_inhand_fused_reset", _record_fused_reset)
            monkeypatch.setattr(env, "_launch_inhand_task_reset_prepare", _record_prepare)
            monkeypatch.setattr(env.scene, "reset_after_graph", _record_scene_reset_after_graph)
            monkeypatch.setattr(env._action_noise_model, "reset", _record_noise_reset)
            for actuator in env.hand.actuators.values():
                original_actuator_reset = actuator.reset

                def _record_actuator_reset(env_ids, original_actuator_reset=original_actuator_reset):
                    env._test_reset_order.append("actuator")
                    return original_actuator_reset(env_ids)

                monkeypatch.setattr(actuator, "reset", _record_actuator_reset)

            reset_env_ids = torch.tensor([1], dtype=torch.int32, device=env.device)

            env._reset_idx(reset_env_ids)
            torch.cuda.synchronize()

            assert fused_reset_modes == [False]
            torch.testing.assert_close(env._test_reset_event_env_ids, reset_env_ids)
            assert env._test_reset_order[0] == "scene"
            assert "actuator" in env._test_reset_order
            assert env._test_reset_order.index("actuator") < env._test_reset_order.index("event")
            assert env._test_reset_order.index("event") < env._test_reset_order.index("noise")
            assert env._test_reset_order.index("noise") < env._test_reset_order.index("prepare")
            torch.testing.assert_close(
                env.hand.data.joint_pos.torch[reset_env_ids.to(torch.long)],
                env.hand.data.default_joint_pos.torch[reset_env_ids.to(torch.long)],
            )
        finally:
            gym_env.close()
