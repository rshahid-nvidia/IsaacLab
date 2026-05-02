# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct in-hand manipulation environments.

See :doc:`/how-to/add_fused_reset_task` for the fused reset task authoring guide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.cuda_graph import ResetContext, ResetGraphPhase
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils.profiling import nvtx_range_pop, nvtx_range_push
from isaaclab.utils.warp_view_registry import CudaGraphTargetRegistry, WarpViewRegistry

if TYPE_CHECKING:
    from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


@wp.func
def _randomize_rotation_wp(rand0: wp.float32, rand1: wp.float32, x_axis: wp.vec3f, y_axis: wp.vec3f) -> wp.quatf:
    return wp.quat_from_axis_angle(x_axis, rand0 * wp.pi) * wp.quat_from_axis_angle(y_axis, rand1 * wp.pi)


@wp.kernel
def _initialize_reset_rng(seed: wp.int32, state: wp.array(dtype=wp.uint32)):
    env_id = wp.tid()
    state[env_id] = wp.rand_init(seed, wp.int32(env_id))


@wp.kernel
def _clear_reset_stats(
    reset_count: wp.array(dtype=wp.int32),
    reset_success_count: wp.array(dtype=wp.int32),
):
    reset_count[0] = wp.int32(0)
    reset_success_count[0] = wp.int32(0)


@wp.kernel
def _snapshot_inhand_reset_success(
    env_mask: wp.array(dtype=wp.bool),
    success_count_threshold: wp.int32,
    successes: wp.array(dtype=wp.float32),
    last_episode_success: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    reset_success_count: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    was_success = successes[env_id] >= wp.float32(success_count_threshold)
    last_episode_success[env_id] = was_success
    wp.atomic_add(reset_count, 0, wp.int32(1))
    if was_success:
        wp.atomic_add(reset_success_count, 0, wp.int32(1))


@wp.kernel
def _prepare_inhand_reset(
    env_mask: wp.array(dtype=wp.bool),
    default_object_pose: wp.array(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    reset_position_noise: wp.float32,
    x_axis: wp.vec3f,
    y_axis: wp.vec3f,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    reset_dof_pos_noise: wp.float32,
    reset_dof_vel_noise: wp.float32,
    num_dofs: wp.int32,
    rng_state: wp.array(dtype=wp.uint32),
    successes: wp.array(dtype=wp.float32),
    reset_goal_buf: wp.array(dtype=wp.bool),
    episode_length_buf: wp.array(dtype=wp.int64),
    goal_rot: wp.array(dtype=wp.quatf),
    object_pose_out: wp.array(dtype=wp.transformf),
    object_velocity_out: wp.array(dtype=wp.spatial_vectorf),
    joint_pos_out: wp.array2d(dtype=wp.float32),
    joint_vel_out: wp.array2d(dtype=wp.float32),
    prev_targets: wp.array2d(dtype=wp.float32),
    cur_targets: wp.array2d(dtype=wp.float32),
    hand_dof_targets: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    reset_goal_buf[env_id] = False
    episode_length_buf[env_id] = wp.int64(0)

    rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    goal_rot[env_id] = _randomize_rotation_wp(rand0, rand1, x_axis, y_axis)

    nx = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    ny = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    nz = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    object_pos = (
        wp.transform_get_translation(default_object_pose[env_id])
        + env_origins[env_id]
        + reset_position_noise * wp.vec3f(nx, ny, nz)
    )

    rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
    rng_state[env_id] += wp.uint32(1)
    object_pose_out[env_id] = wp.transform(object_pos, _randomize_rotation_wp(rand0, rand1, x_axis, y_axis))
    object_velocity_out[env_id] = wp.spatial_vectorf(
        wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    )

    for dof_id in range(num_dofs):
        dof_pos_noise = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        delta_max = upper_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
        delta_min = lower_limits[env_id, dof_id] - default_joint_pos[env_id, dof_id]
        rand_delta = delta_min + (delta_max - delta_min) * wp.float32(0.5) * dof_pos_noise
        pos = default_joint_pos[env_id, dof_id] + reset_dof_pos_noise * rand_delta

        dof_vel_noise = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        vel = default_joint_vel[env_id, dof_id] + reset_dof_vel_noise * dof_vel_noise

        joint_pos_out[env_id, dof_id] = pos
        joint_vel_out[env_id, dof_id] = vel
        prev_targets[env_id, dof_id] = pos
        cur_targets[env_id, dof_id] = pos
        hand_dof_targets[env_id, dof_id] = pos

    successes[env_id] = wp.float32(0.0)


@wp.kernel
def _compute_inhand_intermediate(
    hand_body_pose_w: wp.array2d(dtype=wp.transformf),
    hand_body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    finger_bodies: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.vec3f),
    object_root_pose_w: wp.array(dtype=wp.transformf),
    object_root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    num_fingertips: wp.int32,
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    fingertip_rot: wp.array2d(dtype=wp.quatf),
    fingertip_velocities: wp.array2d(dtype=wp.spatial_vectorf),
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    object_velocities: wp.array(dtype=wp.spatial_vectorf),
    object_linvel: wp.array(dtype=wp.vec3f),
    object_angvel: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()

    for i in range(num_fingertips):
        body_id = finger_bodies[i]
        pose = hand_body_pose_w[env_id, body_id]
        fingertip_pos[env_id, i] = wp.transform_get_translation(pose) - env_origins[env_id]
        fingertip_rot[env_id, i] = wp.transform_get_rotation(pose)
        fingertip_velocities[env_id, i] = hand_body_vel_w[env_id, body_id]

    obj_pose = object_root_pose_w[env_id]
    obj_vel = object_root_vel_w[env_id]
    object_pos[env_id] = wp.transform_get_translation(obj_pose) - env_origins[env_id]
    object_rot[env_id] = wp.transform_get_rotation(obj_pose)
    object_velocities[env_id] = obj_vel
    object_linvel[env_id] = wp.vec3f(obj_vel[0], obj_vel[1], obj_vel[2])
    object_angvel[env_id] = wp.vec3f(obj_vel[3], obj_vel[4], obj_vel[5])


@wp.func
def _rotation_distance_wp(object_rot: wp.quatf, target_rot: wp.quatf) -> wp.float32:
    quat_diff = object_rot * wp.quatf(-target_rot[0], -target_rot[1], -target_rot[2], target_rot[3])
    vec_norm = wp.sqrt(quat_diff[0] * quat_diff[0] + quat_diff[1] * quat_diff[1] + quat_diff[2] * quat_diff[2])
    return wp.float32(2.0) * wp.asin(wp.min(vec_norm, wp.float32(1.0)))


@wp.kernel
def _compute_inhand_intermediate_and_dones(
    hand_body_pose_w: wp.array2d(dtype=wp.transformf),
    hand_body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    finger_bodies: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.vec3f),
    object_root_pose_w: wp.array(dtype=wp.transformf),
    object_root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    num_fingertips: wp.int32,
    in_hand_pos: wp.array(dtype=wp.vec3f),
    goal_rot: wp.array(dtype=wp.quatf),
    successes: wp.array(dtype=wp.float32),
    episode_length_buf: wp.array(dtype=wp.int64),
    max_episode_length_minus_one: wp.int64,
    fall_dist: wp.float32,
    max_consecutive_success: wp.int32,
    success_tolerance: wp.float32,
    fingertip_pos: wp.array2d(dtype=wp.vec3f),
    fingertip_rot: wp.array2d(dtype=wp.quatf),
    fingertip_velocities: wp.array2d(dtype=wp.spatial_vectorf),
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    object_velocities: wp.array(dtype=wp.spatial_vectorf),
    object_linvel: wp.array(dtype=wp.vec3f),
    object_angvel: wp.array(dtype=wp.vec3f),
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()

    for i in range(num_fingertips):
        body_id = finger_bodies[i]
        pose = hand_body_pose_w[env_id, body_id]
        fingertip_pos[env_id, i] = wp.transform_get_translation(pose) - env_origins[env_id]
        fingertip_rot[env_id, i] = wp.transform_get_rotation(pose)
        fingertip_velocities[env_id, i] = hand_body_vel_w[env_id, body_id]

    obj_pose = object_root_pose_w[env_id]
    obj_vel = object_root_vel_w[env_id]
    obj_pos = wp.transform_get_translation(obj_pose) - env_origins[env_id]
    obj_rot = wp.transform_get_rotation(obj_pose)
    object_pos[env_id] = obj_pos
    object_rot[env_id] = obj_rot
    object_velocities[env_id] = obj_vel
    object_linvel[env_id] = wp.vec3f(obj_vel[0], obj_vel[1], obj_vel[2])
    object_angvel[env_id] = wp.vec3f(obj_vel[3], obj_vel[4], obj_vel[5])

    goal_delta = obj_pos - in_hand_pos[env_id]
    goal_dist = wp.sqrt(goal_delta[0] * goal_delta[0] + goal_delta[1] * goal_delta[1] + goal_delta[2] * goal_delta[2])
    reset_terminated[env_id] = goal_dist >= fall_dist

    if max_consecutive_success > wp.int32(0):
        rot_dist = _rotation_distance_wp(obj_rot, goal_rot[env_id])
        if wp.abs(rot_dist) <= success_tolerance:
            episode_length_buf[env_id] = wp.int64(0)
        reset_time_outs[env_id] = episode_length_buf[env_id] >= max_episode_length_minus_one or successes[
            env_id
        ] >= wp.float32(max_consecutive_success)
    else:
        reset_time_outs[env_id] = episode_length_buf[env_id] >= max_episode_length_minus_one


@wp.kernel
def _compute_inhand_rewards_and_reset_goals(
    reset_buf: wp.array(dtype=wp.bool),
    reset_goal_buf: wp.array(dtype=wp.bool),
    successes: wp.array(dtype=wp.float32),
    object_pos: wp.array(dtype=wp.vec3f),
    object_rot: wp.array(dtype=wp.quatf),
    target_pos: wp.array(dtype=wp.vec3f),
    target_rot: wp.array(dtype=wp.quatf),
    actions: wp.array2d(dtype=wp.float32),
    num_actions: wp.int32,
    dist_reward_scale: wp.float32,
    rot_reward_scale: wp.float32,
    rot_eps: wp.float32,
    action_penalty_scale: wp.float32,
    success_tolerance: wp.float32,
    reach_goal_bonus: wp.float32,
    fall_dist: wp.float32,
    fall_penalty: wp.float32,
    rng_state: wp.array(dtype=wp.uint32),
    x_axis: wp.vec3f,
    y_axis: wp.vec3f,
    reward: wp.array(dtype=wp.float32),
    reset_count_terms: wp.array(dtype=wp.float32),
    finished_success_terms: wp.array(dtype=wp.float32),
    goal_reset_terms: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()

    delta = object_pos[env_id] - target_pos[env_id]
    goal_dist = wp.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    rot_dist = _rotation_distance_wp(object_rot[env_id], target_rot[env_id])

    action_penalty = wp.float32(0.0)
    for action_id in range(num_actions):
        action = actions[env_id, action_id]
        action_penalty += action * action

    rew = goal_dist * dist_reward_scale + wp.float32(1.0) / (wp.abs(rot_dist) + rot_eps) * rot_reward_scale
    rew += action_penalty * action_penalty_scale

    goal_reset = reset_goal_buf[env_id] or wp.abs(rot_dist) <= success_tolerance
    success_count = successes[env_id]
    if goal_reset:
        success_count += wp.float32(1.0)
        rew += reach_goal_bonus

    fell = goal_dist >= fall_dist
    if fell:
        rew += fall_penalty

    reset_for_success_average = fell or reset_buf[env_id]
    reset_goal_buf[env_id] = False
    successes[env_id] = success_count
    reward[env_id] = rew
    reset_count_terms[env_id] = wp.float32(1.0) if reset_for_success_average else wp.float32(0.0)
    finished_success_terms[env_id] = success_count if reset_for_success_average else wp.float32(0.0)

    goal_reset_terms[env_id] = wp.int32(0)
    if goal_reset:
        rand0 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        rand1 = wp.randf(rng_state[env_id], wp.float32(-1.0), wp.float32(1.0))
        rng_state[env_id] += wp.uint32(1)
        target_rot[env_id] = _randomize_rotation_wp(rand0, rand1, x_axis, y_axis)
        goal_reset_terms[env_id] = wp.int32(1)


@wp.kernel
def _finalize_inhand_rewards(
    reset_count_terms: wp.array(dtype=wp.float32),
    finished_success_terms: wp.array(dtype=wp.float32),
    num_envs: wp.int32,
    av_factor: wp.float32,
    goal_reset_terms: wp.array(dtype=wp.int32),
    consecutive_successes: wp.array(dtype=wp.float32),
    goal_reset_count: wp.array(dtype=wp.int32),
):
    reset_count = wp.float32(0.0)
    finished_successes = wp.float32(0.0)
    goal_count = wp.int32(0)
    for env_id in range(num_envs):
        reset_count += reset_count_terms[env_id]
        finished_successes += finished_success_terms[env_id]
        goal_count += goal_reset_terms[env_id]
    if reset_count > wp.float32(0.0):
        consecutive_successes[0] = (
            av_factor * finished_successes / reset_count + (wp.float32(1.0) - av_factor) * consecutive_successes[0]
        )
    goal_reset_count[0] = goal_count


class InHandManipulationEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.data.joint_limits.torch.to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_pose.torch[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self._last_episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # bind backend-optimal write methods for the semantic Torch fallback path
        self._is_newton_backend = "newton" in self.sim.physics_manager.__name__.lower()
        if self._is_newton_backend:
            self._set_joint_pos_target = self.hand.set_joint_position_target
            self._write_obj_root_pose = self.object.write_root_pose_to_sim
            self._write_obj_root_vel = self.object.write_root_velocity_to_sim
            self._write_hand_joint_pos = self.hand.write_joint_position_to_sim
            self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim
        else:
            self._set_joint_pos_target = self.hand.set_joint_position_target_index
            self._write_obj_root_pose = self.object.write_root_pose_to_sim_index
            self._write_obj_root_vel = self.object.write_root_velocity_to_sim_index
            self._write_hand_joint_pos = self.hand.write_joint_position_to_sim_index
            self._write_hand_joint_vel = self.hand.write_joint_velocity_to_sim_index

        self._inhand_warp_step_enabled = self._is_newton_backend and self._inhand_warp_state_buffers_available()
        if self._inhand_warp_step_enabled:
            self._setup_inhand_warp_step_buffers()

        self._inhand_fused_reset_enabled = (
            self._inhand_warp_step_enabled and not self._get_inhand_fused_reset_blockers()
        )
        if self._inhand_fused_reset_enabled:
            self._set_joint_pos_target_mask = self.hand.set_joint_position_target_mask
            self._write_obj_root_pose_mask = self.object.write_root_pose_to_sim_mask
            self._write_obj_root_vel_mask = self.object.write_root_velocity_to_sim_mask
            self._write_hand_joint_pos_mask = self.hand.write_joint_position_to_sim_mask
            self._write_hand_joint_vel_mask = self.hand.write_joint_velocity_to_sim_mask
            self._set_joint_pos_target_mask_after_graph = self.hand.set_joint_position_target_mask_after_graph
            self._write_obj_root_pose_mask_after_graph = self.object.write_root_pose_to_sim_mask_after_graph
            self._write_obj_root_vel_mask_after_graph = self.object.write_root_velocity_to_sim_mask_after_graph
            self._write_hand_joint_pos_mask_after_graph = self.hand.write_joint_position_to_sim_mask_after_graph
            self._write_hand_joint_vel_mask_after_graph = self.hand.write_joint_velocity_to_sim_mask_after_graph
            self._setup_inhand_fused_reset_buffers()

        self._configure_reset_cuda_graph(path_name="In-hand reset CUDA graph path")

    def seed(self, seed: int = -1) -> int:
        """Set global RNG state and reseed in-hand Warp RNG buffers when they exist."""

        seed = DirectRLEnv.seed(seed)
        self._reseed_inhand_warp_rng(seed)
        return seed

    def _reseed_inhand_warp_rng(self, seed: int) -> None:
        """Reinitialize persistent Warp RNG states that replaced Torch reset sampling."""

        if hasattr(self, "_goal_reset_rng_state_wp"):
            wp.launch(
                _initialize_reset_rng,
                dim=self.num_envs,
                inputs=[int(seed) + 104729, self._goal_reset_rng_state_wp],
                device=self.device,
            )
        if hasattr(self, "_reset_rng_state_wp"):
            wp.launch(
                _initialize_reset_rng,
                dim=self.num_envs,
                inputs=[int(seed), self._reset_rng_state_wp],
                device=self.device,
            )

    def _inhand_warp_state_buffers_available(self) -> bool:
        try:
            return all(
                hasattr(buffer, "warp")
                for buffer in (
                    self.hand.data.body_link_pose_w,
                    self.hand.data.body_com_vel_w,
                    self.object.data.root_link_pose_w,
                    self.object.data.root_com_vel_w,
                )
            )
        except Exception:
            return False

    def _setup_inhand_warp_step_buffers(self) -> None:
        self._inhand_warp_views = WarpViewRegistry(self, device=self.device)
        self._register_inhand_warp_step_views()
        self._inhand_warp_views.refresh(groups=("step",))
        self._finger_bodies_wp = wp.array(self.finger_bodies, dtype=wp.int32, device=self.device)

        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._register_inhand_reward_views()
        self._inhand_warp_views.refresh(groups=("reward",))
        self._reward_reset_count_terms_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reward_finished_success_terms_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reward_goal_reset_terms_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._reward_goal_reset_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reward_goal_reset_count_torch = wp.to_torch(self._reward_goal_reset_count_wp)

        self._x_unit_vec_wp = wp.vec3f(1.0, 0.0, 0.0)
        self._y_unit_vec_wp = wp.vec3f(0.0, 1.0, 0.0)
        self._goal_reset_rng_state_wp = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        self._reseed_inhand_warp_rng(0 if self.cfg.seed is None else int(self.cfg.seed))

        self._graph_fingertip_pos_wp = wp.zeros(
            (self.num_envs, self.num_fingertips), dtype=wp.vec3f, device=self.device
        )
        self._graph_fingertip_rot_wp = wp.zeros(
            (self.num_envs, self.num_fingertips), dtype=wp.quatf, device=self.device
        )
        self._graph_fingertip_velocities_wp = wp.zeros(
            (self.num_envs, self.num_fingertips), dtype=wp.spatial_vectorf, device=self.device
        )
        self._graph_object_pos_wp = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._graph_object_rot_wp = wp.zeros(self.num_envs, dtype=wp.quatf, device=self.device)
        self._graph_object_velocities_wp = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)
        self._graph_object_linvel_wp = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._graph_object_angvel_wp = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)

        self._graph_fingertip_pos_torch = wp.to_torch(self._graph_fingertip_pos_wp)
        self._graph_fingertip_rot_torch = wp.to_torch(self._graph_fingertip_rot_wp)
        self._graph_fingertip_velocities_torch = wp.to_torch(self._graph_fingertip_velocities_wp)
        self._graph_object_pos_torch = wp.to_torch(self._graph_object_pos_wp)
        self._graph_object_rot_torch = wp.to_torch(self._graph_object_rot_wp)
        self._graph_object_velocities_torch = wp.to_torch(self._graph_object_velocities_wp)
        self._graph_object_linvel_torch = wp.to_torch(self._graph_object_linvel_wp)
        self._graph_object_angvel_torch = wp.to_torch(self._graph_object_angvel_wp)

        self._refresh_inhand_warp_step_buffers()

    def _register_inhand_warp_step_views(self) -> None:
        views = self._inhand_warp_views
        views.register_torch_view(
            "episode_length_buf",
            lambda: self.episode_length_buf,
            warp_attr="_episode_length_buf_wp",
            warp_dtype=wp.int64,
            torch_dtype=torch.int64,
            expected_shape=(self.num_envs,),
            groups=("step", "reset", "reset_graph"),
        )
        views.register_torch_view(
            "successes",
            lambda: self.successes,
            warp_attr="_successes_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs,),
            groups=("step", "reset", "reset_graph"),
        )
        views.register_torch_view(
            "last_episode_success",
            lambda: self._last_episode_success,
            warp_attr="_last_episode_success_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("reset", "reset_graph"),
        )
        views.register_torch_view(
            "consecutive_successes",
            lambda: self.consecutive_successes,
            warp_attr="_consecutive_successes_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=(1,),
            groups=("step",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "goal_rot",
            lambda: self.goal_rot,
            warp_attr="_goal_rot_wp",
            warp_dtype=wp.quatf,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 4),
            groups=("step", "reset", "reset_graph"),
        )
        views.register_torch_view(
            "in_hand_pos",
            lambda: self.in_hand_pos,
            warp_attr="_in_hand_pos_wp",
            warp_dtype=wp.vec3f,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 3),
            groups=("step",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "scene.env_origins",
            lambda: self.scene.env_origins,
            warp_attr="_env_origins_wp",
            warp_dtype=wp.vec3f,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 3),
            groups=("step", "reset", "reset_graph"),
        )
        views.register_torch_view(
            "reset_terminated",
            lambda: self.reset_terminated,
            warp_attr="_reset_terminated_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("step",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "reset_time_outs",
            lambda: self.reset_time_outs,
            warp_attr="_reset_time_outs_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("step",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "reset_buf",
            lambda: self.reset_buf,
            warp_attr="_reset_buf_step_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("step",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "reset_goal_buf",
            lambda: self.reset_goal_buf,
            warp_attr="_reset_goal_buf_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("step", "reset", "reset_graph"),
        )
        self._register_inhand_state_views()

    def _register_inhand_reward_views(self) -> None:
        self._inhand_warp_views.register_torch_view(
            "reward_buf",
            lambda: self.reward_buf,
            warp_attr="_reward_buf_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs,),
            groups=("step", "reward"),
            track_torch=False,
            track_warp=False,
        )

    def _register_inhand_state_views(self) -> None:
        views = self._inhand_warp_views
        views.register_existing_view(
            "hand.body_link_pose_w",
            lambda: self.hand.data.body_link_pose_w.torch,
            lambda: self.hand.data.body_link_pose_w.warp,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, len(self.hand.body_names), 7),
            groups=("step", "state", "reset_graph"),
            warp_attr="_hand_body_pose_w_wp",
            warp_name="hand.body_link_pose_w",
        )
        views.register_existing_view(
            "hand.body_com_vel_w",
            lambda: self.hand.data.body_com_vel_w.torch,
            lambda: self.hand.data.body_com_vel_w.warp,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, len(self.hand.body_names), 6),
            groups=("step", "state", "reset_graph"),
            warp_attr="_hand_body_vel_w_wp",
            warp_name="hand.body_com_vel_w",
        )
        views.register_existing_view(
            "object.root_link_pose_w",
            lambda: self.object.data.root_link_pose_w.torch,
            lambda: self.object.data.root_link_pose_w.warp,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 7),
            groups=("step", "state", "reset_graph"),
            warp_attr="_object_root_pose_w_wp",
        )
        views.register_existing_view(
            "object.root_com_vel_w",
            lambda: self.object.data.root_com_vel_w.torch,
            lambda: self.object.data.root_com_vel_w.warp,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 6),
            groups=("step", "state", "reset_graph"),
            warp_attr="_object_root_vel_w_wp",
        )

    def _check_inhand_warp_step_compatibility(self) -> str | None:
        """Return why direct Warp step kernels cannot run, ignoring CUDA graph replay stability."""

        if not self._inhand_warp_state_buffers_available():
            return "required Warp state buffers are not available"
        return self._inhand_warp_views.check_compatibility(groups=("step", "reward"))

    def _refresh_inhand_warp_step_buffers(self) -> None:
        """Refresh Warp views for direct kernels after possible Torch tensor rebinding."""

        self._inhand_warp_views.refresh(groups=("step", "reward", "state"))

    def _refresh_inhand_warp_state_inputs(self) -> None:
        """Refresh backend-owned Warp state views after simulation writes or FK."""

        self._inhand_warp_views.refresh(groups=("state",))

    def _check_inhand_warp_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> str | None:
        """Return why a transient tensor cannot be consumed by the in-hand Warp kernels."""

        if not torch.is_tensor(tensor):
            return f"{name} is not a torch.Tensor"
        if tuple(tensor.shape) != shape:
            return f"{name} shape changed from {shape} to {tuple(tensor.shape)}"
        if tensor.dtype != dtype:
            return f"{name} dtype changed from {dtype} to {tensor.dtype}"
        if tensor.device != torch.device(self.device):
            return f"{name} device changed from {self.device} to {tensor.device}"
        return None

    def _require_inhand_warp_step(self, context: str) -> None:
        """Validate direct Warp step kernels and refresh their Torch-backed Warp views."""

        reason = self._check_inhand_warp_step_compatibility()
        if reason is not None:
            raise RuntimeError(f"In-hand {context} requires the fused Warp path, but it is not compatible: {reason}.")
        self._refresh_inhand_warp_step_buffers()

    def _should_sync_goal_markers(self) -> bool:
        """Return whether visual goal-marker state can be observed this step."""

        return self.has_rtx_sensors or self.sim.has_gui or self.sim.has_active_visualizers()

    def _setup_inhand_fused_reset_buffers(self) -> None:
        if getattr(self, "_inhand_fused_reset_buffers_ready", False):
            return

        self._reset_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._register_inhand_fused_reset_views()
        self._inhand_warp_views.refresh(groups=("reset",))

        self._reset_rng_state_wp = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        self._reseed_inhand_warp_rng(0 if self.cfg.seed is None else int(self.cfg.seed))

        self._reset_empty_mask_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._reset_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reset_success_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reset_count_torch = wp.to_torch(self._reset_count_wp)
        self._reset_success_count_torch = wp.to_torch(self._reset_success_count_wp)

        self._reset_object_pose_wp = wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device)
        self._reset_object_velocity_wp = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)
        self._reset_joint_pos_wp = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self._reset_joint_vel_wp = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self._register_inhand_reset_graph_targets()
        self._refresh_inhand_reset_torch_views()
        self._inhand_fused_reset_buffers_ready = True

    def _register_inhand_fused_reset_views(self) -> None:
        views = self._inhand_warp_views
        views.register_torch_view(
            "reset_env_mask",
            lambda: self._reset_env_mask,
            warp_attr="_reset_env_mask_wp",
            warp_dtype=wp.bool,
            torch_dtype=torch.bool,
            expected_shape=(self.num_envs,),
            groups=("reset",),
            track_torch=False,
            track_warp=False,
        )
        views.register_torch_view(
            "hand_dof_lower_limits",
            lambda: self.hand_dof_lower_limits,
            warp_attr="_lower_limits_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
            warp_name="lower_limits_wp",
        )
        views.register_torch_view(
            "hand_dof_upper_limits",
            lambda: self.hand_dof_upper_limits,
            warp_attr="_upper_limits_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
            warp_name="upper_limits_wp",
        )
        views.register_torch_view(
            "prev_targets",
            lambda: self.prev_targets,
            warp_attr="_prev_targets_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
        )
        views.register_torch_view(
            "cur_targets",
            lambda: self.cur_targets,
            warp_attr="_cur_targets_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
        )
        views.register_torch_view(
            "hand_dof_targets",
            lambda: self.hand_dof_targets,
            warp_attr="_hand_dof_targets_wp",
            warp_dtype=wp.float32,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
        )
        views.register_existing_view(
            "object.default_root_pose",
            lambda: self.object.data.default_root_pose.torch,
            lambda: self.object.data.default_root_pose.warp,
            torch_dtype=torch.float32,
            expected_shape=(self.num_envs, 7),
            groups=("reset", "reset_graph"),
            track_torch=True,
            warp_name="object.default_root_pose_wp",
        )
        views.register_existing_view(
            "hand.default_joint_pos",
            lambda: self.hand.data.default_joint_pos.torch,
            lambda: self.hand.data.default_joint_pos.warp,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
            track_torch=True,
            warp_name="hand.default_joint_pos_wp",
        )
        views.register_existing_view(
            "hand.default_joint_vel",
            lambda: self.hand.data.default_joint_vel.torch,
            lambda: self.hand.data.default_joint_vel.warp,
            torch_dtype=torch.float32,
            expected_shape=lambda: (self.num_envs, self.num_hand_dofs),
            groups=("reset", "reset_graph"),
            track_torch=True,
            warp_name="hand.default_joint_vel_wp",
        )

    def _register_inhand_reset_graph_targets(self) -> None:
        targets = CudaGraphTargetRegistry()
        targets.update(
            {
                "reset_mask_wp": lambda: self._reset_cuda_graph_mask_wp,
                "reset_rng_state": lambda: self._reset_rng_state_wp,
                "reset_count": lambda: self._reset_count_wp,
                "reset_success_count": lambda: self._reset_success_count_wp,
                "finger_bodies": lambda: self._finger_bodies_wp,
                "reset_object_pose": lambda: self._reset_object_pose_wp,
                "reset_object_velocity": lambda: self._reset_object_velocity_wp,
                "reset_joint_pos": lambda: self._reset_joint_pos_wp,
                "reset_joint_vel": lambda: self._reset_joint_vel_wp,
                "graph_fingertip_pos": lambda: self._graph_fingertip_pos_wp,
                "graph_fingertip_rot": lambda: self._graph_fingertip_rot_wp,
                "graph_fingertip_velocities": lambda: self._graph_fingertip_velocities_wp,
                "graph_object_pos": lambda: self._graph_object_pos_wp,
                "graph_object_rot": lambda: self._graph_object_rot_wp,
                "graph_object_velocities": lambda: self._graph_object_velocities_wp,
                "graph_object_linvel": lambda: self._graph_object_linvel_wp,
                "graph_object_angvel": lambda: self._graph_object_angvel_wp,
                "object.body_com_acc_w": lambda: self.object.data._body_com_acc_w.data,
                "object.root_view.articulation_ids": lambda: self.object.root_view.articulation_ids,
                "hand.root_view.articulation_ids": lambda: self.hand.root_view.articulation_ids,
                "hand.all_joint_mask": lambda: self.hand._ALL_JOINT_MASK,
                "hand.joint_pos_target": lambda: self.hand.data._joint_pos_target,
                "hand.joint_pos": lambda: self.hand.data.joint_pos.warp,
                "hand.joint_vel": lambda: self.hand.data.joint_vel.warp,
                "hand.previous_joint_vel": lambda: self.hand.data._previous_joint_vel,
                "hand.joint_acc": lambda: self.hand.data._joint_acc.data,
            }
        )
        self._inhand_reset_graph_targets = targets

    def _refresh_inhand_reset_torch_views(self) -> None:
        """Refresh Torch views for Warp-owned reset buffers after possible Warp array rebinding."""

        self._reset_count_torch = wp.to_torch(self._reset_count_wp)
        self._reset_success_count_torch = wp.to_torch(self._reset_success_count_wp)
        self._graph_fingertip_pos_torch = wp.to_torch(self._graph_fingertip_pos_wp)
        self._graph_fingertip_rot_torch = wp.to_torch(self._graph_fingertip_rot_wp)
        self._graph_fingertip_velocities_torch = wp.to_torch(self._graph_fingertip_velocities_wp)
        self._graph_object_pos_torch = wp.to_torch(self._graph_object_pos_wp)
        self._graph_object_rot_torch = wp.to_torch(self._graph_object_rot_wp)
        self._graph_object_velocities_torch = wp.to_torch(self._graph_object_velocities_wp)
        self._graph_object_linvel_torch = wp.to_torch(self._graph_object_linvel_wp)
        self._graph_object_angvel_torch = wp.to_torch(self._graph_object_angvel_wp)

    def _setup_reset_cuda_graph_buffers(self) -> None:
        if not self._inhand_fused_reset_enabled:
            raise RuntimeError("fused in-hand reset path is not enabled")
        self._setup_inhand_fused_reset_buffers()
        self._require_inhand_warp_step("reset CUDA graph")
        self._require_inhand_fused_reset("reset CUDA graph")
        self._refresh_inhand_reset_torch_views()
        self._reset_cuda_graph_mask_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
        self._clear_reset_cuda_graph_captures()

    def _clear_reset_cuda_graph_captures(self) -> None:
        super()._clear_reset_cuda_graph_captures()
        self._reset_common_cuda_graph = None
        self._reset_apply_cuda_graph = None

    def _warmup_reset_cuda_graph(self) -> None:
        empty_ctx = ResetContext(env_ids=None, reset_mask_wp=self._reset_empty_mask_wp)
        self._reset_idx_common_graphable(empty_ctx, reset_episode_lengths=False)
        self._launch_inhand_task_reset_graphable(empty_ctx)
        self._prepare_inhand_reset_to_sim_capture_state()
        self._launch_inhand_reset_to_sim_graphable(empty_ctx)

    def _reset_cuda_graph_tensors(self) -> dict[str, torch.Tensor | wp.array]:
        """Return tensors whose storage and metadata must remain stable for reset graph replay."""

        tensors = super()._reset_cuda_graph_tensors()
        tensors.update(self._inhand_warp_views.graph_tensors(groups=("reset_graph",)))
        tensors.update(self._inhand_reset_graph_targets.tensors())
        physics_manager = self.sim.physics_manager
        if getattr(physics_manager, "_world_reset_mask", None) is not None:
            tensors["physics.world_reset_mask"] = physics_manager._world_reset_mask
        if getattr(physics_manager, "_fk_reset_mask", None) is not None:
            tensors["physics.fk_reset_mask"] = physics_manager._fk_reset_mask
        return tensors

    def _reset_cuda_graph_constants(self) -> dict[str, float | int]:
        """Return scalar reset parameters baked into the captured reset graph."""

        return {
            "reset_position_noise": float(self.cfg.reset_position_noise),
            "reset_dof_pos_noise": float(self.cfg.reset_dof_pos_noise),
            "reset_dof_vel_noise": float(self.cfg.reset_dof_vel_noise),
            "success_count_threshold": int(self.cfg.success_count_threshold),
            "num_hand_dofs": int(self.num_hand_dofs),
            "num_fingertips": int(self.num_fingertips),
            "object_num_bodies": int(getattr(self.object.data, "_num_bodies", 1)),
        }

    def _get_reset_cuda_graph_blockers(self) -> list[str]:
        reasons = super()._get_reset_cuda_graph_blockers()
        if not self._is_newton_backend:
            reasons.append("physics backend is not Newton")
        if not self._inhand_warp_step_enabled:
            reasons.append("fused Warp step path is not enabled")
        reasons.extend(self._get_inhand_fused_reset_blockers())
        return reasons

    def _get_inhand_fused_reset_blockers(self) -> list[str]:
        reasons: list[str] = []
        required = (
            (self.hand, "set_joint_position_target_mask"),
            (self.hand, "write_joint_position_to_sim_mask"),
            (self.hand, "write_joint_velocity_to_sim_mask"),
            (self.hand, "set_joint_position_target_mask_after_graph"),
            (self.hand, "write_joint_position_to_sim_mask_after_graph"),
            (self.hand, "write_joint_velocity_to_sim_mask_after_graph"),
            (self.object, "write_root_pose_to_sim_mask"),
            (self.object, "write_root_velocity_to_sim_mask"),
            (self.object, "write_root_pose_to_sim_mask_after_graph"),
            (self.object, "write_root_velocity_to_sim_mask_after_graph"),
        )
        for obj, attr in required:
            if not hasattr(obj, attr):
                reasons.append(f"{type(obj).__name__}.{attr} is missing")

        return reasons

    def _check_inhand_fused_reset_compatibility(self) -> str | None:
        """Return why direct fused reset kernels cannot run, ignoring CUDA graph replay stability."""

        blockers = self._get_inhand_fused_reset_blockers()
        if blockers:
            return "; ".join(blockers)

        return self._inhand_warp_views.check_compatibility(groups=("reset",))

    def _refresh_inhand_fused_reset_buffers(self) -> None:
        """Refresh Torch-backed Warp views used by direct fused reset kernels."""

        self._inhand_warp_views.refresh(groups=("reset",))

    def _require_inhand_fused_reset(self, context: str) -> None:
        """Validate direct fused reset kernels and refresh their Torch-backed Warp views."""

        reason = self._check_inhand_fused_reset_compatibility()
        if reason is not None:
            raise RuntimeError(
                f"In-hand {context} requires the fused Warp reset path, but it is not compatible: {reason}."
            )
        self._refresh_inhand_fused_reset_buffers()

    def _launch_inhand_reset_success_snapshot(self, ctx: ResetContext) -> None:
        """Snapshot reset success metrics before residual reset hooks can mutate task state."""

        wp.launch(
            _clear_reset_stats,
            dim=1,
            inputs=[self._reset_count_wp, self._reset_success_count_wp],
            device=self.device,
        )

        env_mask_wp = ctx.reset_mask_wp
        wp.launch(
            _snapshot_inhand_reset_success,
            dim=self.num_envs,
            inputs=[
                env_mask_wp,
                self.cfg.success_count_threshold,
                self._successes_wp,
                self._last_episode_success_wp,
                self._reset_count_wp,
                self._reset_success_count_wp,
            ],
            device=self.device,
        )

    def _launch_inhand_task_reset_prepare(self, ctx: ResetContext) -> None:
        """Prepare task reset buffers after shared reset residual hooks have run."""

        env_mask_wp = ctx.reset_mask_wp
        wp.launch(
            _prepare_inhand_reset,
            dim=self.num_envs,
            inputs=[
                env_mask_wp,
                self.object.data.default_root_pose.warp,
                self._env_origins_wp,
                self.cfg.reset_position_noise,
                self._x_unit_vec_wp,
                self._y_unit_vec_wp,
                self.hand.data.default_joint_pos.warp,
                self.hand.data.default_joint_vel.warp,
                self._lower_limits_wp,
                self._upper_limits_wp,
                self.cfg.reset_dof_pos_noise,
                self.cfg.reset_dof_vel_noise,
                self.num_hand_dofs,
                self._reset_rng_state_wp,
                self._successes_wp,
                self._reset_goal_buf_wp,
                self._episode_length_buf_wp,
                self._goal_rot_wp,
                self._reset_object_pose_wp,
                self._reset_object_velocity_wp,
                self._reset_joint_pos_wp,
                self._reset_joint_vel_wp,
                self._prev_targets_wp,
                self._cur_targets_wp,
                self._hand_dof_targets_wp,
            ],
            device=self.device,
        )

    def _launch_inhand_task_reset_graphable(self, ctx: ResetContext) -> None:
        """Launch graph-capturable task reset kernels.

        This deliberately excludes the shared :class:`DirectRLEnv` residual reset sequence, asset writer methods, and
        ``sim.forward()`` because those methods maintain Python-side state in addition to launching GPU work.
        """

        self._launch_inhand_reset_success_snapshot(ctx)
        self._launch_inhand_task_reset_prepare(ctx)

    def _prepare_inhand_reset_to_sim_capture_state(self) -> None:
        """Force lazy writer output dependencies to be captured explicitly."""

        self.object.data._body_com_acc_w.timestamp = -1.0
        self.hand.data._joint_acc.timestamp = -1.0

    def _launch_inhand_reset_to_sim_graphable(self, ctx: ResetContext) -> None:
        """Launch graph-capturable simulation writes and derived in-hand reset state."""

        env_mask_wp = ctx.reset_mask_wp

        self._write_obj_root_pose_mask(root_pose=self._reset_object_pose_wp, env_mask=env_mask_wp)
        self._write_obj_root_vel_mask(root_velocity=self._reset_object_velocity_wp, env_mask=env_mask_wp)
        self._set_joint_pos_target_mask(target=self._cur_targets_wp, env_mask=env_mask_wp)
        self._write_hand_joint_pos_mask(position=self._reset_joint_pos_wp, env_mask=env_mask_wp)
        self._write_hand_joint_vel_mask(velocity=self._reset_joint_vel_wp, env_mask=env_mask_wp)

        # The default reset path reads lazy body pose properties immediately after writes, which forces FK. The fused
        # core does the FK explicitly and then derives the same intermediate tensors into persistent buffers.
        self.sim.forward()
        self._refresh_inhand_warp_state_inputs()

        wp.launch(
            _compute_inhand_intermediate,
            dim=self.num_envs,
            inputs=[
                self._hand_body_pose_w_wp,
                self._hand_body_vel_w_wp,
                self._finger_bodies_wp,
                self._env_origins_wp,
                self._object_root_pose_w_wp,
                self._object_root_vel_w_wp,
                self.num_fingertips,
                self._graph_fingertip_pos_wp,
                self._graph_fingertip_rot_wp,
                self._graph_fingertip_velocities_wp,
                self._graph_object_pos_wp,
                self._graph_object_rot_wp,
                self._graph_object_velocities_wp,
                self._graph_object_linvel_wp,
                self._graph_object_angvel_wp,
            ],
            device=self.device,
        )

    def _apply_inhand_reset_to_sim_after_graph(self) -> None:
        """Update Python-side lazy state after replaying graph-captured reset writes."""

        self._write_obj_root_pose_mask_after_graph()
        self._write_obj_root_vel_mask_after_graph()
        self._set_joint_pos_target_mask_after_graph()
        self._write_hand_joint_pos_mask_after_graph()
        self._write_hand_joint_vel_mask_after_graph()

    def _apply_inhand_reset_to_sim(self, ctx: ResetContext) -> None:
        """Write prepared reset buffers to simulation and refresh in-hand intermediate state."""

        self._launch_inhand_reset_to_sim_graphable(ctx)
        self._apply_inhand_reset_to_sim_after_graph()

    def _publish_inhand_intermediate_values(self) -> None:
        """Point observation inputs at the persistent buffers produced by fused Warp kernels."""
        self.fingertip_pos = self._graph_fingertip_pos_torch
        self.fingertip_rot = self._graph_fingertip_rot_torch
        self.fingertip_velocities = self._graph_fingertip_velocities_torch
        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch
        self.object_pos = self._graph_object_pos_torch
        self.object_rot = self._graph_object_rot_torch
        self.object_velocities = self._graph_object_velocities_torch
        self.object_linvel = self._graph_object_linvel_torch
        self.object_angvel = self._graph_object_angvel_torch

    def _reset_idx_cuda_graph_impl(self, ctx: ResetContext) -> torch.Tensor | None:
        return self._run_inhand_fused_reset(ctx, use_cuda_graph=True)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._set_joint_pos_target(
            target=self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            # Newton does not implement body_incoming_joint_wrench_b; fall back to zeros.
            try:
                self.fingertip_force_sensors = self.hand.data.body_incoming_joint_wrench_b.torch[:, self.finger_bodies]
            except NotImplementedError:
                self.fingertip_force_sensors = torch.zeros(
                    self.num_envs, len(self.finger_bodies), 6, dtype=torch.float32, device=self.device
                )

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        if not self._inhand_warp_step_enabled:
            return self._get_rewards_torch()

        self._require_inhand_warp_step("reward computation")
        action_reason = self._check_inhand_warp_tensor(
            "actions", self.actions, shape=(self.num_envs, len(self.actuated_dof_indices)), dtype=torch.float32
        )
        if action_reason is not None:
            raise RuntimeError(
                f"In-hand reward computation requires the fused Warp path, but it is not compatible: {action_reason}."
            )
        return self._get_rewards_warp()

    def _get_rewards_warp(self) -> torch.Tensor:
        actions_wp = wp.from_torch(self.actions, dtype=wp.float32)
        wp.launch(
            _compute_inhand_rewards_and_reset_goals,
            dim=self.num_envs,
            inputs=[
                self._reset_buf_step_wp,
                self._reset_goal_buf_wp,
                self._successes_wp,
                self._graph_object_pos_wp,
                self._graph_object_rot_wp,
                self._in_hand_pos_wp,
                self._goal_rot_wp,
                actions_wp,
                self.actions.shape[1],
                self.cfg.dist_reward_scale,
                self.cfg.rot_reward_scale,
                self.cfg.rot_eps,
                self.cfg.action_penalty_scale,
                self.cfg.success_tolerance,
                self.cfg.reach_goal_bonus,
                self.cfg.fall_dist,
                self.cfg.fall_penalty,
                self._goal_reset_rng_state_wp,
                self._x_unit_vec_wp,
                self._y_unit_vec_wp,
                self._reward_buf_wp,
                self._reward_reset_count_terms_wp,
                self._reward_finished_success_terms_wp,
                self._reward_goal_reset_terms_wp,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_inhand_rewards,
            dim=1,
            inputs=[
                self._reward_reset_count_terms_wp,
                self._reward_finished_success_terms_wp,
                self.num_envs,
                self.cfg.av_factor,
                self._reward_goal_reset_terms_wp,
                self._consecutive_successes_wp,
                self._reward_goal_reset_count_wp,
            ],
            device=self.device,
        )
        if self._should_sync_goal_markers() and int(self._reward_goal_reset_count_torch.item()) > 0:
            self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        return self.reward_buf

    def _get_rewards_torch(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._inhand_warp_step_enabled:
            return self._get_dones_torch()

        self._require_inhand_warp_step("done computation")
        wp.launch(
            _compute_inhand_intermediate_and_dones,
            dim=self.num_envs,
            inputs=[
                self._hand_body_pose_w_wp,
                self._hand_body_vel_w_wp,
                self._finger_bodies_wp,
                self._env_origins_wp,
                self._object_root_pose_w_wp,
                self._object_root_vel_w_wp,
                self.num_fingertips,
                self._in_hand_pos_wp,
                self._goal_rot_wp,
                self._successes_wp,
                self._episode_length_buf_wp,
                self.max_episode_length - 1,
                self.cfg.fall_dist,
                self.cfg.max_consecutive_success,
                self.cfg.success_tolerance,
                self._graph_fingertip_pos_wp,
                self._graph_fingertip_rot_wp,
                self._graph_fingertip_velocities_wp,
                self._graph_object_pos_wp,
                self._graph_object_rot_wp,
                self._graph_object_velocities_wp,
                self._graph_object_linvel_wp,
                self._graph_object_angvel_wp,
                self._reset_terminated_wp,
                self._reset_time_outs_wp,
            ],
            device=self.device,
        )
        self._publish_inhand_intermediate_values()
        return self.reset_terminated, self.reset_time_outs

    def _get_dones_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        goal_dist = torch.linalg.norm(self.object_pos - self.in_hand_pos, ord=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf[:] = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        if not self._inhand_fused_reset_enabled:
            self._reset_idx_torch(env_ids)
            return

        self._require_inhand_warp_step("reset")
        self._require_inhand_fused_reset("reset")
        env_ids = self._reset_env_ids_tensor(env_ids)
        env_ids_long = env_ids.to(dtype=torch.long)

        self._reset_env_mask.zero_()
        self._reset_env_mask[env_ids_long] = True
        self._run_inhand_fused_reset(
            ResetContext(env_ids=env_ids, reset_mask_wp=self._reset_env_mask_wp),
            use_cuda_graph=False,
        )

    def _reset_env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if torch.is_tensor(env_ids):
            return env_ids.to(device=self.device, dtype=torch.int32)
        return torch.tensor(env_ids, dtype=torch.int32, device=self.device)

    def _materialize_reset_context_env_ids(self, ctx: ResetContext) -> tuple[ResetContext, bool]:
        """Return a reset context with concrete env ids for residual Python reset hooks."""

        if ctx.env_ids is not None:
            return ctx, False

        nvtx_range_push("env.step:reset_buf.nonzero")
        env_ids = ctx.selection.materialize_env_ids(device=self.device)
        nvtx_range_pop()
        return ctx.with_env_ids(env_ids), True

    def _run_inhand_fused_reset(self, ctx: ResetContext, *, use_cuda_graph: bool) -> torch.Tensor | None:
        if ctx.env_ids is None and not use_cuda_graph:
            raise ValueError("Fused in-hand reset requires concrete env_ids.")

        if use_cuda_graph:

            def after_common_reset(graph_ctx: ResetContext) -> tuple[ResetContext, bool]:
                graph_ctx, env_ids_from_reset_mask = self._materialize_reset_context_env_ids(graph_ctx)
                if env_ids_from_reset_mask and len(graph_ctx.env_ids) == 0:
                    return graph_ctx, False
                self._reset_idx_common_after_graph(graph_ctx, reset_episode_lengths=False)
                return graph_ctx, True

            def after_apply_reset(graph_ctx: ResetContext) -> tuple[ResetContext, bool]:
                self._apply_inhand_reset_to_sim_after_graph()
                return graph_ctx, True

            phases = (
                ResetGraphPhase(
                    graph_attr="_reset_common_cuda_graph",
                    name="common reset",
                    launch_fn=lambda graph_ctx: self._reset_idx_common_graphable(
                        graph_ctx, reset_episode_lengths=False
                    ),
                    between_hook=after_common_reset,
                ),
                ResetGraphPhase(
                    graph_attr="_reset_cuda_graph",
                    name="task reset",
                    launch_fn=self._launch_inhand_task_reset_graphable,
                ),
                ResetGraphPhase(
                    graph_attr="_reset_apply_cuda_graph",
                    name="apply reset",
                    launch_fn=self._launch_inhand_reset_to_sim_graphable,
                    prepare_capture=self._prepare_inhand_reset_to_sim_capture_state,
                    between_hook=after_apply_reset,
                ),
            )
            replay_result = self._replay_reset_cuda_graph_phases(ctx, phases)
            if replay_result is None:
                return None
            ctx, replay_completed = replay_result
            if not replay_completed:
                return ctx.env_ids
        else:
            self._launch_inhand_reset_success_snapshot(ctx)
            self._reset_idx_common_graphable(ctx, reset_episode_lengths=False)
            self._reset_idx_common_after_graph(ctx, reset_episode_lengths=False)
            self._launch_inhand_task_reset_prepare(ctx)
            self._apply_inhand_reset_to_sim(ctx)

        self._finish_inhand_fused_reset(ctx)
        return ctx.env_ids

    def _finish_inhand_fused_reset(self, ctx: ResetContext) -> None:
        if ctx.env_ids is None:
            raise ValueError("In-hand fused reset requires concrete env_ids.")
        self._publish_inhand_intermediate_values()
        self._publish_inhand_reset_metrics()

    def _publish_inhand_reset_metrics(self) -> None:
        reset_count = int(self._reset_count_torch.item())
        if reset_count > 0:
            success_count = float(self._reset_success_count_torch.item())
            self.extras.setdefault("log", {})["Metrics/success_rate"] = success_count / float(reset_count)
        if self._should_sync_goal_markers() and reset_count > 0:
            self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

    def _reset_idx_torch(self, env_ids: Sequence[int]) -> None:
        """Fallback reset path preserving the pre-fused in-hand semantics."""

        env_ids = self._reset_env_ids_tensor(env_ids)
        env_ids_long = env_ids.to(dtype=torch.long)

        self._last_episode_success[env_ids_long] = self.successes[env_ids_long] >= self.cfg.success_count_threshold
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._last_episode_success[env_ids_long].float().mean().item()
        )

        super()._reset_idx(env_ids)

        self._reset_target_pose(env_ids)

        object_default_pose = self.object.data.default_root_pose.torch.clone()[env_ids_long]
        object_default_vel = self.object.data.default_root_vel.torch.clone()[env_ids_long]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_pose[:, 0:3] = (
            object_default_pose[:, 0:3]
            + self.cfg.reset_position_noise * pos_noise
            + self.scene.env_origins[env_ids_long]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids_long], self.y_unit_tensor[env_ids_long]
        )

        object_default_vel[:] = 0.0
        self._write_obj_root_pose(root_pose=object_default_pose, env_ids=env_ids)
        self._write_obj_root_vel(root_velocity=object_default_vel, env_ids=env_ids)

        delta_max = self.hand_dof_upper_limits[env_ids_long] - self.hand.data.default_joint_pos.torch[env_ids_long]
        delta_min = self.hand_dof_lower_limits[env_ids_long] - self.hand.data.default_joint_pos.torch[env_ids_long]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos.torch[env_ids_long] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids_long] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids_long] = dof_pos
        self.cur_targets[env_ids_long] = dof_pos
        self.hand_dof_targets[env_ids_long] = dof_pos

        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids)

        self.successes[env_ids_long] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids: Sequence[int]) -> None:
        env_ids = self._reset_env_ids_tensor(env_ids)
        env_ids_long = env_ids.to(dtype=torch.long)

        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids_long], self.y_unit_tensor[env_ids_long]
        )

        self.goal_rot[env_ids_long] = new_rot
        if self._should_sync_goal_markers():
            self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

        self.reset_goal_buf[env_ids_long] = 0

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w.torch[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w.torch[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w.torch[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos.torch
        self.hand_dof_vel = self.hand.data.joint_vel.torch

        # data for object
        self.object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w.torch
        self.object_velocities = self.object.data.root_vel_w.torch
        self.object_linvel = self.object.data.root_lin_vel_w.torch
        self.object_angvel = self.object.data.root_ang_vel_w.torch

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.linalg.norm(quat_diff[:, 0:3], ord=2, dim=-1), max=1.0))


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    goal_dist = torch.linalg.norm(object_pos - target_pos, ord=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1)
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
