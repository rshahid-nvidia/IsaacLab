# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.cuda_graph import (
    ResetContext,
    capture_cuda_graph_relaxed,
    launch_cuda_graph_on_current_torch_stream,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

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
    success_count_threshold: wp.int32,
    rng_state: wp.array(dtype=wp.uint32),
    successes: wp.array(dtype=wp.float32),
    episode_length_buf: wp.array(dtype=wp.int64),
    goal_rot: wp.array(dtype=wp.quatf),
    object_pose_out: wp.array(dtype=wp.transformf),
    object_velocity_out: wp.array(dtype=wp.spatial_vectorf),
    joint_pos_out: wp.array2d(dtype=wp.float32),
    joint_vel_out: wp.array2d(dtype=wp.float32),
    prev_targets: wp.array2d(dtype=wp.float32),
    cur_targets: wp.array2d(dtype=wp.float32),
    hand_dof_targets: wp.array2d(dtype=wp.float32),
    reset_count: wp.array(dtype=wp.int32),
    reset_success_count: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()
    if not env_mask[env_id]:
        return

    was_success = successes[env_id] >= wp.float32(success_count_threshold)
    wp.atomic_add(reset_count, 0, wp.int32(1))
    if was_success:
        wp.atomic_add(reset_success_count, 0, wp.int32(1))

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
        reset_time_outs[env_id] = (
            episode_length_buf[env_id] >= max_episode_length_minus_one
            or successes[env_id] >= wp.float32(max_consecutive_success)
        )
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

        # bind backend-optimal joint target method (Newton prefers mask-based, PhysX prefers indexed)
        use_mask = "newton" in self.sim.physics_manager.__name__.lower()
        if use_mask:
            self._set_joint_pos_target = self.hand.set_joint_position_target
        else:
            self._set_joint_pos_target = self.hand.set_joint_position_target_index

        if not self._inhand_warp_state_buffers_available():
            raise RuntimeError(
                "In-hand manipulation requires the fused Warp step path; "
                "required Warp state buffers are not available."
            )
        self._setup_inhand_warp_step_buffers()

        fused_reset_blockers = self._get_inhand_fused_reset_blockers()
        if fused_reset_blockers:
            raise RuntimeError(
                "In-hand manipulation requires the fused Warp reset path; "
                f"{'; '.join(fused_reset_blockers)}."
            )
        self._setup_inhand_fused_reset_buffers()

        self._configure_reset_cuda_graph(path_name="In-hand reset CUDA graph path")

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
        self._episode_length_buf_wp = wp.from_torch(self.episode_length_buf, dtype=wp.int64)
        self._successes_wp = wp.from_torch(self.successes, dtype=wp.float32)
        self._consecutive_successes_wp = wp.from_torch(self.consecutive_successes, dtype=wp.float32)
        self._goal_rot_wp = wp.from_torch(self.goal_rot, dtype=wp.quatf)
        self._in_hand_pos_wp = wp.from_torch(self.in_hand_pos, dtype=wp.vec3f)
        self._env_origins_wp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)
        self._reset_terminated_wp = wp.from_torch(self.reset_terminated, dtype=wp.bool)
        self._reset_time_outs_wp = wp.from_torch(self.reset_time_outs, dtype=wp.bool)
        self._reset_buf_step_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
        self._reset_goal_buf_wp = wp.from_torch(self.reset_goal_buf, dtype=wp.bool)
        self._finger_bodies_wp = wp.array(self.finger_bodies, dtype=wp.int32, device=self.device)

        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._reward_buf_wp = wp.from_torch(self.reward_buf, dtype=wp.float32)
        self._reward_reset_count_terms_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reward_finished_success_terms_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._reward_goal_reset_terms_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._reward_goal_reset_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reward_goal_reset_count_torch = wp.to_torch(self._reward_goal_reset_count_wp)

        self._x_unit_vec_wp = wp.vec3f(1.0, 0.0, 0.0)
        self._y_unit_vec_wp = wp.vec3f(0.0, 1.0, 0.0)
        self._goal_reset_rng_state_wp = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        goal_reset_seed = 0 if self.cfg.seed is None else int(self.cfg.seed) + 104729
        wp.launch(
            _initialize_reset_rng,
            dim=self.num_envs,
            inputs=[goal_reset_seed, self._goal_reset_rng_state_wp],
            device=self.device,
        )

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

    def _refresh_inhand_warp_state_inputs(self) -> None:
        self._hand_body_pose_w_wp = self.hand.data.body_link_pose_w.warp
        self._hand_body_vel_w_wp = self.hand.data.body_com_vel_w.warp
        self._object_root_pose_w_wp = self.object.data.root_link_pose_w.warp
        self._object_root_vel_w_wp = self.object.data.root_com_vel_w.warp

    def _check_inhand_warp_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> str | None:
        if not torch.is_tensor(tensor):
            return f"{name} is not a torch.Tensor"
        if tuple(tensor.shape) != shape:
            return f"{name} shape changed from {shape} to {tuple(tensor.shape)}"
        if tensor.dtype != dtype:
            return f"{name} dtype changed from {dtype} to {tensor.dtype}"
        if tensor.device != torch.device(self.device):
            return f"{name} device changed from {self.device} to {tensor.device}"
        return None

    def _check_inhand_warp_step_compatibility(self) -> str | None:
        """Return why direct Warp step kernels cannot run, ignoring CUDA graph replay stability."""

        if not self._inhand_warp_state_buffers_available():
            return "required Warp state buffers are not available"

        tensor_specs = (
            ("episode_length_buf", self.episode_length_buf, (self.num_envs,), torch.int64),
            ("reset_terminated", self.reset_terminated, (self.num_envs,), torch.bool),
            ("reset_time_outs", self.reset_time_outs, (self.num_envs,), torch.bool),
            ("reset_buf", self.reset_buf, (self.num_envs,), torch.bool),
            ("reward_buf", self.reward_buf, (self.num_envs,), torch.float32),
            ("reset_goal_buf", self.reset_goal_buf, (self.num_envs,), torch.bool),
            ("successes", self.successes, (self.num_envs,), torch.float32),
            ("consecutive_successes", self.consecutive_successes, (1,), torch.float32),
            ("goal_rot", self.goal_rot, (self.num_envs, 4), torch.float32),
            ("in_hand_pos", self.in_hand_pos, (self.num_envs, 3), torch.float32),
            ("scene.env_origins", self.scene.env_origins, (self.num_envs, 3), torch.float32),
            (
                "hand.body_link_pose_w",
                self.hand.data.body_link_pose_w.torch,
                (self.num_envs, len(self.hand.body_names), 7),
                torch.float32,
            ),
            (
                "hand.body_com_vel_w",
                self.hand.data.body_com_vel_w.torch,
                (self.num_envs, len(self.hand.body_names), 6),
                torch.float32,
            ),
            ("object.root_link_pose_w", self.object.data.root_link_pose_w.torch, (self.num_envs, 7), torch.float32),
            ("object.root_com_vel_w", self.object.data.root_com_vel_w.torch, (self.num_envs, 6), torch.float32),
        )
        for name, tensor, shape, dtype in tensor_specs:
            reason = self._check_inhand_warp_tensor(name, tensor, shape=shape, dtype=dtype)
            if reason is not None:
                return reason
        return None

    def _refresh_inhand_warp_step_buffers(self) -> None:
        """Refresh Warp views for direct kernels after possible Torch tensor rebinding."""

        self._episode_length_buf_wp = wp.from_torch(self.episode_length_buf, dtype=wp.int64)
        self._successes_wp = wp.from_torch(self.successes, dtype=wp.float32)
        self._consecutive_successes_wp = wp.from_torch(self.consecutive_successes, dtype=wp.float32)
        self._goal_rot_wp = wp.from_torch(self.goal_rot, dtype=wp.quatf)
        self._in_hand_pos_wp = wp.from_torch(self.in_hand_pos, dtype=wp.vec3f)
        self._env_origins_wp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)
        self._reset_terminated_wp = wp.from_torch(self.reset_terminated, dtype=wp.bool)
        self._reset_time_outs_wp = wp.from_torch(self.reset_time_outs, dtype=wp.bool)
        self._reset_buf_step_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
        self._reset_goal_buf_wp = wp.from_torch(self.reset_goal_buf, dtype=wp.bool)
        self._reward_buf_wp = wp.from_torch(self.reward_buf, dtype=wp.float32)
        self._refresh_inhand_warp_state_inputs()

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
        self._reset_env_mask_wp = wp.from_torch(self._reset_env_mask, dtype=wp.bool)
        self._lower_limits_wp = wp.from_torch(self.hand_dof_lower_limits, dtype=wp.float32)
        self._upper_limits_wp = wp.from_torch(self.hand_dof_upper_limits, dtype=wp.float32)
        self._prev_targets_wp = wp.from_torch(self.prev_targets, dtype=wp.float32)
        self._cur_targets_wp = wp.from_torch(self.cur_targets, dtype=wp.float32)
        self._hand_dof_targets_wp = wp.from_torch(self.hand_dof_targets, dtype=wp.float32)

        self._reset_rng_state_wp = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        reset_seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        wp.launch(
            _initialize_reset_rng,
            dim=self.num_envs,
            inputs=[reset_seed, self._reset_rng_state_wp],
            device=self.device,
        )

        self._reset_empty_mask_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._reset_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reset_success_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._reset_count_torch = wp.to_torch(self._reset_count_wp)
        self._reset_success_count_torch = wp.to_torch(self._reset_success_count_wp)

        self._reset_object_pose_wp = wp.zeros(self.num_envs, dtype=wp.transformf, device=self.device)
        self._reset_object_velocity_wp = wp.zeros(self.num_envs, dtype=wp.spatial_vectorf, device=self.device)
        self._reset_joint_pos_wp = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self._reset_joint_vel_wp = wp.zeros((self.num_envs, self.num_hand_dofs), dtype=wp.float32, device=self.device)
        self._inhand_fused_reset_buffers_ready = True

    def _setup_reset_cuda_graph_buffers(self) -> None:
        self._setup_inhand_fused_reset_buffers()
        self._reset_cuda_graph_mask_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)

    def _warmup_reset_cuda_graph(self) -> None:
        self._launch_inhand_reset_prepare(ResetContext(env_ids=None, reset_mask_wp=self._reset_empty_mask_wp))

    def _reset_cuda_graph_tensors(self) -> dict[str, torch.Tensor]:
        """Return tensors whose storage and metadata must remain stable for reset graph replay."""

        tensors = super()._reset_cuda_graph_tensors()
        tensors.update(
            {
                "episode_length_buf": self.episode_length_buf,
                "successes": self.successes,
                "goal_rot": self.goal_rot,
                "scene.env_origins": self.scene.env_origins,
                "hand_dof_lower_limits": self.hand_dof_lower_limits,
                "hand_dof_upper_limits": self.hand_dof_upper_limits,
                "prev_targets": self.prev_targets,
                "cur_targets": self.cur_targets,
                "hand_dof_targets": self.hand_dof_targets,
                "object.default_root_pose": self.object.data.default_root_pose.torch,
                "hand.default_joint_pos": self.hand.data.default_joint_pos.torch,
                "hand.default_joint_vel": self.hand.data.default_joint_vel.torch,
            }
        )
        return tensors

    def _reset_cuda_graph_constants(self) -> dict[str, float | int]:
        """Return scalar reset parameters baked into the captured reset graph."""

        return {
            "reset_position_noise": float(self.cfg.reset_position_noise),
            "reset_dof_pos_noise": float(self.cfg.reset_dof_pos_noise),
            "reset_dof_vel_noise": float(self.cfg.reset_dof_vel_noise),
            "success_count_threshold": int(self.cfg.success_count_threshold),
            "num_hand_dofs": int(self.num_hand_dofs),
        }

    def _get_reset_cuda_graph_blockers(self) -> list[str]:
        reasons = super()._get_reset_cuda_graph_blockers()
        reasons.extend(self._get_inhand_fused_reset_blockers())
        if "newton" not in self.sim.physics_manager.__name__.lower():
            reasons.append("physics backend is not Newton")
        return reasons

    def _get_inhand_fused_reset_blockers(self) -> list[str]:
        reasons: list[str] = []
        required = (
            (self.hand, "set_joint_position_target_mask"),
            (self.hand, "write_joint_position_to_sim_mask"),
            (self.hand, "write_joint_velocity_to_sim_mask"),
            (self.object, "write_root_pose_to_sim_mask"),
            (self.object, "write_root_velocity_to_sim_mask"),
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

        tensor_specs = (
            ("reset_env_mask", self._reset_env_mask, (self.num_envs,), torch.bool),
            ("episode_length_buf", self.episode_length_buf, (self.num_envs,), torch.int64),
            ("successes", self.successes, (self.num_envs,), torch.float32),
            ("goal_rot", self.goal_rot, (self.num_envs, 4), torch.float32),
            ("scene.env_origins", self.scene.env_origins, (self.num_envs, 3), torch.float32),
            ("hand_dof_lower_limits", self.hand_dof_lower_limits, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("hand_dof_upper_limits", self.hand_dof_upper_limits, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("prev_targets", self.prev_targets, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("cur_targets", self.cur_targets, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("hand_dof_targets", self.hand_dof_targets, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("object.default_root_pose", self.object.data.default_root_pose.torch, (self.num_envs, 7), torch.float32),
            ("hand.default_joint_pos", self.hand.data.default_joint_pos.torch, (self.num_envs, self.num_hand_dofs), torch.float32),
            ("hand.default_joint_vel", self.hand.data.default_joint_vel.torch, (self.num_envs, self.num_hand_dofs), torch.float32),
        )
        for name, tensor, shape, dtype in tensor_specs:
            reason = self._check_inhand_warp_tensor(name, tensor, shape=shape, dtype=dtype)
            if reason is not None:
                return reason
        return None

    def _refresh_inhand_fused_reset_buffers(self) -> None:
        """Refresh Torch-backed Warp views used by direct fused reset kernels."""

        self._reset_env_mask_wp = wp.from_torch(self._reset_env_mask, dtype=wp.bool)
        self._lower_limits_wp = wp.from_torch(self.hand_dof_lower_limits, dtype=wp.float32)
        self._upper_limits_wp = wp.from_torch(self.hand_dof_upper_limits, dtype=wp.float32)
        self._prev_targets_wp = wp.from_torch(self.prev_targets, dtype=wp.float32)
        self._cur_targets_wp = wp.from_torch(self.cur_targets, dtype=wp.float32)
        self._hand_dof_targets_wp = wp.from_torch(self.hand_dof_targets, dtype=wp.float32)

    def _require_inhand_fused_reset(self, context: str) -> None:
        """Validate direct fused reset kernels and refresh their Torch-backed Warp views."""

        reason = self._check_inhand_fused_reset_compatibility()
        if reason is not None:
            raise RuntimeError(f"In-hand {context} requires the fused Warp reset path, but it is not compatible: {reason}.")
        self._refresh_inhand_fused_reset_buffers()

    def _launch_inhand_reset_prepare(self, ctx: ResetContext) -> None:
        """Launch pure reset preparation kernels.

        This is the only reset subset captured into the optional CUDA graph. It deliberately excludes the shared
        :class:`DirectRLEnv` reset sequence, asset writer methods, and ``sim.forward()`` because those methods maintain
        Python-side state in addition to launching GPU work.
        """
        wp.launch(
            _clear_reset_stats,
            dim=1,
            inputs=[self._reset_count_wp, self._reset_success_count_wp],
            device=self.device,
        )

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
                self.cfg.success_count_threshold,
                self._reset_rng_state_wp,
                self._successes_wp,
                self._episode_length_buf_wp,
                self._goal_rot_wp,
                self._reset_object_pose_wp,
                self._reset_object_velocity_wp,
                self._reset_joint_pos_wp,
                self._reset_joint_vel_wp,
                self._prev_targets_wp,
                self._cur_targets_wp,
                self._hand_dof_targets_wp,
                self._reset_count_wp,
                self._reset_success_count_wp,
            ],
            device=self.device,
        )

    def _apply_inhand_reset_to_sim(self, ctx: ResetContext) -> None:
        """Write prepared reset buffers to simulation and refresh in-hand intermediate state.

        This intentionally stays outside CUDA graph replay because the asset writer APIs also update Python-side lazy
        buffer timestamps and FK invalidation state.
        """
        env_mask_wp = ctx.reset_mask_wp

        self.object.write_root_pose_to_sim_mask(root_pose=self._reset_object_pose_wp, env_mask=env_mask_wp)
        self.object.write_root_velocity_to_sim_mask(root_velocity=self._reset_object_velocity_wp, env_mask=env_mask_wp)
        self.hand.set_joint_position_target_mask(target=self._cur_targets_wp, env_mask=env_mask_wp)
        self.hand.write_joint_position_to_sim_mask(position=self._reset_joint_pos_wp, env_mask=env_mask_wp)
        self.hand.write_joint_velocity_to_sim_mask(velocity=self._reset_joint_vel_wp, env_mask=env_mask_wp)

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

    def _capture_inhand_reset_prepare_cuda_graph(self):
        """Capture the pure reset preparation kernels on a non-blocking stream in relaxed mode."""
        ctx = ResetContext(env_ids=None, reset_mask_wp=self._reset_cuda_graph_mask_wp)
        return capture_cuda_graph_relaxed(
            self.device, lambda: self._launch_inhand_reset_prepare(ctx)
        )

    def _reset_idx_cuda_graph_impl(self, ctx: ResetContext) -> bool:
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
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes[0]
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _reset_idx(self, env_ids: Sequence[int]):
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

    def _run_inhand_fused_reset(self, ctx: ResetContext, *, use_cuda_graph: bool) -> bool:
        if ctx.env_ids is None:
            if use_cuda_graph:
                return self._disable_reset_cuda_graph("fused reset requires concrete env_ids")
            raise ValueError("Fused in-hand reset requires concrete env_ids.")

        env_ids_long = ctx.env_ids.to(dtype=torch.long)
        self._last_episode_success[env_ids_long] = self.successes[env_ids_long] >= self.cfg.success_count_threshold

        self._reset_idx_common(ctx.env_ids, reset_episode_lengths=False)

        if use_cuda_graph:
            if self._reset_cuda_graph is None:
                try:
                    self._reset_cuda_graph = self._capture_inhand_reset_prepare_cuda_graph()
                except Exception as exc:
                    reason = f"capture failed: {exc}"
                    return self._disable_reset_cuda_graph(reason)
            replay_stream = launch_cuda_graph_on_current_torch_stream(self.device, self._reset_cuda_graph)
            with wp.ScopedStream(replay_stream, sync_enter=False):
                self._apply_inhand_reset_to_sim(ctx)
        else:
            self._launch_inhand_reset_prepare(ctx)
            self._apply_inhand_reset_to_sim(ctx)

        self._finish_inhand_fused_reset(ctx)
        return True

    def _finish_inhand_fused_reset(self, ctx: ResetContext) -> None:
        if ctx.env_ids is None:
            raise ValueError("In-hand fused reset requires concrete env_ids.")
        self.reset_goal_buf[ctx.env_ids.to(dtype=torch.long)] = False
        self._publish_inhand_intermediate_values()
        self._publish_inhand_reset_metrics()

    def _publish_inhand_reset_metrics(self) -> None:
        reset_count = int(self._reset_count_torch.item())
        if reset_count > 0:
            success_count = float(self._reset_success_count_torch.item())
            self.extras.setdefault("log", {})["Metrics/success_rate"] = success_count / float(reset_count)
        if self._should_sync_goal_markers() and reset_count > 0:
            self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

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
