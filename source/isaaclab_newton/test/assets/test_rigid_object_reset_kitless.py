# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless Newton RigidObject reset tests."""

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def _newton_sim_context(device: str):
    sim_cfg = SimulationCfg(device=device, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    return build_simulation_context(device=device, sim_cfg=sim_cfg, add_ground_plane=True, auto_add_lighting=True)


def _generate_cubes_scene(num_cubes: int, device: str) -> RigidObject:
    origins = torch.tensor([(i * 1.0, 0.0, 1.0) for i in range(num_cubes)], device=device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=origin)

    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cube_object_cfg)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_rigid_object_wrench_buffers_with_env_mask_kitless(device):
    """env_mask reset should clear only selected external wrench buffers."""
    with _newton_sim_context(device) as sim:
        cube_object = _generate_cubes_scene(num_cubes=4, device=device)
        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")
        forces = torch.ones(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        torques = 2.0 * torch.ones_like(forces)
        env_mask_torch = torch.tensor([True, False, True, False], device=sim.device)
        env_mask_wp = wp.from_torch(env_mask_torch, dtype=wp.bool)

        cube_object.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torques, body_ids=body_ids
        )
        cube_object.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torques, body_ids=body_ids
        )

        cube_object.reset(env_mask=env_mask_wp)

        for composer in (cube_object.instantaneous_wrench_composer, cube_object.permanent_wrench_composer):
            out_force = composer.out_force_b.torch
            out_torque = composer.out_torque_b.torch
            torch.testing.assert_close(
                out_force[env_mask_torch],
                torch.zeros_like(out_force[env_mask_torch]),
            )
            torch.testing.assert_close(
                out_torque[env_mask_torch],
                torch.zeros_like(out_torque[env_mask_torch]),
            )
            torch.testing.assert_close(out_force[~env_mask_torch], forces[~env_mask_torch])
            torch.testing.assert_close(out_torque[~env_mask_torch], torques[~env_mask_torch])
