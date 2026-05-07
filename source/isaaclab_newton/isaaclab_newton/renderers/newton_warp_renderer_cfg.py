# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for Newton Warp Renderer."""

    renderer_type: str = "newton_warp"
    """Type identifier for Newton Warp renderer."""

    enable_textures: bool = True
    """Enable texture-mapped rendering for meshes."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights."""

    enable_ambient_lighting: bool = True
    """Enable ambient lighting for the scene."""

    enable_global_world: bool = True
    """Include Newton shapes that belong to the global world in ray traversal."""

    enable_particles: bool = True
    """Enable Newton particle rendering."""

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    render_order: int = 0
    """Newton Warp renderer traversal order.

    ``0`` is pixel-priority, ``1`` is view-priority, and ``2`` is tiled rendering.
    """

    tile_width: int = 16
    """Tile width in pixels when ``render_order`` selects tiled rendering."""

    tile_height: int = 8
    """Tile height in pixels when ``render_order`` selects tiled rendering."""

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    block_dim: int = 128
    """Thread block size used to launch the Newton Warp renderer megakernel.

    A value of ``0`` preserves Warp's default launch configuration when using Newton versions that support this
    option.
    """

    mesh_bvh_constructor: str | None = None
    """Optional BVH constructor for Newton renderer-owned mesh handles.

    Set to ``"cubql"`` to use a cuBQL-enabled Warp build for mesh ray traversal. ``None`` preserves Newton's default
    behavior and reuses the simulation mesh handles.
    """

    create_default_light: bool = True
    """Create a default directional light source in the scene."""

    colorize_instance_segmentation: bool = True
    """Expose ``instance_segmentation_fast`` as ``(N, H, W, 4) uint8`` if True, else ``(N, H, W, 1) int32``."""
