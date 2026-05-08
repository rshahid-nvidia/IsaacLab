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

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    block_dim: int = 0
    """Thread block size for the Newton Warp renderer megakernel.

    A value of ``0`` preserves Newton/Warp's default launch configuration. Positive values are applied by IsaacLab
    through a scoped monkey patch around the Newton renderer update call, so this knob does not require a patched
    Newton install.
    """

    render_order: int = 0
    """Newton raytracer traversal order.

    This is passed through to ``newton.sensors.SensorTiledCamera.RenderConfig.render_order``. Current Newton values are
    ``0`` for ``PIXEL_PRIORITY``, ``1`` for ``VIEW_PRIORITY``, and ``2`` for ``TILED``.
    """

    tile_width: int = 16
    """Tile width in pixels when ``render_order`` is ``TILED``."""

    tile_height: int = 8
    """Tile height in pixels when ``render_order`` is ``TILED``."""

    create_default_light: bool = True
    """Create a default directional light source in the scene."""

    colorize_instance_segmentation: bool = True
    """Expose ``instance_segmentation_fast`` as ``(N, H, W, 4) uint8`` if True, else ``(N, H, W, 1) int32``."""
