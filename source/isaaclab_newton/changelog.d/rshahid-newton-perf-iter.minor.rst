Added
^^^^^

* Added split reset support for Newton
  :class:`~isaaclab_newton.assets.Articulation`,
  :class:`~isaaclab_newton.assets.RigidObject`,
  :class:`~isaaclab_newton.assets.RigidObjectCollection`, and Newton sensor reset paths.
* Added mask-native Newton reset writer hooks for graph-capturable in-hand manipulation
  reset paths.
* Added :meth:`isaaclab_newton.renderers.NewtonWarpRenderer.supports_camera_reset_graphable`
  so Newton renderer camera pose reset can participate in task-level reset CUDA graph replay.

Changed
^^^^^^^

* Changed Newton reset selector handling so ``env_mask`` takes precedence over ``env_ids``
  when both are provided. Prefer passing one selector source to avoid ambiguous resets.
