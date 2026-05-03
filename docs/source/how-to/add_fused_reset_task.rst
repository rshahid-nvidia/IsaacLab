Adding a Fused Reset Task
=========================

.. currentmodule:: isaaclab

Large direct RL tasks can spend more time launching many small CUDA kernels than executing the kernels themselves.
For Newton-backed tasks with simple per-environment reset, done, and reward logic, fused Warp kernels and optional CUDA
graph replay can reduce that launch overhead while preserving the existing semantic reset path.

When to Use This Path
---------------------

Use a fused or graph-replayed reset path when all of the following are true:

* The task runs on the Newton backend with CUDA tensors.
* The workload uses many environments and reset/done/reward kernels are launch-overhead dominated.
* The reset math can be expressed as Warp kernels or graph-capturable Torch operations over stable
  per-environment buffers.
* Any simulator or renderer writes inside graph replay consume stable buffers and have been validated under capture.
* Any Python-side reset state can run before or after the graphable kernels without reading freshly reset GPU state
  from another entity out of order.

Do not add a Warp kernel only to avoid thinking about capture. Prefer graph-captured Torch operations when they preserve
semantics and avoid dynamic selector shapes. PyTorch's CUDA graph integration can handle allocator-owned temporaries, but
the external tensors read or written across replay still need stable storage. Under Isaac Lab's relaxed Warp/CUDA capture
helper, replay-time Torch RNG does not advance today; reset logic that must randomize on every replay should use a
task-owned RNG kernel or stay outside capture until that capture path changes. The Torch path remains the semantic
fallback for unsupported backends, CPU runs, and tests that compare fused kernels against the reference behavior.

Reset Phases
------------

The reusable environment hooks live on :class:`~isaaclab.envs.DirectRLEnv`. A task that supports CUDA graph replay must
call ``_configure_reset_cuda_graph(...)`` after the task state needed by its graph hooks has been initialized and usually
implements these methods:

* ``_setup_reset_cuda_graph_buffers()`` allocates stable tensors and Warp arrays used by captured reset phases.
* ``_warmup_reset_cuda_graph()`` launches each graphable kernel once so lazy initialization does not happen during
  capture.
* ``_reset_cuda_graph_tensors()`` returns tensors and Warp arrays whose storage and metadata must remain stable.
* ``_reset_cuda_graph_constants()`` returns scalar values baked into the captured graph.
* ``_reset_idx_cuda_graph_impl(ctx)`` declares the ordered
  :class:`~isaaclab.envs.cuda_graph.ResetGraphPhase` sequence and calls
  :meth:`~isaaclab.envs.DirectRLEnv._replay_reset_cuda_graph_phases`.

The task still owns the fused kernels. The base environment owns mode handling, replay guard checks, capture, replay,
recapture, and fallback behavior.

The split scene/entity APIs do not automatically enable CUDA graph replay for every task. They are building blocks that
become captured only when a task-level implementation calls the reset graph replay hooks.

The current in-hand implementation uses two ordered graph phases:

1. A common reset graph for graphable pieces of :class:`~isaaclab.envs.DirectRLEnv`, mainly
   :meth:`~isaaclab.scene.InteractiveScene.reset_graphable`.
2. A task/apply graph that combines task reset preparation, task-owned buffer updates, simulator writes, and the
   graphable simulation-forward work needed by those writes.

The task/apply graph used to be split into separate task and apply phases. Keep them merged when no Python residual work
must run between them: this preserves the reset ordering after the common residual hook while saving one CUDA graph
launch. Split them back into separate :class:`~isaaclab.envs.cuda_graph.ResetGraphPhase` objects only when correctness
requires a residual hook between task-state reset and simulator writes.

Selectors
---------

Graphable reset code should consume :class:`~isaaclab.envs.cuda_graph.ResetContext`. It contains:

* ``ctx.reset_mask_wp``: the stable Warp boolean mask used by CUDA graph capture and replay.
* ``ctx.env_ids``: concrete environment ids, available only after a residual path materializes them.
* ``ctx.selection``: a :class:`~isaaclab.utils.reset.ResetSelection` with one selector rule: when ``env_mask`` is
  present, it is the source of truth.

Keep environment ids lazy as long as possible. Materialize them only for Python residual hooks or legacy env-id-only
APIs that cannot consume a mask.

Residual Ordering Rule
----------------------

Split reset support lets each scene entity implement:

* ``reset_graphable(env_ids=None, env_mask=None)`` for graph-capturable tensor/kernel work.
* ``reset_after_graph(env_ids=None, env_mask=None, graphable_reset_applied=False)`` for Python-side residual state.

The scene runs graphable work for all entities before residual work. Therefore, a residual hook must not read another
entity's freshly reset GPU state unless that dependency is captured in an earlier ordered phase. When ordering matters,
use separate :class:`~isaaclab.envs.cuda_graph.ResetGraphPhase` objects and place the residual hook in
``between_hook``.

Residual hooks must use the explicit ``graphable_reset_applied`` argument rather than Python state mutated by
``reset_graphable(...)``. Python in ``reset_graphable(...)`` runs during warm-up and capture, but it does not run on CUDA
graph replay. A hook that needs to skip a legacy full reset because its graphable work already replayed should check
``graphable_reset_applied``.

Camera and Renderer Reset
-------------------------

Camera reset graphability is supported only for initialized cameras whose renderer and view advertise graphable reset
support. The optimized camera path requires a stable mask selector; it does not use dynamic ``env_ids`` indexing inside
capture.

For the Newton renderer path, :meth:`~isaaclab.sensors.Camera.reset_graphable` captures the graphable pieces of camera
reset when all of these are true:

* ``env_mask`` is provided.
* The camera is initialized.
* The renderer provides ``supports_camera_reset_graphable(...)``.
* ``RenderData`` already owns stable renderer buffers such as camera transforms and rays.
* The camera view exposes graph tensor access for world-pose computation.

In that path, camera reset updates base sensor timestamp/outdated buffers, camera frame counters, camera world pose data,
and renderer camera transform buffers through stable full-view operations. ``reset_after_graph(...)`` then intentionally
skips the full camera reset for that replay. Unsupported camera backends use the residual full reset path, so adding a
camera to a task is not enough to make reset graph replay valid; the renderer/view pair must explicitly opt in and its
captured tensors must be returned by ``reset_graph_tensors()``.

Buffer Wiring
-------------

Use :class:`~isaaclab.utils.warp_view_registry.WarpViewRegistry` for Torch-backed Warp views and
:class:`~isaaclab.utils.warp_view_registry.CudaGraphTargetRegistry` for backend-owned Warp arrays. Prefer group names
that describe the use site:

* ``step`` for fused done/reward kernels.
* ``reset`` for direct fused reset kernels.
* ``reset_graph`` for buffers captured by CUDA graph replay guards.

This keeps view refresh and replay-guard registration centralized instead of hand-maintaining parallel lists.

Minimum Example
---------------

.. code-block:: python

   class MyTask(DirectRLEnv):
       def __init__(self, cfg, render_mode=None, **kwargs):
           super().__init__(cfg, render_mode=render_mode, **kwargs)
           self._configure_reset_cuda_graph(path_name="MyTask reset CUDA graph path")

       def _setup_reset_cuda_graph_buffers(self):
           self._reset_env_mask_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
           self._reset_scratch_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

       def _warmup_reset_cuda_graph(self):
           ctx = ResetContext(env_ids=None, reset_mask_wp=self._reset_env_mask_wp)
           self._reset_idx_common_graphable(ctx, reset_episode_lengths=False)
           self._prepare_task_apply_capture_state()
           self._launch_task_apply_reset_graphable(ctx)

       def _reset_cuda_graph_tensors(self):
           tensors = super()._reset_cuda_graph_tensors()
           tensors["task.reset_scratch"] = self._reset_scratch_wp
           return tensors

       def _reset_cuda_graph_constants(self):
           return {"num_envs": int(self.num_envs)}

       def _reset_idx_cuda_graph_impl(self, ctx):
           def after_common_reset(graph_ctx):
               graph_ctx = graph_ctx.with_env_ids(graph_ctx.selection.materialize_env_ids(device=self.device))
               if len(graph_ctx.env_ids) == 0:
                   return graph_ctx, False
               self._reset_idx_common_after_graph(graph_ctx, reset_episode_lengths=False)
               return graph_ctx, True

           def after_task_apply(graph_ctx):
               self._apply_task_reset_to_sim_after_graph()
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
                   name="task/apply reset",
                   launch_fn=self._launch_task_apply_reset_graphable,
                   prepare_capture=self._prepare_task_apply_capture_state,
                   between_hook=after_task_apply,
               ),
           )
           replay_result = self._replay_reset_cuda_graph_phases(ctx, phases)
           if replay_result is None:
               return None
           graph_ctx, replay_completed = replay_result
           return graph_ctx.env_ids

       def _launch_task_apply_reset_graphable(self, ctx):
           wp.launch(_my_task_reset_kernel, dim=self.num_envs, inputs=[ctx.reset_mask_wp, self._reset_scratch_wp])
           self._write_task_reset_to_sim_graphable(ctx)

The in-hand manipulation environment is the reference implementation. It keeps the fused reset kernels task-local,
uses the shared replay helper for capture orchestration, and tests fused Warp results against the Torch reference path.
