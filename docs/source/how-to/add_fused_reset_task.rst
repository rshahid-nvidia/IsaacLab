Adding a Fused Reset Task
=========================

.. currentmodule:: isaaclab

Large direct RL tasks can spend more time launching many small CUDA kernels than executing the kernels themselves.
For Newton-backed tasks with simple per-environment reset, done, and reward logic, a fused Warp path can reduce that
launch overhead while preserving the existing semantic reset path.

When to Use This Path
---------------------

Use a fused reset path when all of the following are true:

* The task runs on the Newton backend with CUDA tensors.
* The workload uses many environments and reset/done/reward kernels are launch-overhead dominated.
* The reset math can be expressed as Warp kernels or graph-capturable Torch operations over stable
  per-environment buffers.
* Any Python-side reset state can run before or after the graphable kernels without reading freshly reset GPU state
  from another entity out of order.

Do not add this path only to avoid writing an efficient Torch implementation. Prefer graph-captured Torch operations
when they preserve semantics and avoid dynamic selector shapes. In particular, PyTorch's CUDA graph integration can
handle allocator-owned temporaries, but the external tensors read or written across replay still need stable storage.
Under Isaac Lab's relaxed Warp/CUDA capture helper, replay-time Torch RNG does not advance today; reset logic that
must randomize on every replay should use a task-owned RNG kernel or stay outside capture until that capture path
changes. The Torch path remains the semantic fallback for unsupported backends, CPU runs, and tests that compare fused
kernels against the reference behavior.

Reset Phases
------------

The reusable environment hooks live on :class:`~isaaclab.envs.DirectRLEnv`. A task that supports CUDA graph replay
usually implements these methods:

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
* ``reset_after_graph(env_ids=None, env_mask=None)`` for Python-side residual state.

The scene runs graphable work for all entities before residual work. Therefore, a residual hook must not read another
entity's freshly reset GPU state unless that dependency is captured in an earlier ordered phase. When ordering matters,
use separate :class:`~isaaclab.envs.cuda_graph.ResetGraphPhase` objects and place the residual hook in
``between_hook``.

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
       def _setup_reset_cuda_graph_buffers(self):
           self._reset_env_mask_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
           self._reset_scratch_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

       def _warmup_reset_cuda_graph(self):
           ctx = ResetContext(env_ids=None, reset_mask_wp=self._reset_env_mask_wp)
           self._launch_task_reset_graphable(ctx)

       def _reset_cuda_graph_tensors(self):
           tensors = super()._reset_cuda_graph_tensors()
           tensors["task.reset_scratch"] = self._reset_scratch_wp
           return tensors

       def _reset_cuda_graph_constants(self):
           return {"num_envs": int(self.num_envs)}

       def _reset_idx_cuda_graph_impl(self, ctx):
           phases = (
               ResetGraphPhase(
                   graph_attr="_reset_common_cuda_graph",
                   name="common reset",
                   launch_fn=self._reset_idx_common_graphable,
                   between_hook=self._after_common_reset,
               ),
               ResetGraphPhase(
                   graph_attr="_reset_task_cuda_graph",
                   name="task reset",
                   launch_fn=self._launch_task_reset_graphable,
               ),
           )
           return self._replay_reset_cuda_graph_phases(ctx, phases)

       def _launch_task_reset_graphable(self, ctx):
           wp.launch(_my_task_reset_kernel, dim=self.num_envs, inputs=[ctx.reset_mask_wp, self._reset_scratch_wp])

       def _after_common_reset(self, ctx):
           ctx = ctx.with_env_ids(ctx.selection.materialize_env_ids(device=self.device))
           self._reset_idx_common_after_graph(ctx)
           return ctx, True

The in-hand manipulation environment is the reference implementation. It keeps the fused reset kernels task-local,
uses the shared replay helper for capture orchestration, and tests fused Warp results against the Torch reference path.
