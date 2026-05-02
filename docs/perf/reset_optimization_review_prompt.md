You are reviewing an IsaacLab Newton backend performance prototype in:

`/home/rshahid/Projects/isaac/IsaacLab`

Branch:

`rshahid/newton-perf-iter`

Review target:

Review all changes in this commit range from scratch:

`61c8a9457c461f5d7097c451f734b63fc02e7594..8359cc99a8c93e606056549e1b253e2a48dd69f2`

The left side is the branch merge-base with `origin/develop`; the right side is the reset optimization code/test/report commit. If the branch has a later prompt-only commit, do not include that prompt commit in the code review range.

Do not anchor on any previous review or assume earlier findings were complete or correct. Read the diff and source directly, reconstruct the old behavior from the left side of the range where needed, and produce an independent review.

Ignore unrelated untracked files unless they affect the reset optimization work. Do not modify code unless explicitly asked; this is a review task.

Goal:

Review the reset optimization work for correctness, maintainability, API cleanliness, and test quality. Be extremely critical. Correctness is non-negotiable; do not accept performance wins that subtly change reset semantics.

Context:

The working tree tries to reduce CUDA launch overhead in the Newton backend, especially for in-hand manipulation reset, done, and reward paths. The broad design includes:

1. Split reset APIs:
   - `reset_graphable(...)`
   - `reset_after_graph(...)`
   - `reset_graph_tensors(...)`
2. Optional CUDA graph capture/replay for graphable reset work.
3. Fused Warp kernels for in-hand reset, dones, and rewards.
4. A `DirectRLEnv.step()` reset entrypoint that lets a subclass consume `reset_buf` as a GPU mask before materializing `reset_buf.nonzero()`.
5. Newton asset/sensor reset support for mask-native graphable reset pieces.

Important instruction:

Do a from-scratch source review. Treat every changed file as potentially wrong until you verify it against old behavior and local invariants. The previous report `docs/perf/reset_optimization_review.md` and handoff `docs/perf/inhand_reset_cuda_graph_report.md` may be useful context, but do not rely on their conclusions.

Suggested first commands:

```bash
cd /home/rshahid/Projects/isaac/IsaacLab
git status --short
git diff --stat 61c8a9457c461f5d7097c451f734b63fc02e7594..8359cc99a8c93e606056549e1b253e2a48dd69f2
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594..8359cc99a8c93e606056549e1b253e2a48dd69f2 -- source/isaaclab/isaaclab/envs/direct_rl_env.py
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594..8359cc99a8c93e606056549e1b253e2a48dd69f2 -- source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
```

Changed source files to review:

- `source/isaaclab/isaaclab/envs/direct_rl_env.py`
- `source/isaaclab/isaaclab/envs/cuda_graph.py`
- `source/isaaclab/isaaclab/scene/interactive_scene.py`
- `source/isaaclab/isaaclab/assets/asset_base.py`
- `source/isaaclab/isaaclab/sensors/sensor_base.py`
- `source/isaaclab/isaaclab/utils/wrench_composer.py`
- `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`
- `source/isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py`
- `source/isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/imu/imu.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/pva/pva.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py`
- `docs/perf/inhand_reset_cuda_graph_report.md`

Changed tests to review:

- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py`
- `source/isaaclab/test/scene/test_interactive_scene.py`
- `source/isaaclab/test/envs/test_cuda_graph_replay_guard.py`
- `source/isaaclab/test/utils/test_wrench_composer.py`
- `source/isaaclab_newton/test/assets/test_rigid_object_reset_kitless.py`
- `source/isaaclab_newton/test/sensors/test_imu.py`
- `source/isaaclab_newton/test/sensors/test_pva.py`
- `source/isaaclab_newton/test/sensors/test_contact_sensor.py`

Core review questions:

1. Does `DirectRLEnv.step()` still preserve old reset semantics?
   - Old behavior materialized `reset_env_ids = reset_buf.nonzero(...)`, then called `_reset_idx(reset_env_ids)` only if nonempty, then optionally rerendered on reset.
   - New behavior routes through `_reset_idx_from_reset_buf()`, `_try_reset_idx_cuda_graph()`, and fallback materialization.
   - Check rerender-on-reset, event/noise reset, extras/logging, episode length reset, observations, and no-reset behavior.

2. Is the split reset API clean and semantically sound?
   - `reset(...)` should remain the full semantic reset.
   - `reset_graphable(...)` should contain only graph-capturable tensor/kernel work.
   - `reset_after_graph(...)` should contain residual Python/non-graph-safe state.
   - `reset_graph_tensors(...)` should enumerate every tensor/Warp array whose storage/metadata is captured.
   - Check whether contributors can extend this without duplicating reset logic or accidentally skipping behavior.

3. Does `InteractiveScene` preserve ordering where it must?
   - `reset(env_mask=None)` should match old per-entity full reset order.
   - `reset(env_mask=...)` composes graphable and residual work per entity.
   - `reset_graphable(); reset_after_graph()` batches all graphable work before residual work. Decide whether that is safe and clearly documented.
   - Look for cross-entity or own-entity ordering assumptions broken by batching.

4. Are Newton asset resets correct?
   - Review `Articulation.reset`, `reset_graphable`, `reset_after_graph`, actuator reset, and wrench composer reset.
   - Review `RigidObject` and `RigidObjectCollection` reset behavior.
   - Check full reset, partial env-id reset, mask reset, and fallback behavior.
   - Verify Python-side flags like WrenchComposer `_active` and `_dirty` remain correct.

5. Are sensor resets correct?
   - `SensorBase.reset_graphable()` resets base timestamp/outdated buffers.
   - Child Newton sensors call `super().reset_graphable(...)` and launch their own reset kernels.
   - `reset_graph_tensors()` should cover base and child buffers.
   - Check fallback behavior for sensors that do not override graph reset APIs.

6. Is CUDA graph safety actually guaranteed?
   - Review `CudaGraphReplayGuard` and its use from `DirectRLEnv`.
   - Check every tensor/Warp array captured by each reset graph is tracked.
   - Check scalar constants baked into captures are tracked.
   - Check recapture after compatible storage rebinding.
   - Check disable/fail behavior after incompatible metadata changes.
   - Check graph capture does not include unsafe CPU/Python state mutation.
   - Check no stale `wp.to_torch(...)` views remain after Warp array rebinding.

7. Is the in-hand reset path semantically equivalent to old `_reset_idx`?
   - Compare against `HEAD` implementation.
   - Review graph mode and non-graph fused mode separately.
   - Check reset events and noise reset ordering.
   - Check episode length reset, `successes`, `_last_episode_success`, `reset_goal_buf`, success-rate metrics, RNG state, target tensors, object/hand state writes, `sim.forward()`, and intermediate value publication.
   - Check empty masks, partial masks, noncontiguous env ids, repeated resets, and all-env resets.
   - Check goal marker visualization behavior for visual and non-visual cases.

8. Are the fused Warp dones/rewards correct?
   - Compare the Warp kernels to old Torch logic.
   - Check `reset_goal_buf`, `successes`, `consecutive_successes`, timeout/termination logic, max consecutive success behavior, and goal reset RNG.
   - Check intermediate tensors match canonical recomputation.

9. Are the tests strong enough?
   - Do tests prove semantic equivalence, or mostly implementation details?
   - Are stale CUDA graph pointer cases actually caught?
   - Are edge cases covered: empty resets, no reset after capture, partial/noncontiguous ids, repeated resets, storage rebinding, metadata changes, reset events that mutate task tensors, vision/goal marker behavior, sensors, assets, WrenchComposer flags?
   - Identify missing old-path/reference comparisons.

10. Is the API maintainable?
   - Look for duplicate implementations of the same behavior.
   - Look for names that still imply CUDA graph when the code is really just fused Warp or graphable reset.
   - Look for task-specific assumptions leaking into base classes.
   - Look for fragile `getattr` or duck-typing patterns.

Current intended in-hand graph reset order, to verify rather than trust:

1. replay common reset graph
2. materialize env ids if needed
3. return early if mask is empty and ids were materialized from mask
4. run common after-graph residual
5. replay task reset graph
6. replay apply-to-sim graph
7. run writer after-graph hooks
8. publish intermediates and metrics

Current intended in-hand non-graph fused reset order, to verify rather than trust:

1. snapshot success metrics
2. run common graphable kernels directly
3. run common residual
4. run task reset-prepare kernels directly
5. apply reset to sim directly
6. publish intermediates and metrics

Known validation results from the latest run:

- `python -m py_compile` on changed source/test files: passed
- `git diff --check`: passed
- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py -q --tb=short`
  - `57 passed, 195 warnings in 139.83s`
- `source/isaaclab/test/scene/test_interactive_scene.py -q --tb=short`
  - `12 passed, 43 warnings in 11.50s`
- `source/isaaclab/test/envs/test_cuda_graph_replay_guard.py -q --tb=short`
  - `7 passed in 2.77s`
- `source/isaaclab/test/utils/test_wrench_composer.py -q --tb=short`
  - `370 passed in 11.55s`
- `source/isaaclab_newton/test/sensors/test_imu.py source/isaaclab_newton/test/sensors/test_pva.py source/isaaclab_newton/test/sensors/test_contact_sensor.py -q --tb=short`
  - `99 passed, 6 xpassed, 880 warnings in 451.21s`

Full repository tests were not run.

Recent traces:

- Instrumentation-only baseline:
  - `/home/rshahid/Projects/isaac/reset_instrumentation_baseline.nsys-rep`
  - `/home/rshahid/Projects/isaac/reset_instrumentation_baseline.sqlite`
- Fused Warp kernels, reset CUDA graph disabled:
  - `/home/rshahid/Projects/isaac/reset_fused_warp_no_reset_graph.nsys-rep`
  - `/home/rshahid/Projects/isaac/reset_fused_warp_no_reset_graph.sqlite`
- Fused Warp kernels, reset CUDA graph forced:
  - `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.nsys-rep`
  - `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.sqlite`
- Current split graph-reset API, reset CUDA graph forced:
  - `/home/rshahid/Projects/isaac/reset_current_graph_split_force.nsys-rep`
  - `/home/rshahid/Projects/isaac/reset_current_graph_split_force.sqlite`

Useful context files:

- `docs/perf/inhand_reset_cuda_graph_report.md`
- `docs/perf/reset_optimization_review.md`

Deliverable:

Provide a code-review style response:

- Findings first, ordered by severity, with file/line references.
- Include correctness risks, API/maintainability concerns, and missing tests.
- If no severe issues are found, say that explicitly and list residual risks.
- Include any test commands you ran and their results.
- Do not rewrite the code unless asked; this is a review.
