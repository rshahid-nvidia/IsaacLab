You are reviewing an IsaacLab Newton backend reset/step performance prototype in:

`/home/rshahid/Projects/isaac/IsaacLab`

Branch:

`rshahid/newton-perf-iter`

Write the review report to:

`docs/perf/reset_optimization_review_codex_pass6.md`

## Review Target

Review all reset optimization changes from scratch. Do not limit the review to the latest small fix. The target is the full diff from the base commit through the latest committed fixes, plus any current tracked working-tree edits in the implementation files.

Base commit:

`61c8a9457c461f5d7097c451f734b63fc02e7594`

Latest committed fix at the time this prompt was written:

`c7d99a70e13 Avoid reset residual mask rematerialization`

Important implementation commits currently in the stack:

- `3027956b2b3 Add reset profiling instrumentation`
- `3b4172bff4a Optimize in-hand Newton reset and step kernels`
- `365f780a64f Refine in-hand reset writer and mask hooks`
- `cf14f1bd949 Add in-hand reset graph correctness tests`
- `11158acf5c9 Document in-hand reset optimization status`
- `4528fa3cb79 Remove premature reset mask hook`
- `dac6ef08664 Record reset nsys trace comparisons`
- `8359cc99a8c Address reset optimization review feedback`
- `d846ca9cbf6 Update reset optimization review prompt`
- `593fec79c58 Fix reset graph explicit env selection`
- `a9834108513 Honor Newton reset mask precedence`
- `c7d99a70e13 Avoid reset residual mask rematerialization`

Primary review diff, including current tracked working-tree edits:

```bash
cd /home/rshahid/Projects/isaac/IsaacLab
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594 -- \
  ':(exclude)docs/perf/reset_optimization_review_prompt.md' \
  ':(exclude)docs/perf/reset_optimization_review*.md'
```

Committed-only reference diff:

```bash
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594..c7d99a70e13 -- \
  ':(exclude)docs/perf/reset_optimization_review_prompt.md' \
  ':(exclude)docs/perf/reset_optimization_review*.md'
```

Do not anchor on previous reviews or assume earlier findings were complete. Read the diff and source directly. Reconstruct old behavior from the base commit where needed.

Do not modify code. This is a production-level code review task.

## Output Requirements

Use a code-review format:

- Findings first, ordered by severity.
- Use severity labels: P0, P1, P2, P3.
- Every finding must include precise file and line references.
- Explain the observable correctness, performance, CUDA safety, or maintainability impact.
- Explain why the issue is real by tying it to old behavior, API contracts, or runtime ordering.
- Include concrete fix guidance.
- If there are no findings, say that explicitly and list residual risks or test gaps.
- Include a short "Tests/Verification Reviewed" section at the end.

Be extremely critical. Correctness is non-negotiable. Do not accept performance wins that subtly change reset semantics, selector precedence, randomization, graph replay assumptions, event/noise behavior, or public task behavior.

## Context

The prototype tries to reduce CUDA launch overhead in the Newton backend, especially for in-hand manipulation reset, done, and reward paths. The broad design includes:

1. Split reset APIs:
   - `reset_graphable(...)`
   - `reset_after_graph(...)`
   - `reset_graph_tensors(...)`
2. Optional CUDA graph capture/replay for graphable reset work.
3. Fused Warp kernels for in-hand reset, dones, rewards, and reset-to-sim writes.
4. A `DirectRLEnv.step()` reset entrypoint that lets a subclass consume `reset_buf` as a GPU mask before materializing `reset_buf.nonzero()`.
5. Newton asset/sensor reset support for mask-native graphable reset pieces.
6. Compatibility fallback for unsupported/default in-hand paths, especially public PhysX task configs.
7. CUDA/NVTX instrumentation for profiling, guarded so CPU and non-CUDA paths do not break.

## Recent Feedback Fixes To Re-review

Treat these as areas to verify, not as trusted facts.

1. Pass 3 fixed Newton mask-only articulation reset:
   - `Articulation.reset(env_mask=...)` and `reset_after_graph(env_mask=...)` should not reset all actuator envs.
   - Actuator reset receives materialized ids from the mask when only a mask is provided.

2. Pass 3 fixed explicit graph reset ids:
   - `_reset_idx_cuda_graph(env_ids)` now rewrites `reset_buf` from explicit ids before graph replay.
   - This should make the stable graph mask and residual ids a single source of truth.
   - Verify this does not break normal `step()` where `_try_reset_idx_cuda_graph()` is called with `env_ids=None`.

3. Pass 4 fixed Newton mask precedence:
   - Public Newton articulation reset must follow the base contract: if both `env_ids` and `env_mask` are provided, `env_mask` takes precedence.
   - Verify actuator reset and wrench-composer reset now select the same environments.

4. Pass 5 fixed redundant residual rematerialization:
   - `_reset_idx_common_after_graph()` now calls `scene.reset_after_graph(env_ids=ctx.env_ids, env_mask=None)`.
   - The graphable phase still consumes `ctx.reset_mask_wp`.
   - The residual phase relies on `ctx.env_ids` already being derived from the same mask.
   - Verify this is safe for all current callers of `_reset_idx_common_after_graph()`, including graph and non-graph fused in-hand paths.
   - Verify this does not weaken public `InteractiveScene.reset(env_mask=...)` or `Articulation.reset(env_ids=..., env_mask=...)` mask-precedence semantics.

5. Previous fixes also touched:
   - default PhysX in-hand task fallback
   - reset seed reproducibility for persistent Warp RNG buffers
   - CPU/non-CUDA NVTX guards
   - `SensorBase` `wp.array` env-id support
   - default sensor split-reset fallback behavior
   - `AssetBase.reset_after_graph(env_mask=...)` fallback requiring concrete ids

## Suggested First Commands

```bash
cd /home/rshahid/Projects/isaac/IsaacLab
git status --short
git log --oneline --decorate --graph -18
git diff --stat 61c8a9457c461f5d7097c451f734b63fc02e7594 -- \
  ':(exclude)docs/perf/reset_optimization_review_prompt.md' \
  ':(exclude)docs/perf/reset_optimization_review*.md'
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594 -- source/isaaclab/isaaclab/envs/direct_rl_env.py
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594 -- source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
git diff 61c8a9457c461f5d7097c451f734b63fc02e7594 -- source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py
```

## Changed Files To Review

Source, scripts, and docs:

- `docs/perf/inhand_reset_cuda_graph_report.md`
- `nsys_benchmark.sh`
- `scripts/benchmarks/benchmark_non_rl.py`
- `scripts/benchmarks/minimal_newton_stepper.py`
- `scripts/benchmarks/nvtx_sync_instrumenter.py`
- `scripts/benchmarks/pipelined_stepper.py`
- `source/isaaclab/isaaclab/assets/asset_base.py`
- `source/isaaclab/isaaclab/envs/cuda_graph.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env.py`
- `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py`
- `source/isaaclab/isaaclab/scene/interactive_scene.py`
- `source/isaaclab/isaaclab/sensors/sensor_base.py`
- `source/isaaclab/isaaclab/utils/wrench_composer.py`
- `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`
- `source/isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py`
- `source/isaaclab_newton/isaaclab_newton/assets/rigid_object_collection/rigid_object_collection.py`
- `source/isaaclab_newton/isaaclab_newton/physics/_cubric.py`
- `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/imu/imu.py`
- `source/isaaclab_newton/isaaclab_newton/sensors/pva/pva.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_vision_env_cfg.py`

Tests:

- `source/isaaclab/test/assets/test_articulation_iface.py`
- `source/isaaclab/test/envs/test_cuda_graph_replay_guard.py`
- `source/isaaclab/test/scene/test_interactive_scene.py`
- `source/isaaclab/test/utils/test_wrench_composer.py`
- `source/isaaclab_newton/test/assets/test_rigid_object_reset_kitless.py`
- `source/isaaclab_newton/test/sensors/test_contact_sensor.py`
- `source/isaaclab_newton/test/sensors/test_imu.py`
- `source/isaaclab_newton/test/sensors/test_pva.py`
- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py`

Ignore untracked benchmark outputs and review reports unless they directly affect the implementation under review. Use previous review reports only as background, not as authority.

## Core Review Questions

### DirectRLEnv Reset Semantics

Does `DirectRLEnv.step()` still preserve old reset semantics?

- Old behavior materialized `reset_env_ids = reset_buf.nonzero(...)`, then called `_reset_idx(reset_env_ids)` only if nonempty, then optionally rerendered on reset.
- New behavior routes through `_reset_idx_from_reset_buf()`, `_try_reset_idx_cuda_graph()`, optional graph reset, and fallback materialization.
- Check rerender-on-reset, event reset, action/observation noise reset, extras/logging, episode length reset, observations, no-reset behavior, and explicit `_reset_idx_cuda_graph(env_ids)` calls.

### Split Reset API Design

Is the split reset API clean and semantically sound?

- `reset(...)` must remain the full semantic reset.
- `reset_graphable(...)` should contain only graph-capturable tensor/kernel work.
- `reset_after_graph(...)` should contain residual Python or non-graph-safe state.
- `reset_graph_tensors(...)` should enumerate every tensor/Warp array whose storage or metadata is captured.
- Check whether contributors can extend this without duplicating reset logic or skipping behavior.
- Check whether any method names imply CUDA graph when the code is really just fused Warp or mask-native reset.

### InteractiveScene Ordering

Does `InteractiveScene` preserve ordering where it must?

- `reset(env_mask=None)` should match old per-entity full reset order.
- `reset(env_mask=...)` composes graphable and residual work per entity.
- `reset_graphable(); reset_after_graph()` batches all graphable work before residual work. Decide whether that is safe and clearly documented.
- The internal graph reset residual path now passes `env_mask=None` to `scene.reset_after_graph(...)` when concrete ids are known. Verify this does not affect public scene reset semantics.
- Look for cross-entity or own-entity ordering assumptions broken by batching.

### Newton Asset Resets

Are Newton asset resets correct?

- Review `Articulation.reset`, `reset_graphable`, `reset_after_graph`, `_reset_actuators`, and WrenchComposer reset handling.
- Review `RigidObject` and `RigidObjectCollection` reset behavior.
- Check full reset, partial env-id reset, mask reset, both-selector mismatch reset, and fallback behavior.
- Verify Python-side flags like WrenchComposer `_active` and `_dirty` remain correct.
- Verify pass-5 did not reintroduce inconsistent selector behavior between actuator state and graphable wrench state.

### Sensor Resets

Are sensor resets correct?

- `SensorBase.reset_graphable()` resets base timestamp/outdated buffers.
- Child Newton sensors should call `super().reset_graphable(...)` and launch their own reset kernels where needed.
- `reset_graph_tensors()` should cover base and child buffers.
- Check fallback behavior for sensors that do not override graph reset APIs.
- Check `wp.array`, Torch tensor, host sequence, `None`, and mask reset selections.

### CUDA Graph Safety

Is CUDA graph safety actually guaranteed?

- Review `CudaGraphReplayGuard` and its use from `DirectRLEnv`.
- Check every tensor/Warp array captured by each reset graph is tracked.
- Check scalar constants baked into captures are tracked.
- Check recapture after compatible storage rebinding.
- Check disable/fail behavior after incompatible metadata changes.
- Check graph capture does not include unsafe CPU/Python state mutation.
- Check no stale `wp.to_torch(...)` views remain after Warp array rebinding.
- Check that graphable work uses stable buffers and that reset masks are not replaced without recapture.

### In-hand Reset Equivalence

Is the in-hand reset path semantically equivalent to old `_reset_idx`?

- Compare against the base commit implementation.
- Review graph mode, non-graph fused mode, and unsupported fallback mode separately.
- Check reset events and noise reset ordering.
- Check episode length reset, `successes`, `_last_episode_success`, `reset_goal_buf`, success-rate metrics, RNG state, target tensors, object/hand state writes, `sim.forward()`, and intermediate value publication.
- Check empty masks, partial masks, noncontiguous env ids, repeated resets, all-env resets, explicit env ids, and stale `reset_buf`.
- Check default PhysX/public task behavior.
- Check goal marker visualization behavior for visual and non-visual cases.
- Check `reset(seed=...)` reproducibility for both reset RNG and reward/goal reset RNG.

Current intended in-hand graph reset order, to verify rather than trust:

1. replay common reset graph
2. materialize env ids if needed
3. return early if mask is empty and ids were materialized from the mask
4. run common after-graph residual with `env_ids` and `env_mask=None`
5. replay task reset graph
6. replay apply-to-sim graph
7. run writer after-graph hooks
8. publish intermediates and metrics

Current intended in-hand non-graph fused reset order, to verify rather than trust:

1. snapshot success metrics
2. run common graphable kernels directly
3. run common residual with `env_ids` and `env_mask=None`
4. run task reset-prepare kernels directly
5. apply reset to sim directly
6. publish intermediates and metrics

Current intended unsupported fallback behavior, to verify rather than trust:

1. constructor should not raise just because fused Warp/reset graph support is unavailable
2. `_get_dones()` should route to old Torch-style logic
3. `_get_rewards()` should route to old Torch-style logic
4. `_reset_idx()` should route to old Torch-style reset logic
5. reset CUDA graph mode `force` should reject unsupported paths clearly

### Fused Warp Dones/Rewards

Are fused Warp dones/rewards correct?

- Compare the Warp kernels to old Torch logic.
- Check `reset_goal_buf`, `successes`, `consecutive_successes`, timeout/termination logic, max consecutive success behavior, and goal reset RNG.
- Check intermediate tensors match canonical recomputation.
- Check whether any remaining fallback Torch path diverges from the Warp path.
- Check whether the code still has duplicate implementations that are likely to drift.

### CPU, Non-CUDA, And Profiling Safety

Are CPU/non-CUDA and profiling paths safe?

- `DirectRLEnv.step()` and `InteractiveScene.update()` are generic paths and must not require CUDA/NVTX.
- `benchmark_non_rl.py` profiler start/stop and NVTX ranges should be guarded.
- `nsys_benchmark.sh` should not hardcode local machine assumptions, sudo assumptions, or paths that should not be committed.

### Tests

Are the tests strong enough for production confidence?

- Do tests prove semantic equivalence, or mostly implementation details?
- Are stale CUDA graph pointer cases actually caught?
- Are edge cases covered: empty resets, no reset after capture, partial/noncontiguous ids, repeated resets, all-env resets, explicit env ids, stale `reset_buf`, storage rebinding, metadata changes, reset events that mutate task tensors, vision/goal marker behavior, sensors, assets, WrenchComposer flags, seed reproducibility, CPU/NVTX no-op behavior, and default PhysX fallback?
- Identify missing old-path/reference comparisons.
- Identify tests that are too brittle because they assert implementation details rather than behavior.

## Known Validation Results

Latest targeted checks after pass-5:

- `py_compile` on `direct_rl_env.py` and `test_inhand_reset_cuda_graph.py`: passed
- `git diff --check`: passed
- `source/isaaclab/test/assets/test_articulation_iface.py -k "newton_mask_only_reset or newton_env_mask_takes_precedence"`: `6 passed`
- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py -k "orders_common_residual_before_task_graph"`: `1 passed`

Additional targeted checks from earlier passes:

- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py`: `59 passed, 2 skipped`
- `source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py -k "explicit_env_ids_are_single_source_of_truth"`: `1 passed`
- `source/isaaclab/test/envs/test_cuda_graph_replay_guard.py`: `7 passed`
- `source/isaaclab/test/scene/test_interactive_scene.py -k "scene_reset or sensor_base or asset_default_graph or scene_update_nvtx"`: passed
- `source/isaaclab/test/utils/test_wrench_composer.py`: `370 passed`
- `source/isaaclab_newton/test/sensors/test_imu.py source/isaaclab_newton/test/sensors/test_pva.py source/isaaclab_newton/test/sensors/test_contact_sensor.py`: `99 passed, 6 xpassed`

Full repository tests were not run. Some default PhysX smoke tests were skipped in kitless shells and should be run in an environment with Kit available if possible.

## Recent Traces

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

Use traces only as supporting evidence. Source-level correctness review is the priority.

## Background Files

- `docs/perf/inhand_reset_cuda_graph_report.md`
- `docs/perf/reset_optimization_review.md`
- `docs/perf/reset_optimization_review_codex.md`
- `docs/perf/reset_optimization_review_codex_pass2.md`
- `docs/perf/reset_optimization_review_codex_pass3.md`
- `docs/perf/reset_optimization_review_codex_pass4.md`
- `docs/perf/reset_optimization_review_codex_pass5.md`

Use these only as background. Do not rely on their conclusions.
