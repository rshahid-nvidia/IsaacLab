# In-Hand Newton Reset Optimization Handoff

Date: 2026-05-01

Branch: `rshahid/newton-perf-iter`

## Commits

- `3027956b2b3 Add reset profiling instrumentation`
  - Adds step/reset NVTX ranges and benchmark capture frame controls.
- `3b4172bff4a Optimize in-hand Newton reset and step kernels`
  - Adds the in-hand fused Warp reset core, fused dones/intermediate kernels, fused reward kernels, and reset CUDA graph support for the pure reset-preparation subset.
- `365f780a64f Refine in-hand reset writer and mask hooks`
  - Restores mask writer aliases for readability and briefly added a base mask-reset hook.
- `Remove premature reset mask hook`
  - Removes the base `DirectRLEnv._reset_idx_from_mask()` hook because it did not yet eliminate `nonzero()` for in-hand and made the generic reset flow harder to read.
- `cf14f1bd949 Add in-hand reset graph correctness tests`
  - Adds replay-guard, in-hand reset/dones/rewards, Newton rigid-object reset, and kitless wrench-composer coverage.

## Current Design

The reset API is now split by responsibility instead of by task-specific special cases:

1. `reset(...)`
   - Remains the full semantic reset.
   - For scene resets without `env_mask`, `InteractiveScene.reset()` preserves the old per-entity reset order.
2. `reset_graphable(...)`
   - Optional graph-capturable tensor/kernel work.
   - Default asset implementation is a no-op; `SensorBase` resets its base timestamp/outdated buffers.
3. `reset_after_graph(...)`
   - Residual Python or non-graph-safe state.
   - Default asset/sensor fallback calls the old full `reset(...)` when concrete `env_ids` are available.

The in-hand reset implementation uses the split in this order:

1. `_reset_idx_common_graphable(...)`
   - Captures graphable pieces of the parent `DirectRLEnv` reset sequence, currently scene graphable reset work and optional episode-length reset.
2. `_reset_idx_common_after_graph(...)`
   - Runs the parent residual work: scene `reset_after_graph`, reset events, action noise reset, and observation noise reset.
3. `_launch_inhand_task_reset_graphable(...)`
   - Runs fused Warp kernels that prepare goal rotation, object pose/velocity, hand joint pos/vel, target tensors, episode length reset, success clearing, and reset metrics inputs.
4. `_apply_inhand_reset_to_sim(...)`
   - Applies the prepared buffers through the asset writer APIs:
     - object root pose
     - object root velocity
     - hand joint target
     - hand joint position
     - hand joint velocity
   - Then calls `sim.forward()`, refreshes Warp state inputs, and recomputes intermediate values.

The CUDA-graph mode uses three separate graphs to avoid reordering parent residual work after task reset work:

1. Replay common reset graph.
2. Run common residual reset outside graph.
3. Replay in-hand task reset graph.
4. Replay the graph-captured simulation writes and immediate intermediate recompute.
5. Run Python-side writer after-graph hooks.

The non-graph fused mode uses the same ordering with direct kernel launches instead of graph replay. The
`_apply_inhand_reset_to_sim()` ordering matches the old task-specific reset write order. The only meaningful task-local
ordering difference is that `successes[env] = 0` happens inside the task graphable kernel before sim writes. That is safe
because parent reset/events/noise and reset success accounting happen before the clear, and the writer APIs do not depend
on `successes`.

Graph-aware scene support currently covers Newton external-wrench composers on rigid objects, rigid object collections,
and articulations, plus Newton IMU/PVA/ContactSensor data reset. `SensorBase.reset_graphable(...)` owns the base
timestamp/outdated-buffer reset and returns the resolved mask; sensor child classes call `super().reset_graphable(...)`
and then launch only their own sensor-specific reset kernels. Unsupported entities use the base fallback path.

Actuator resets remain outside graph capture. Implicit and ideal PD actuators are no-op resets. Delayed and learned
actuators mutate delay/history tensors and may use reset-time randomness, so they need separate mask/graph-safe support
before being moved into `reset_graphable(...)`.

## Correctness Status

I did not run the full Isaac Lab repository test suite. The focused suites below passed.

Commands run with:

```bash
env -u VIRTUAL_ENV CONDA_PREFIX=/home/rshahid/miniconda3/envs/env_isaaclab \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ./isaaclab.sh -p ...
```

Passed:

- `-m py_compile` on all changed source and test files.
- `git diff --check`
- `-m pytest source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py -q --tb=short`
  - `57 passed, 195 warnings in 139.83s`
- `-m pytest source/isaaclab/test/scene/test_interactive_scene.py -q --tb=short`
  - `12 passed, 43 warnings in 11.50s`
- `-m pytest source/isaaclab/test/envs/test_cuda_graph_replay_guard.py -q --tb=short`
  - `7 passed in 2.77s`
- `-m pytest source/isaaclab/test/utils/test_wrench_composer.py -q --tb=short`
  - `370 passed in 11.55s`
- `-m pytest source/isaaclab_newton/test/sensors/test_imu.py source/isaaclab_newton/test/sensors/test_pva.py source/isaaclab_newton/test/sensors/test_contact_sensor.py -q --tb=short`
  - `99 passed, 6 xpassed, 880 warnings in 451.21s`
- `-m pytest source/isaaclab_newton/test/assets/test_rigid_object_reset_kitless.py source/isaaclab_newton/test/assets/test_rigid_object_collection.py::test_reset_object_collection -q --tb=short`
  - `12 passed, 143 warnings in 38.25s`

Review feedback addressed after `docs/perf/reset_optimization_review.md`:

- Expanded the in-hand reset CUDA graph replay guard to cover all task/apply graph Warp arrays, persistent output
  buffers, Newton FK reset masks, articulation id maps, joint masks, and relevant scalar constants.
- Split reset success accounting into a pre-residual snapshot kernel so reset events that mutate `successes` do not
  change `_last_episode_success` in the non-graph fused reset path.
- Added a true step-entry no-reset test with `ctx.env_ids is None` after graph capture.
- Added graph-reset goal-marker visualization coverage with `_should_sync_goal_markers()` forced true.
- Documented and tested the scene split ordering: graph capture batches graphable work before residual work, while
  `InteractiveScene.reset(env_mask=...)` preserves per-entity semantic composition.
- Moved base sensor timestamp/outdated buffers into `SensorBase.reset_graph_tensors()` and made Newton sensor overrides
  extend `super().reset_graph_tensors()`.

Additional isolated/manual checks:

- Two PhysX compatibility checks for in-hand Warp buffer stability and PhysX reward/done reference matching passed when run individually.
- The same PhysX checks caused an ordering/lifetime problem when appended after the full Newton in-hand test file, so they were not committed into the default test file. Do not treat PhysX graph/reset compatibility as fully validated from this work.
- Earlier grouped `warp_dones` selection had one transient `SIGSEGV`; the individual tests and final full Newton in-hand file passed afterward.

Current confidence:

- High for the covered Newton/CUDA in-hand reset, dones, rewards, replay guard, scene reset split, WrenchComposer reset,
  Newton rigid-object/collection reset, and Newton IMU/PVA/ContactSensor reset split paths.
- Not global correctness. Full repository tests and broader task coverage were not run.

## Trace

Generated traces with matching settings:

- Task: `Isaac-Repose-Cube-Allegro-Direct-v0`
- Backend: Newton
- Renderer: non-vision/headless
- Envs: 1024
- Frames: 80
- Capture frames: 30-60

Instrumentation-only baseline at `3027956b2b3`:

- `/home/rshahid/Projects/isaac/reset_instrumentation_baseline.nsys-rep`
- `/home/rshahid/Projects/isaac/reset_instrumentation_baseline.sqlite`
- `/tmp/nsys_reset_instrumentation_baseline.log`
- `/tmp/nsys_reset_instrumentation_baseline_json/benchmark_non_rl_Isaac-Repose-Cube-Allegro-Direct-v0_2026-05-01_05-17-36.json`

Fused Warp kernels with reset CUDA graph disabled (`env.reset_cuda_graph=off`):

- `/home/rshahid/Projects/isaac/reset_fused_warp_no_reset_graph.nsys-rep`
- `/home/rshahid/Projects/isaac/reset_fused_warp_no_reset_graph.sqlite`
- `/tmp/nsys_reset_fused_warp_no_reset_graph.log`
- `/tmp/nsys_reset_fused_warp_no_reset_graph_json/benchmark_non_rl_Isaac-Repose-Cube-Allegro-Direct-v0_2026-05-01_05-18-41.json`

Fused Warp kernels with reset CUDA graph forced (`env.reset_cuda_graph=force`):

- `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.nsys-rep`
- `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.sqlite`
- `/tmp/nsys_reset_cuda_graph_force_fused_rewards.log`
- `/tmp/nsys_reset_cuda_graph_force_fused_rewards_json/benchmark_non_rl_Isaac-Repose-Cube-Allegro-Direct-v0_2026-05-01_04-05-59.json`

Current working tree after split graph-reset API refactor (`env.reset_cuda_graph=force`):

- `/home/rshahid/Projects/isaac/reset_current_graph_split_force.nsys-rep`
- `/home/rshahid/Projects/isaac/reset_current_graph_split_force.sqlite`
- `/tmp/nsys_reset_current_graph_split_force.log`
- `/tmp/nsys_reset_current_graph_split_force_json/benchmark_non_rl_Isaac-Repose-Cube-Allegro-Direct-v0_2026-05-01_13-25-51.json`

Headline benchmark JSON stats:

- Baseline: mean step time 52.50 ms; mean step FPS 27.54; mean effective FPS 28197.67.
- Fused/no-reset-graph: mean step time 36.72 ms; mean step FPS 30.88; mean effective FPS 31620.33.
- Fused/reset-graph-force: mean step time 33.82 ms; mean step FPS 33.56; mean effective FPS 34360.48.
- Current/split-graph-force: mean step time 34.82 ms including first-frame warmup outlier; 30.21 ms excluding frame 0;
  mean step FPS 33.24 including frame 0; mean effective FPS 34037.24 including frame 0.

Sanity from `nsys stats`:

- Baseline CUDA API calls: `cudaGraphLaunch_v10000` 120; `cudaLaunchKernel` 7459; `cudaStreamSynchronize` 2019.
- Fused/no-reset-graph CUDA API calls: `cudaGraphLaunch_v10000` 120; `cudaLaunchKernel` 3900; `cudaStreamSynchronize` 1710.
- Fused/reset-graph-force CUDA API calls: `cudaGraphLaunch_v10000` 150; `cudaLaunchKernel` 3810; `cudaStreamSynchronize` 1680.
- NVTX reset ranges were present in all traces. Baseline `_reset_idx` averaged 6.63 ms, fused/no-reset-graph `_reset_idx` averaged 2.95 ms, and fused/reset-graph-force `_reset_idx` averaged 2.06 ms.
- Current/split-graph-force CUDA API calls: `cudaGraphLaunch_v10000` 180; `cudaLaunchKernel` 3810;
  `cudaStreamSynchronize` 1680. NVTX `_reset_idx` averaged 2.12 ms across the captured 30-frame bracket.

## Known Limitations

- `_reset_idx_common_after_graph(...)` still requires exact `env_ids` for residual scene fallback, reset events, and noise reset.
- Reset CUDA graph capture covers graphable parent reset kernels and graphable in-hand task kernels, but not residual parent
  reset work, asset writer APIs, or `sim.forward()` because those update Python-side lazy-buffer/FK state.
- `reset_graph_tensors()` is intentionally required for graph-aware entities. CUDA graphs capture raw tensor/Warp-array
  pointers, so the replay guard must validate storage/metadata stability for graphable scene buffers.
- Reward computation is intentionally two Warp launches, not one, because `consecutive_successes` depends on a grid-wide reduction. A single parallel kernel would either race or serialize.
- Goal marker visualization is skipped in non-vision headless paths and kept only when renderer/visualizer state can observe it.
- The premature base `DirectRLEnv._reset_idx_from_mask()` hook was removed. The current in-hand path still converts `reset_buf` to exact `env_ids` before reset because exact-env side effects still need careful handling.

## Next Work

For the `reset_buf.nonzero()` synchronization:

1. Extend `ResetContext` into a true reset-selection object:
   - always carries `reset_mask_torch/reset_mask_wp`
   - materializes `env_ids` lazily only when a legacy Python hook requests exact IDs
2. Move in-hand reset bookkeeping that currently requires `env_ids` into Warp kernels:
   - `_last_episode_success`
   - `reset_goal_buf` clearing
   - success-rate metric buffer
3. Add a mask-safe replacement for the relevant subset of `_reset_idx_common(...)`:
   - only when reset events/noise/sensors are absent or have mask-safe APIs
   - only when actuators are stateless/no-op for reset, or actuator reset gets mask-safe support
   - reset Newton external wrench composers with the mask
4. Only then add a clear task-level mask reset API for in-hand. Until that is done, falling back to `nonzero()` is the safer behavior.

For broader task support:

- Keep the base hook and reset-preparation API task-agnostic.
- Add task-specific fused reset cores only where the task can prove all reset side effects are either mask-safe or still correctly handled by exact-ID fallback.
- Avoid maintaining duplicate Torch and Warp production paths. Keep Torch/reference implementations only in tests where possible.
