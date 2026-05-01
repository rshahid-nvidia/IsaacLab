# In-Hand Newton Reset Optimization Handoff

Date: 2026-05-01

Branch: `rshahid/newton-perf-iter`

## Commits

- `3027956b2b3 Add reset profiling instrumentation`
  - Adds step/reset NVTX ranges and benchmark capture frame controls.
- `3b4172bff4a Optimize in-hand Newton reset and step kernels`
  - Adds the in-hand fused Warp reset core, fused dones/intermediate kernels, fused reward kernels, and reset CUDA graph support for the pure reset-preparation subset.
- `365f780a64f Refine in-hand reset writer and mask hooks`
  - Restores mask writer aliases for readability and adds the base `DirectRLEnv._reset_idx_from_mask()` hook so future supported tasks can consume dense reset masks before `reset_buf.nonzero()`.
- `cf14f1bd949 Add in-hand reset graph correctness tests`
  - Adds replay-guard, in-hand reset/dones/rewards, Newton rigid-object reset, and kitless wrench-composer coverage.

## Current Design

The in-hand reset implementation is split into three phases:

1. `_reset_idx_common(...)`
   - Preserves the parent `DirectRLEnv` reset contract: scene reset, reset events, noise reset, and optionally episode length reset.
2. `_launch_inhand_reset_prepare(...)`
   - Runs fused Warp kernels that prepare goal rotation, object pose/velocity, hand joint pos/vel, target tensors, episode length reset, success clearing, and reset metrics inputs.
   - This is the only part captured by the optional reset CUDA graph.
3. `_apply_inhand_reset_to_sim(...)`
   - Applies the prepared buffers through the asset writer APIs:
     - object root pose
     - object root velocity
     - hand joint target
     - hand joint position
     - hand joint velocity
   - Then calls `sim.forward()`, refreshes Warp state inputs, and recomputes intermediate values.

The `_apply_inhand_reset_to_sim()` ordering matches the old task-specific reset write order. The only meaningful ordering difference is that `successes[env] = 0` now happens inside the prepare kernel before sim writes. That is safe because parent reset/events/noise and reset success accounting happen before the clear, and the writer APIs do not depend on `successes`.

## Correctness Status

I did not run the full Isaac Lab repository test suite. The focused suites below passed.

Commands run with:

```bash
env -u VIRTUAL_ENV CONDA_PREFIX=/home/rshahid/miniconda3/envs/env_isaaclab \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ./isaaclab.sh -p ...
```

Passed:

- `-m py_compile source/isaaclab/isaaclab/envs/direct_rl_env.py source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py source/isaaclab/isaaclab/envs/cuda_graph.py source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py source/isaaclab/test/envs/test_cuda_graph_replay_guard.py source/isaaclab_newton/test/assets/test_rigid_object_reset_kitless.py`
- `git diff --check`
- `-m pytest source/isaaclab_tasks/test/test_inhand_reset_cuda_graph.py -q --tb=short`
  - `41 passed, 159 warnings in 108.36s`
- `-m pytest source/isaaclab/test/envs/test_cuda_graph_replay_guard.py -q`
  - `5 passed in 2.08s`
- `-m pytest source/isaaclab_newton/test/assets/test_rigid_object_reset_kitless.py -q --tb=short`
  - `2 passed, 37 warnings in 21.53s`
- `-m pytest source/isaaclab/test/utils/test_wrench_composer.py -q --tb=short`
  - `366 passed in 11.95s`

Additional isolated/manual checks:

- Two PhysX compatibility checks for in-hand Warp buffer stability and PhysX reward/done reference matching passed when run individually.
- The same PhysX checks caused an ordering/lifetime problem when appended after the full Newton in-hand test file, so they were not committed into the default test file. Do not treat PhysX graph/reset compatibility as fully validated from this work.
- Earlier grouped `warp_dones` selection had one transient `SIGSEGV`; the individual tests and final full Newton in-hand file passed afterward.

Confidence:

- High for the covered Newton/CUDA in-hand reset, dones, rewards, guard, and wrench-reset paths.
- Not 100% global correctness. Full repo tests and broader task coverage were not run.

## Trace

Latest generated trace:

- `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.nsys-rep`
- `/home/rshahid/Projects/isaac/reset_cuda_graph_force_fused_rewards.sqlite`
- `/tmp/nsys_reset_cuda_graph_force_fused_rewards.log`
- `/tmp/nsys_reset_cuda_graph_force_fused_rewards_json/benchmark_non_rl_Isaac-Repose-Cube-Allegro-Direct-v0_2026-05-01_04-05-59.json`

Trace settings:

- Task: `Isaac-Repose-Cube-Allegro-Direct-v0`
- Backend: Newton
- Renderer: non-vision/headless
- Envs: 1024
- Frames: 80
- Capture frames: 30-60
- `env.reset_cuda_graph=force`

Sanity from `nsys stats`:

- `cudaGraphLaunch_v10000`: 150 calls
- `cudaLaunchKernel`: 3810 calls
- `cudaStreamSynchronize` dominated CUDA API time
- NVTX reset ranges were present, including `env.step:_reset_idx` and `env.step:reset_buf.nonzero`

## Known Limitations

- `_reset_idx_common(...)` is still the parent reset body and still uses exact `env_ids`. Its scene reset, reset events, and noise reset work are not fused.
- Reset CUDA graph capture only covers pure reset-preparation kernels. Asset writer APIs and `sim.forward()` remain outside capture because they update Python-side lazy-buffer/FK state.
- Reward computation is intentionally two Warp launches, not one, because `consecutive_successes` depends on a grid-wide reduction. A single parallel kernel would either race or serialize.
- Goal marker visualization is skipped in non-vision headless paths and kept only when renderer/visualizer state can observe it.
- The new `DirectRLEnv._reset_idx_from_mask()` hook avoids eager `nonzero()` only after a subclass opts in. The current in-hand path does not yet consume the mask in `step()` because exact-env side effects still need careful handling.

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
4. Only then override `_reset_idx_from_mask()` for in-hand. Until that is done, falling back to `nonzero()` is the safer behavior.

For broader task support:

- Keep the base hook and reset-preparation API task-agnostic.
- Add task-specific fused reset cores only where the task can prove all reset side effects are either mask-safe or still correctly handled by exact-ID fallback.
- Avoid maintaining duplicate Torch and Warp production paths. Keep Torch/reference implementations only in tests where possible.
