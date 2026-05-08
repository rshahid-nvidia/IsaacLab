# Newton Warp Renderer Block-Dim Test Branch

This branch exposes `NewtonWarpRendererCfg.block_dim` from IsaacLab without requiring a patched Newton checkout.
IsaacLab applies the value by temporarily monkey-patching `warp.launch` only around the Newton renderer update and only
for Newton's `render_megakernel`.

## Branch

```bash
git clone https://github.com/rshahid-nvidia/IsaacLab.git
cd IsaacLab
git checkout rshahid/newton-renderer-blockdim-monkeypatch
```

Do not add a local Newton checkout to `PYTHONPATH` for this experiment. The point of this branch is to use the Newton
version already installed in the IsaacLab environment.

## Environment Setup

Use the normal IsaacLab setup for the target machine. A minimal conda setup is:

```bash
conda create -n env_isaaclab_blockdim python=3.12 -y
conda activate env_isaaclab_blockdim
./isaaclab.sh --install
```

Quick import check:

```bash
./isaaclab.sh -p - <<'PY'
import newton
import warp as wp
from isaaclab_newton.renderers import NewtonWarpRendererCfg
print("newton:", newton.__file__)
print("warp:", wp.__version__)
print("default block_dim:", NewtonWarpRendererCfg().block_dim)
PY
```

`block_dim=0` preserves Newton/Warp's default launch configuration. Positive values such as `64` or `128` are injected
into the renderer megakernel launch.

This branch also exposes Newton's render traversal knobs:

- `renderer_cfg.render_order=0`: `PIXEL_PRIORITY` traversal, Newton's default.
- `renderer_cfg.render_order=2`: `TILED` traversal.
- `renderer_cfg.tile_width=<N>` / `renderer_cfg.tile_height=<N>`: tile size for `TILED`; both must divide the camera
  resolution exactly.

## Benchmark Commands

Shadow vision:

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
    --task=Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 \
    --headless --enable_cameras --num_envs=4096 --num_frames=100 \
    --benchmark_backend json --output_path hdc/benchmarks/shadow_rgb_bd128 \
    presets=newton,newton_renderer,rgb \
    env.tiled_camera.renderer_cfg.block_dim=128
```

Cartpole camera:

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
    --task=Isaac-Cartpole-Camera-Presets-Direct-v0 \
    --headless --enable_cameras --num_envs=4096 --num_frames=100 \
    --benchmark_backend json --output_path hdc/benchmarks/cartpole_rgb_bd128 \
    presets=newton,newton_renderer,rgb \
    env.tiled_camera.renderer_cfg.block_dim=128
```

Dexsuite Kuka Allegro:

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
    --task=Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
    --headless --enable_cameras --num_envs=4096 --num_frames=100 \
    --benchmark_backend json --output_path hdc/benchmarks/dexsuite_depth64_bd64 \
    presets=cube,single_camera,newton,newton_renderer,depth64 \
    env.scene.base_camera.renderer_cfg.block_dim=64
```

Repeat each command with `block_dim=0`, `64`, and `128` for the sweep.

## Run the Full Sweep Script

This branch also includes a one-shot sweep script for the three camera workloads:

```bash
./scripts/benchmarks/sweep_newton_renderer_block_dim.py
```

By default it runs:

- tasks: Dexsuite Kuka Allegro Lift, Shadow vision benchmark, Cartpole camera presets
- `num_envs`: `1024 2048 4096 8192`
- `block_dim`: `64 128 256`
- frames per run: `100`

The script writes per-run benchmark JSON files plus a live-updated summary at:

```text
hdc/benchmarks/newton_renderer_block_dim_sweep_<timestamp>/summary.json
```

Dexsuite uses `presets=cube,single_camera,newton,newton_renderer,rgb64` because its camera presets are
resolution-specific and the scene camera must be explicitly enabled. Shadow and Cartpole use
`presets=newton,newton_renderer,rgb`.

## Run the Render-Order / Tile-Size Sweep Script

To compare `PIXEL_PRIORITY` against `TILED` while also sweeping `block_dim`, tile width, tile height, and Dexsuite
camera resolution:

```bash
./scripts/benchmarks/sweep_newton_renderer_render_order_tiles.py
```

By default it runs:

- Dexsuite Kuka Allegro Lift with `rgb64`, `rgb128`, and `rgb256`
- Shadow vision RGB and Cartpole camera RGB
- `num_envs`: `1024 2048 4096 8192`
- `block_dim`: `64 128 256`
- render orders: `pixel_priority tiled`
- tile pairs: `8x8 10x10 16x8 16x16 20x10 20x20 25x25 32x32 40x40 64x64`
- frames per run: `100`

Invalid tile sizes are recorded under `skipped_invalid_tiles` in the summary instead of being executed. For example,
Cartpole's `100x100` camera skips `16x8` because `100 % 16 != 0`.

If the existing `PIXEL_PRIORITY` sweep is already sufficient and you only want the tiled add-on:

```bash
./scripts/benchmarks/sweep_newton_renderer_render_order_tiles.py --render-orders tiled
```

To override the tile set:

```bash
./scripts/benchmarks/sweep_newton_renderer_render_order_tiles.py \
    --render-orders tiled \
    --tile-pairs 8x8 16x8 16x16 32x32
```

The script writes:

```text
hdc/benchmarks/newton_renderer_render_order_tile_sweep_<timestamp>/summary.json
```

## Optional NSYS Trace

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --output hdc/benchmarks/cartpole_rgb_bd128_nsys \
    ./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py \
        --task=Isaac-Cartpole-Camera-Presets-Direct-v0 \
        --headless --enable_cameras --num_envs=4096 --num_frames=100 \
        --benchmark_backend json --output_path hdc/benchmarks/cartpole_rgb_bd128_nsys_json \
        presets=newton,newton_renderer,rgb \
        env.tiled_camera.renderer_cfg.block_dim=128
```
