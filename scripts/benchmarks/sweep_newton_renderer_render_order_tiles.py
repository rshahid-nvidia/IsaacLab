#!/usr/bin/env python3

"""Sweep Newton Warp renderer launch shape plus render order / tile size.

This is intentionally a plain subprocess driver around ``benchmark_non_rl.py``. It writes a live-updated
``summary.json`` after every run, so partial results survive interrupts.

By default this runs both render orders:

* ``pixel_priority``: block-dim sweep only.
* ``tiled``: block-dim sweep crossed with tile-width / tile-height sweep.

If you already have pixel-priority results, pass ``--render-orders tiled`` to only run the tiled render-order sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: str
    presets: str
    renderer_cfg_path: str
    width: int
    height: int


@dataclass(frozen=True)
class RenderOrderSpec:
    name: str
    value: int
    uses_tiles: bool


@dataclass(frozen=True)
class Experiment:
    task: TaskSpec
    render_order: RenderOrderSpec
    num_envs: int
    block_dim: int
    tile_width: int | None
    tile_height: int | None


TASKS = (
    TaskSpec(
        name="dexsuite_kuka_allegro_lift_rgb64",
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        presets="cube,single_camera,newton,newton_renderer,rgb64",
        renderer_cfg_path="env.scene.base_camera.renderer_cfg",
        width=64,
        height=64,
    ),
    TaskSpec(
        name="dexsuite_kuka_allegro_lift_rgb128",
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        presets="cube,single_camera,newton,newton_renderer,rgb128",
        renderer_cfg_path="env.scene.base_camera.renderer_cfg",
        width=128,
        height=128,
    ),
    TaskSpec(
        name="dexsuite_kuka_allegro_lift_rgb256",
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        presets="cube,single_camera,newton,newton_renderer,rgb256",
        renderer_cfg_path="env.scene.base_camera.renderer_cfg",
        width=256,
        height=256,
    ),
    TaskSpec(
        name="shadow_vision_rgb",
        task_id="Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
        presets="newton,newton_renderer,rgb",
        renderer_cfg_path="env.tiled_camera.renderer_cfg",
        width=120,
        height=120,
    ),
    TaskSpec(
        name="cartpole_camera_rgb",
        task_id="Isaac-Cartpole-Camera-Presets-Direct-v0",
        presets="newton,newton_renderer,rgb",
        renderer_cfg_path="env.tiled_camera.renderer_cfg",
        width=100,
        height=100,
    ),
)

RENDER_ORDERS = {
    "pixel": RenderOrderSpec("pixel_priority", 0, False),
    "pixel_priority": RenderOrderSpec("pixel_priority", 0, False),
    "tiled": RenderOrderSpec("tiled", 2, True),
    "tiled_rendering": RenderOrderSpec("tiled", 2, True),
}


def _task_names() -> str:
    return ", ".join(task.name for task in TASKS)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _read_measurements(json_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(json_path.read_text())
    measurements = {}
    for phase in data:
        phase_name = phase.get("phase_name", "")
        for measurement in phase.get("measurements", []):
            name = measurement.get("name")
            if not name:
                continue
            measurements[name] = {
                "name": name,
                "phase": phase_name,
                "value": measurement.get("value"),
                "unit": measurement.get("unit"),
                "type": measurement.get("type"),
            }
    return measurements


def _metric_value(measurements: dict[str, dict[str, Any]], name: str) -> Any:
    entry = measurements.get(name)
    if entry is None:
        return None
    return entry.get("value")


def _latest_benchmark_json(output_dir: Path) -> Path | None:
    json_files = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    return json_files[-1] if json_files else None


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _is_valid_tile(task: TaskSpec, tile_width: int, tile_height: int) -> bool:
    return task.width % tile_width == 0 and task.height % tile_height == 0


def _normalize_render_orders(names: list[str]) -> list[RenderOrderSpec]:
    specs: list[RenderOrderSpec] = []
    seen = set()
    for name in names:
        key = name.strip().lower()
        if key not in RENDER_ORDERS:
            valid = ", ".join(sorted(RENDER_ORDERS))
            raise ValueError(f"Unknown render order {name!r}. Valid values: {valid}")
        spec = RENDER_ORDERS[key]
        if spec.name not in seen:
            specs.append(spec)
            seen.add(spec.name)
    return specs


def _normalize_tasks(names: list[str] | None) -> list[TaskSpec]:
    if not names:
        return list(TASKS)

    task_by_name = {task.name: task for task in TASKS}
    tasks: list[TaskSpec] = []
    seen = set()
    for name in names:
        if name not in task_by_name:
            raise ValueError(f"Unknown task {name!r}. Valid values: {_task_names()}")
        if name in seen:
            continue
        tasks.append(task_by_name[name])
        seen.add(name)
    return tasks


def _parse_tile_pair(value: str) -> tuple[int, int]:
    try:
        width_str, height_str = value.lower().split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise ValueError(f"Expected tile pair formatted as WIDTHxHEIGHT. Received: {value!r}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Tile dimensions must be positive. Received: {value!r}")
    return width, height


def _normalize_tile_pairs(values: list[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen = set()
    for value in values:
        pair = _parse_tile_pair(value)
        if pair in seen:
            continue
        pairs.append(pair)
        seen.add(pair)
    return pairs


def _make_experiments(args: argparse.Namespace) -> tuple[list[Experiment], list[dict[str, Any]]]:
    tasks = _normalize_tasks(args.tasks)
    render_orders = _normalize_render_orders(args.render_orders)
    tile_pairs = _normalize_tile_pairs(args.tile_pairs)
    experiments: list[Experiment] = []
    skipped: list[dict[str, Any]] = []

    for task in tasks:
        for num_envs in args.num_envs:
            for block_dim in args.block_dims:
                for render_order in render_orders:
                    if not render_order.uses_tiles:
                        experiments.append(
                            Experiment(
                                task=task,
                                render_order=render_order,
                                num_envs=num_envs,
                                block_dim=block_dim,
                                tile_width=None,
                                tile_height=None,
                            )
                        )
                        continue

                    for tile_width, tile_height in tile_pairs:
                        if not _is_valid_tile(task, tile_width, tile_height):
                            skipped.append(
                                {
                                    "task_name": task.name,
                                    "task_id": task.task_id,
                                    "num_envs": num_envs,
                                    "block_dim": block_dim,
                                    "render_order": render_order.name,
                                    "render_order_value": render_order.value,
                                    "tile_width": tile_width,
                                    "tile_height": tile_height,
                                    "reason": f"tile must divide camera resolution {task.width}x{task.height}",
                                }
                            )
                            continue

                        experiments.append(
                            Experiment(
                                task=task,
                                render_order=render_order,
                                num_envs=num_envs,
                                block_dim=block_dim,
                                tile_width=tile_width,
                                tile_height=tile_height,
                            )
                        )

    return experiments, skipped


def _build_command(
    repo_root: Path,
    experiment: Experiment,
    num_frames: int,
    output_dir: Path,
    benchmark_backend: str,
) -> list[str]:
    renderer_cfg_path = experiment.task.renderer_cfg_path
    command = [
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(repo_root / "scripts/benchmarks/benchmark_non_rl.py"),
        f"--task={experiment.task.task_id}",
        "--headless",
        "--enable_cameras",
        f"--num_envs={experiment.num_envs}",
        f"--num_frames={num_frames}",
        "--benchmark_backend",
        benchmark_backend,
        "--output_path",
        str(output_dir),
        f"presets={experiment.task.presets}",
        f"{renderer_cfg_path}.block_dim={experiment.block_dim}",
        f"{renderer_cfg_path}.render_order={experiment.render_order.value}",
    ]
    if experiment.render_order.uses_tiles:
        command.extend(
            [
                f"{renderer_cfg_path}.tile_width={experiment.tile_width}",
                f"{renderer_cfg_path}.tile_height={experiment.tile_height}",
            ]
        )
    return command


def _run_name(experiment: Experiment) -> str:
    name = (
        f"{experiment.task.name}_envs_{experiment.num_envs}"
        f"_order_{experiment.render_order.name}_block_dim_{experiment.block_dim}"
    )
    if experiment.render_order.uses_tiles:
        name += f"_tile_{experiment.tile_width}x{experiment.tile_height}"
    return name


def _run_one(
    repo_root: Path,
    experiment: Experiment,
    num_frames: int,
    output_root: Path,
    benchmark_backend: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_name = _run_name(experiment)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    command = _build_command(repo_root, experiment, num_frames, run_dir, benchmark_backend)

    result: dict[str, Any] = {
        "task_name": experiment.task.name,
        "task_id": experiment.task.task_id,
        "presets": experiment.task.presets,
        "camera_resolution": [experiment.task.width, experiment.task.height],
        "num_envs": experiment.num_envs,
        "block_dim": experiment.block_dim,
        "render_order": experiment.render_order.name,
        "render_order_value": experiment.render_order.value,
        "tile_width": experiment.tile_width,
        "tile_height": experiment.tile_height,
        "num_frames": num_frames,
        "renderer_cfg_path": experiment.task.renderer_cfg_path,
        "output_dir": run_dir,
        "log_path": log_path,
        "command": command,
        "status": "dry_run" if dry_run else "running",
        "returncode": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "ended_at": None,
        "elapsed_s": None,
        "benchmark_json": None,
        "metrics": {},
    }

    if dry_run:
        return result

    start = time.perf_counter()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            check=False,
        )

    result["returncode"] = process.returncode
    result["ended_at"] = datetime.now().isoformat(timespec="seconds")
    result["elapsed_s"] = time.perf_counter() - start
    result["status"] = "passed" if process.returncode == 0 else "failed"

    benchmark_json = _latest_benchmark_json(run_dir)
    if benchmark_json is None:
        result["status"] = "failed_no_json" if process.returncode == 0 else result["status"]
        return result

    measurements = _read_measurements(benchmark_json)
    selected_measurement_names = (
        "benchmark_non_rl runtime Mean Environment step FPS",
        "benchmark_non_rl runtime Mean Environment step effective FPS",
        "benchmark_non_rl runtime Mean Environment step times",
        "benchmark_non_rl runtime Min Environment step times",
        "benchmark_non_rl runtime Max Environment step times",
    )
    result["benchmark_json"] = benchmark_json
    result["selected_measurements"] = [
        measurements[name] for name in selected_measurement_names if name in measurements
    ]
    result["metrics"] = {
        "mean_environment_step_fps": _metric_value(
            measurements, "benchmark_non_rl runtime Mean Environment step FPS"
        ),
        "mean_environment_step_effective_fps": _metric_value(
            measurements, "benchmark_non_rl runtime Mean Environment step effective FPS"
        ),
        "mean_environment_step_time_ms": _metric_value(
            measurements, "benchmark_non_rl runtime Mean Environment step times"
        ),
        "min_environment_step_time_ms": _metric_value(
            measurements, "benchmark_non_rl runtime Min Environment step times"
        ),
        "max_environment_step_time_ms": _metric_value(
            measurements, "benchmark_non_rl runtime Max Environment step times"
        ),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Task names to run. Defaults to all. Valid values: {_task_names()}",
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--block-dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--render-orders",
        nargs="+",
        default=["pixel_priority", "tiled"],
        help="Render orders to run. Use 'tiled' to only add tiled results if pixel-priority already exists.",
    )
    parser.add_argument(
        "--tile-pairs",
        nargs="+",
        default=["8x8", "10x10", "16x8", "16x16", "20x10", "20x20", "25x25", "32x32", "40x40", "64x64"],
        help=(
            "Tile sizes for TILED render order, formatted as WIDTHxHEIGHT. Invalid pairs for a camera resolution are "
            "skipped and recorded in summary.json."
        ),
    )
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--benchmark-backend", default="json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    output_root = (
        args.output_root
        or repo_root / "hdc/benchmarks" / f"newton_renderer_render_order_tile_sweep_{_timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"

    experiments, skipped = _make_experiments(args)
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": repo_root,
        "output_root": output_root,
        "num_envs": args.num_envs,
        "block_dims": args.block_dims,
        "render_orders": [spec.name for spec in _normalize_render_orders(args.render_orders)],
        "tile_pairs": [f"{width}x{height}" for width, height in _normalize_tile_pairs(args.tile_pairs)],
        "num_frames": args.num_frames,
        "benchmark_backend": args.benchmark_backend,
        "tasks": [asdict(task) for task in _normalize_tasks(args.tasks)],
        "planned_run_count": len(experiments),
        "skipped_invalid_tiles": skipped,
        "runs": [],
    }
    _write_summary(summary_path, summary)

    total_runs = len(experiments)
    for run_index, experiment in enumerate(experiments, start=1):
        tile_label = (
            f" tile={experiment.tile_width}x{experiment.tile_height}"
            if experiment.render_order.uses_tiles
            else ""
        )
        print(
            f"[{run_index}/{total_runs}] task={experiment.task.name} num_envs={experiment.num_envs} "
            f"order={experiment.render_order.name} block_dim={experiment.block_dim}{tile_label}",
            flush=True,
        )
        result = _run_one(
            repo_root=repo_root,
            experiment=experiment,
            num_frames=args.num_frames,
            output_root=output_root,
            benchmark_backend=args.benchmark_backend,
            dry_run=args.dry_run,
        )
        summary["runs"].append(result)
        _write_summary(summary_path, summary)

        if result["status"] not in ("passed", "dry_run") and args.stop_on_failure:
            print(f"Stopping after failure. See {result['log_path']}", file=sys.stderr)
            return 1

    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
