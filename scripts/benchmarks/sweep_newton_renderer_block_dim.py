#!/usr/bin/env python3

"""Run Newton Warp renderer block-dim benchmark sweeps and collect summary JSON.

This script intentionally shells out to ``scripts/benchmarks/benchmark_non_rl.py`` for each run so it can be launched
once and left unattended. It writes a live-updated ``summary.json`` after every run, so partial results are preserved
if a long sweep is interrupted.
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
    block_dim_override: str


TASKS = (
    TaskSpec(
        name="dexsuite_kuka_allegro_lift_rgb64",
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        presets="cube,single_camera,newton,newton_renderer,rgb64",
        block_dim_override="env.scene.base_camera.renderer_cfg.block_dim",
    ),
    TaskSpec(
        name="shadow_vision_rgb",
        task_id="Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
        presets="newton,newton_renderer,rgb",
        block_dim_override="env.tiled_camera.renderer_cfg.block_dim",
    ),
    TaskSpec(
        name="cartpole_camera_rgb",
        task_id="Isaac-Cartpole-Camera-Presets-Direct-v0",
        presets="newton,newton_renderer,rgb",
        block_dim_override="env.tiled_camera.renderer_cfg.block_dim",
    ),
)


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


def _build_command(
    repo_root: Path,
    task: TaskSpec,
    num_envs: int,
    block_dim: int,
    num_frames: int,
    output_dir: Path,
    benchmark_backend: str,
) -> list[str]:
    return [
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(repo_root / "scripts/benchmarks/benchmark_non_rl.py"),
        f"--task={task.task_id}",
        "--headless",
        "--enable_cameras",
        f"--num_envs={num_envs}",
        f"--num_frames={num_frames}",
        "--benchmark_backend",
        benchmark_backend,
        "--output_path",
        str(output_dir),
        f"presets={task.presets}",
        f"{task.block_dim_override}={block_dim}",
    ]


def _run_one(
    repo_root: Path,
    task: TaskSpec,
    num_envs: int,
    block_dim: int,
    num_frames: int,
    output_root: Path,
    benchmark_backend: str,
    dry_run: bool,
) -> dict[str, Any]:
    run_name = f"{task.name}_envs_{num_envs}_block_dim_{block_dim}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    command = _build_command(repo_root, task, num_envs, block_dim, num_frames, run_dir, benchmark_backend)

    result: dict[str, Any] = {
        "task_name": task.name,
        "task_id": task.task_id,
        "presets": task.presets,
        "num_envs": num_envs,
        "block_dim": block_dim,
        "num_frames": num_frames,
        "block_dim_override": task.block_dim_override,
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
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--block-dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--benchmark-backend", default="json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    output_root = args.output_root or repo_root / "hdc/benchmarks" / f"newton_renderer_block_dim_sweep_{_timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": repo_root,
        "output_root": output_root,
        "num_envs": args.num_envs,
        "block_dims": args.block_dims,
        "num_frames": args.num_frames,
        "benchmark_backend": args.benchmark_backend,
        "tasks": [asdict(task) for task in TASKS],
        "runs": [],
    }
    _write_summary(summary_path, summary)

    total_runs = len(TASKS) * len(args.num_envs) * len(args.block_dims)
    run_index = 0
    for task in TASKS:
        for num_envs in args.num_envs:
            for block_dim in args.block_dims:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] task={task.name} num_envs={num_envs} block_dim={block_dim}",
                    flush=True,
                )
                result = _run_one(
                    repo_root=repo_root,
                    task=task,
                    num_envs=num_envs,
                    block_dim=block_dim,
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
