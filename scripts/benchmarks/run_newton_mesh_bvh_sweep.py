# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run and plot an IsaacLab-only Newton mesh BVH backend sweep.

This is an experiment helper for comparing Newton/Warp's default GPU mesh BVH
backend (LBVH) against cuBQL without requiring a local Newton source change.
The backend overrides use ``env.sim.physics.mesh_bvh_constructor=<name>``,
which is handled by the experiment-only IsaacLab monkey patch in
``NewtonManager``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    task_id: str
    presets: str


TASKS = {
    "dexsuite_kuka_allegro_lift_rgb64": TaskSpec(
        name="dexsuite_kuka_allegro_lift_rgb64",
        title="Dexsuite Kuka Allegro Lift RGB64",
        task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
        presets="cube,single_camera,newton_mjwarp,newton_renderer,rgb64",
    ),
    "shadow_vision_rgb": TaskSpec(
        name="shadow_vision_rgb",
        title="Shadow Vision RGB",
        task_id="Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0",
        presets="newton_mjwarp,newton_renderer,rgb",
    ),
}

BACKENDS = {
    "default": {
        "label": "default LBVH BVH",
        "override": "env.sim.physics.mesh_bvh_constructor=lbvh",
    },
    "cubql": {
        "label": "cuBQL BVH",
        "override": "env.sim.physics.mesh_bvh_constructor=cubql",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_metric(benchmark_json: Path, measurement_suffix: str) -> float:
    data = json.loads(benchmark_json.read_text())
    for phase in data:
        for measurement in phase.get("measurements", []):
            if measurement.get("name", "").endswith(measurement_suffix):
                return float(measurement["value"])
    raise KeyError(f"Could not find measurement ending with {measurement_suffix!r} in {benchmark_json}")


def _find_benchmark_json(output_dir: Path, task_id: str) -> Path | None:
    matches = sorted(output_dir.rglob(f"benchmark_non_rl_{task_id}_*.json"))
    return matches[-1] if matches else None


def _read_log_tail(log_path: Path, line_count: int = 80) -> str:
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-line_count:])


def _run_one(
    *,
    repo_root: Path,
    output_dir: Path,
    task: TaskSpec,
    backend_name: str,
    num_envs: int,
    num_frames: int,
    seed: int,
    trial: int,
    dry_run: bool,
) -> dict[str, Any]:
    backend = BACKENDS[backend_name]
    run_dir = output_dir / f"{task.name}_envs_{num_envs}_{backend_name}_trial_{trial}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    command = [
        str(repo_root / "isaaclab.sh"),
        "-p",
        str(repo_root / "scripts/benchmarks/benchmark_non_rl.py"),
        f"--task={task.task_id}",
        "--headless",
        "--enable_cameras",
        f"--num_envs={num_envs}",
        f"--num_frames={num_frames}",
        f"--seed={seed}",
        "--benchmark_backend",
        "json",
        "--output_path",
        str(run_dir),
        f"presets={task.presets}",
    ]
    if backend["override"] is not None:
        command.append(backend["override"])

    record: dict[str, Any] = {
        "task_name": task.name,
        "task_id": task.task_id,
        "presets": task.presets,
        "backend": backend_name,
        "backend_label": backend["label"],
        "num_envs": num_envs,
        "num_frames": num_frames,
        "seed": seed,
        "trial": trial,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
    }

    if dry_run:
        record.update({"status": "dry_run", "returncode": None})
        print(" ".join(command))
        return record

    print(f"[RUN] {task.name} envs={num_envs} backend={backend_name} trial={trial}")
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_path.open("w") as log_file:
        result = subprocess.run(command, cwd=repo_root, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    ended_at = datetime.now().isoformat(timespec="seconds")

    benchmark_json = _find_benchmark_json(run_dir, task.task_id)
    status = "passed" if result.returncode == 0 and benchmark_json is not None else "failed"
    record.update(
        {
            "status": status,
            "returncode": result.returncode,
            "started_at": started_at,
            "ended_at": ended_at,
            "benchmark_json": str(benchmark_json) if benchmark_json else None,
        }
    )
    if status == "failed":
        if result.returncode != 0:
            record["failure_reason"] = f"benchmark command exited with return code {result.returncode}"
        else:
            record["failure_reason"] = f"benchmark JSON not found under {run_dir}"
        record["log_tail"] = _read_log_tail(log_path)
    if benchmark_json is not None:
        record["metrics"] = {
            "mean_environment_step_fps": _extract_metric(benchmark_json, "Mean Environment step FPS"),
            "mean_environment_step_effective_fps": _extract_metric(
                benchmark_json,
                "Mean Environment step effective FPS",
            ),
            "mean_environment_step_time_ms": _extract_metric(benchmark_json, "Mean Environment step times"),
            "min_environment_step_time_ms": _extract_metric(benchmark_json, "Min Environment step times"),
        }
    return record


def _aggregate(summary: dict[str, Any]) -> None:
    aggregates: list[dict[str, Any]] = []
    for task_name in sorted({run["task_name"] for run in summary["runs"]}):
        for num_envs in summary["num_envs"]:
            for backend_name in summary["backends"]:
                values = [
                    run["metrics"]["mean_environment_step_effective_fps"]
                    for run in summary["runs"]
                    if run.get("status") == "passed"
                    and run["task_name"] == task_name
                    and run["num_envs"] == num_envs
                    and run["backend"] == backend_name
                ]
                if values:
                    aggregates.append(
                        {
                            "task_name": task_name,
                            "num_envs": num_envs,
                            "backend": backend_name,
                            "backend_label": BACKENDS[backend_name]["label"],
                            "trials": len(values),
                            "mean_environment_step_effective_fps": mean(values),
                            "min_environment_step_effective_fps": min(values),
                            "max_environment_step_effective_fps": max(values),
                        }
                    )
    summary["aggregates"] = aggregates


def _plot_summary(summary: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("[WARN] matplotlib is not installed; skipping plots and writing JSON summary only.")
        summary["plots"] = []
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []

    colors = {"default": "#4C78A8", "cubql": "#F58518"}
    markers = {"default": "o", "cubql": "s"}
    linestyles = {"default": "-", "cubql": "--"}

    for task in summary["tasks"]:
        task_name = task["name"]
        envs = list(summary["num_envs"])
        by_backend: dict[str, list[float | None]] = {}
        for backend_name in summary["backends"]:
            series: list[float | None] = []
            for num_envs in envs:
                row = next(
                    (
                        item
                        for item in summary.get("aggregates", [])
                        if item["task_name"] == task_name
                        and item["backend"] == backend_name
                        and item["num_envs"] == num_envs
                    ),
                    None,
                )
                series.append(row["mean_environment_step_effective_fps"] if row else None)
            by_backend[backend_name] = series

        baseline = by_backend["default"]
        pct_by_backend: dict[str, list[float | None]] = {}
        for backend_name, series in by_backend.items():
            pct_values: list[float | None] = []
            for value, base in zip(series, baseline, strict=True):
                pct_values.append(None if value is None or base in (None, 0.0) else (value / base - 1.0) * 100.0)
            pct_by_backend[backend_name] = pct_values

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10.8, 7.6),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.05]},
        )
        fig.suptitle(task["title"], fontsize=23, fontweight="bold", y=0.975)
        fig.text(
            0.5,
            0.92,
            "Top: effective FPS. Bottom: speedup vs default BVH baseline.",
            ha="center",
            va="center",
            fontsize=14,
            color="#555555",
        )

        handles = []
        labels = []
        for backend_name in summary["backends"]:
            y = by_backend[backend_name]
            handle = axes[0].plot(
                envs,
                y,
                color=colors[backend_name],
                marker=markers[backend_name],
                linestyle=linestyles[backend_name],
                linewidth=2.4,
                markersize=7.5,
                label=BACKENDS[backend_name]["label"],
            )[0]
            axes[1].plot(
                envs,
                pct_by_backend[backend_name],
                color=colors[backend_name],
                marker=markers[backend_name],
                linestyle=linestyles[backend_name],
                linewidth=2.4,
                markersize=7.5,
            )
            handles.append(handle)
            labels.append(BACKENDS[backend_name]["label"])

        axes[0].set_ylabel("Effective FPS", fontsize=12)
        axes[1].set_ylabel("% vs default BVH", fontsize=12)
        axes[1].set_xlabel("Number of environments", fontsize=12)
        axes[1].axhline(0.0, color="#888888", linewidth=1.0)

        for axis in axes:
            axis.grid(True, which="major", axis="both", color="#D8D8D8", linewidth=0.8)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.tick_params(axis="both", labelsize=11)
            axis.set_xticks(envs)
            for spine in axis.spines.values():
                spine.set_color("#A0A0A0")

        axes[0].legend(handles, labels, loc="upper left", frameon=True, fontsize=12)
        fig.text(
            0.965,
            0.025,
            f"source: {Path(summary['summary_path']).name}",
            ha="right",
            fontsize=11,
            color="#777777",
        )
        fig.tight_layout(rect=[0.04, 0.045, 0.98, 0.89])

        stem = f"{task_name}_effective_fps_pct_vs_default_bvh"
        for suffix in ("png", "svg"):
            path = plots_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=160)
            plot_paths.append(str(path))
        plt.close(fig)

    summary["plots"] = plot_paths
    return plot_paths


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for raw results and plots.")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKENDS), default=["default", "cubql"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write summary without running.")
    parser.add_argument("--plot-only", type=Path, default=None, help="Create plots from an existing summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()

    if args.plot_only is not None:
        summary_path = args.plot_only.resolve()
        summary = json.loads(summary_path.read_text())
        summary["summary_path"] = str(summary_path)
        _aggregate(summary)
        plot_paths = _plot_summary(summary, summary_path.parent)
        _write_summary(summary, summary_path)
        if plot_paths:
            print(f"[OK] Wrote plots under {summary_path.parent / 'plots'}")
        else:
            print("[OK] Plot generation skipped.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or repo_root / "hdc/benchmarks" / f"newton_mesh_bvh_sweep_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "num_envs": args.num_envs,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "trials": args.trials,
        "backends": args.backends,
        "tasks": [TASKS[name].__dict__ for name in args.tasks],
        "runs": [],
    }

    for task_name in args.tasks:
        task = TASKS[task_name]
        for num_envs in args.num_envs:
            for backend_name in args.backends:
                for trial in range(args.trials):
                    record = _run_one(
                        repo_root=repo_root,
                        output_dir=output_root,
                        task=task,
                        backend_name=backend_name,
                        num_envs=num_envs,
                        num_frames=args.num_frames,
                        seed=args.seed + trial,
                        trial=trial,
                        dry_run=args.dry_run,
                    )
                    summary["runs"].append(record)
                    _aggregate(summary)
                    _write_summary(summary, summary_path)

    _aggregate(summary)
    plot_paths = []
    if not args.dry_run:
        plot_paths = _plot_summary(summary, output_root)
    _write_summary(summary, summary_path)

    failed = [run for run in summary["runs"] if run["status"] == "failed"]
    print(f"[OK] Summary: {summary_path}")
    if plot_paths:
        print(f"[OK] Plots: {output_root / 'plots'}")
    if failed:
        print(f"[FAIL] {len(failed)} run(s) failed. Inspect each run's log_path, failure_reason, and log_tail in summary.json.")
        for run in failed[:8]:
            print(
                "[FAIL] "
                f"{run['task_name']} envs={run['num_envs']} backend={run['backend']} "
                f"reason={run.get('failure_reason', 'unknown')} log={run['log_path']}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
