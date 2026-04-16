from __future__ import annotations

import inspect
import json
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import config as config_module


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _display_path(path: Optional[Path], base: Optional[Path] = None) -> Optional[str]:
    if path is None:
        return None
    try:
        if base is not None:
            return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        pass
    return path.name


def _resolve_training_stats_path(episode_dir: Optional[Path]) -> Optional[Path]:
    if episode_dir is None:
        return None

    candidates = [
        episode_dir / "training_stats.json",
        episode_dir.parent / "training_stats.json",
        episode_dir.parent.parent / "training_stats.json" if episode_dir.parent != episode_dir else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def config_snapshot() -> Dict[str, Any]:
    ppo_defaults = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "batch_size": 64,
    }
    try:
        from ppo_agent import PPOAgent

        ppo_signature = inspect.signature(PPOAgent.__init__)
        ppo_defaults = {
            name: parameter.default
            for name, parameter in ppo_signature.parameters.items()
            if parameter.default is not inspect._empty
            and name in ppo_defaults
        }
    except ModuleNotFoundError:
        pass
    return {
        "source_file": str(Path(config_module.__file__).resolve()),
        "environment": {
            "track_length": config_module.TRACK_LENGTH,
            "max_vehicles": config_module.MAX_VEHICLES,
            "max_speed": config_module.MAX_SPEED,
            "max_acceleration": config_module.MAX_ACCELERATION,
            "safety_distance": config_module.SAFETY_DISTANCE,
            "speed_tolerance": config_module.SPEED_TOLERANCE,
        },
        "stations": {
            "loading_positions": list(config_module.LOADING_POSITIONS),
            "unloading_positions": list(config_module.UNLOADING_POSITIONS),
            "loading_station_slots": config_module.LOADING_STATION_SLOTS,
            "unloading_station_slots": config_module.UNLOADING_STATION_SLOTS,
        },
        "cargo": {
            "arrival_interval_min": config_module.ARRIVAL_INTERVAL_MIN,
            "arrival_interval_max": config_module.ARRIVAL_INTERVAL_MAX,
            "cargo_timeout": config_module.CARGO_TIMEOUT,
        },
        "timing": {
            "loading_time": config_module.LOADING_TIME,
            "unloading_time": config_module.UNLOADING_TIME,
            "episode_duration": config_module.EPISODE_DURATION,
            "high_level_decision_interval": config_module.HIGH_LEVEL_DECISION_INTERVAL,
            "low_level_control_interval": config_module.LOW_LEVEL_CONTROL_INTERVAL,
        },
        "rewards": {
            "delivery": config_module.REWARD_DELIVERY,
            "pickup": config_module.REWARD_PICKUP,
            "assignment": config_module.REWARD_ASSIGNMENT,
            "wait_penalty_coeff": config_module.REWARD_WAIT_PENALTY_COEFF,
            "timeout_penalty": config_module.REWARD_TIMEOUT_PENALTY,
            "holding_penalty_coeff": config_module.REWARD_HOLDING_PENALTY_COEFF,
            "safety_violation": config_module.REWARD_SAFETY_VIOLATION,
            "speed_change_penalty": config_module.REWARD_SPEED_CHANGE_PENALTY,
        },
        "learning": {
            "hidden_dim": config_module.HIDDEN_DIM,
            "learning_rate": config_module.LEARNING_RATE,
            "batch_size": config_module.BATCH_SIZE,
            "gamma": config_module.GAMMA,
            "steps_per_update": config_module.STEPS_PER_UPDATE,
            "lr_scheduler_enabled": config_module.LR_SCHEDULER_ENABLED,
            "lr_warmup_episodes": config_module.LR_WARMUP_EPISODES,
            "lr_start_warmup_value": config_module.LR_START_WARMUP_VALUE,
            "lr_final_value": config_module.LR_FINAL_VALUE,
            "ppo_defaults": ppo_defaults,
        },
        "training": {
            "num_episodes_default": config_module.NUM_EPISODES,
            "max_steps_per_episode": config_module.MAX_STEPS_PER_EPISODE,
            "train_frequency": config_module.TRAIN_FREQUENCY,
            "target_update_frequency": config_module.TARGET_UPDATE_FREQUENCY,
            "save_frequency": config_module.SAVE_FREQUENCY,
            "eval_seed": config_module.EVAL_SEED,
            "eval_interval": config_module.EVAL_INTERVAL,
        },
    }


def config_snapshot_summary() -> Dict[str, Any]:
    snapshot = config_snapshot()
    learning = snapshot["learning"]
    return {
        "source_file": snapshot["source_file"],
        "track_length": snapshot["environment"]["track_length"],
        "loading_positions": snapshot["stations"]["loading_positions"],
        "unloading_positions": snapshot["stations"]["unloading_positions"],
        "arrival_window": [
            snapshot["cargo"]["arrival_interval_min"],
            snapshot["cargo"]["arrival_interval_max"],
        ],
        "cargo_timeout": snapshot["cargo"]["cargo_timeout"],
        "episode_duration": snapshot["timing"]["episode_duration"],
        "decision_interval": snapshot["timing"]["high_level_decision_interval"],
        "control_interval": snapshot["timing"]["low_level_control_interval"],
        "hidden_dim": learning["hidden_dim"],
        "learning_rate": learning["learning_rate"],
        "batch_size": learning["batch_size"],
        "gamma": learning["gamma"],
        "steps_per_update": learning["steps_per_update"],
        "scheduler": {
            "enabled": learning["lr_scheduler_enabled"],
            "warmup_episodes": learning["lr_warmup_episodes"],
            "start": learning["lr_start_warmup_value"],
            "end": learning["lr_final_value"],
        },
        "ppo_defaults": learning["ppo_defaults"],
    }


def training_stats_preview(stats_path: Optional[Path], workspace: Path) -> Optional[Dict[str, Any]]:
    if stats_path is None or not stats_path.exists():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    rewards = [float(value) for value in stats.get("episode_rewards", [])]
    completions = [int(value) for value in stats.get("episode_completions", [])]
    actor_losses = [float(value) for value in stats.get("episode_actor_losses", [])]
    critic_losses = [float(value) for value in stats.get("episode_critic_losses", [])]
    entropies = [float(value) for value in stats.get("episode_entropies", [])]
    compact_config = stats.get("config", {})

    return {
        "available": True,
        "path": str(stats_path.resolve()),
        "display_path": _display_path(stats_path, workspace),
        "episode_count": len(rewards),
        "best_avg_reward": stats.get("best_avg_reward"),
        "best_avg_completion": stats.get("best_avg_completion"),
        "best_eval_reward": stats.get("best_eval_reward"),
        "latest_reward": rewards[-1] if rewards else None,
        "latest_completion": completions[-1] if completions else None,
        "latest_actor_loss": actor_losses[-1] if actor_losses else None,
        "latest_critic_loss": critic_losses[-1] if critic_losses else None,
        "latest_entropy": entropies[-1] if entropies else None,
        "config": compact_config,
    }


def build_data_provenance(
    *,
    workspace: Path,
    episode_dir: Optional[Path],
    training_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = config_snapshot_summary()
    return {
        "kind": "seeded_simulation",
        "description": "数据来源为带 seed 的仿真环境，不依赖外部数据集文件。",
        "config_source": {
            "path": snapshot["source_file"],
            "display_path": _display_path(Path(snapshot["source_file"]), workspace),
        },
        "episode_artifacts": {
            "episode_dir": str(episode_dir.resolve()) if episode_dir is not None else None,
            "display_dir": _display_path(episode_dir, workspace) if episode_dir is not None else None,
            "training_stats_available": bool(training_stats),
        },
        "station_layout": {
            "loading_positions": snapshot["loading_positions"],
            "unloading_positions": snapshot["unloading_positions"],
            "track_length": snapshot["track_length"],
        },
        "cargo_generation": {
            "arrival_window_seconds": snapshot["arrival_window"],
            "cargo_timeout_seconds": snapshot["cargo_timeout"],
        },
        "control_timing": {
            "episode_duration_seconds": snapshot["episode_duration"],
            "decision_interval_seconds": snapshot["decision_interval"],
            "control_interval_seconds": snapshot["control_interval"],
        },
    }


def enrich_episode_payload(
    episode: Dict[str, Any],
    *,
    workspace: Path,
    episode_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = _json_clone(episode)
    summary = payload.setdefault("summary", {})
    if episode_dir is None and report_path is not None:
        episode_dir = report_path.parent

    resolved_episode_dir = episode_dir.resolve() if episode_dir is not None else None
    stats_path = _resolve_training_stats_path(resolved_episode_dir)
    stats_preview = training_stats_preview(stats_path, workspace)
    snapshot = config_snapshot_summary()

    summary.setdefault("selected_model_id", summary.get("selected_model_id"))
    summary.setdefault("source_model_path", summary.get("source_model_path"))

    payload["metadata"] = {
        "selected_model_id": summary.get("selected_model_id"),
        "source_model_path": summary.get("source_model_path"),
        "deterministic_inference": summary.get("deterministic_inference"),
        "training_stats": stats_preview
        or {
            "available": False,
            "path": str(stats_path.resolve()) if stats_path is not None else None,
            "display_path": _display_path(stats_path, workspace) if stats_path is not None else None,
        },
        "data_provenance": build_data_provenance(
            workspace=workspace,
            episode_dir=resolved_episode_dir,
            training_stats=stats_preview,
        ),
        "config_snapshot": snapshot,
        "paths": {
            "episode_dir": str(resolved_episode_dir) if resolved_episode_dir is not None else None,
            "episode_dir_display": _display_path(resolved_episode_dir, workspace) if resolved_episode_dir is not None else None,
            "report_path": str(report_path.resolve()) if report_path is not None else None,
            "report_path_display": _display_path(report_path, workspace) if report_path is not None else None,
        },
        "notes": {
            "training_config_precision": (
                "当前展示优先使用现有产物链路；若历史 episode 未记录完整超参数，则使用当前 config.py 快照并明确标注。"
            ),
            "model_parameter_precision": (
                "checkpoint 本身未记录完整 PPO 超参数时，仅展示模型组、路径和当前默认配置。"
            ),
        },
    }
    return payload


def summarize_episode_record(summary: Dict[str, Any], episode_dir: Path, workspace: Path) -> Dict[str, Any]:
    created_at = summary.get("created_at") or datetime.fromtimestamp(
        episode_dir.joinpath("summary.json").stat().st_mtime
    ).isoformat(timespec="seconds")
    stats_path = _resolve_training_stats_path(episode_dir)
    stats_preview = training_stats_preview(stats_path, workspace)
    selected_model_id = summary.get("selected_model_id")
    report_path = episode_dir / "simulation_report.html"
    episode_path = episode_dir / "episode.json"

    return {
        "episode_id": summary.get("episode_id") or episode_dir.name,
        "kind": summary.get("kind"),
        "job_id": summary.get("job_id"),
        "episode_index": summary.get("episode_index"),
        "seed": summary.get("seed"),
        "completed_cargos": summary.get("completed_cargos"),
        "timed_out_cargos": summary.get("timed_out_cargos"),
        "average_wait_time": summary.get("average_wait_time"),
        "average_completion_time": summary.get("average_completion_time"),
        "controller_mode": summary.get("controller_mode"),
        "model_mapping": summary.get("model_mapping", {}),
        "selected_model_id": selected_model_id,
        "source_model_path": summary.get("source_model_path"),
        "created_at": created_at,
        "report_path": str(report_path.resolve()),
        "report_path_display": _display_path(report_path, workspace),
        "episode_path": str(episode_path.resolve()),
        "episode_path_display": _display_path(episode_path, workspace),
        "training_stats_path": str(stats_path.resolve()) if stats_path is not None else None,
        "training_stats_path_display": _display_path(stats_path, workspace) if stats_path is not None else None,
        "training_stats_available": bool(stats_preview),
        "data_source_label": "Seeded simulation",
    }


def summarize_job_record(job: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    cloned = _json_clone(job)
    output_dir = Path(cloned["output_dir"]) if cloned.get("output_dir") else None
    config = cloned.get("config", {})
    kind = cloned.get("kind")
    config_summary: Dict[str, Any]
    if kind == "train":
        config_summary = {
            "episodes": config.get("num_episodes"),
            "sample_interval": config.get("sample_interval"),
            "controller_mode": config.get("controller_mode"),
            "device": config.get("device"),
            "save_final_models": config.get("save_final_models"),
            "episode_duration": config.get("episode_duration"),
        }
    else:
        config_summary = {
            "model_id": config.get("model_id"),
            "episodes": config.get("num_episodes"),
            "seed": config.get("seed"),
            "sample_interval": config.get("sample_interval"),
            "simulation_duration": config.get("simulation_duration"),
            "deterministic_inference": config.get("deterministic_inference"),
        }

    cloned["output_dir_display"] = _display_path(output_dir, workspace) if output_dir is not None else None
    cloned["config_summary"] = config_summary
    cloned["data_source_label"] = "Seeded simulation"
    if cloned.get("error"):
        cloned["error_summary"] = str(cloned["error"]).splitlines()[0]
    return cloned
