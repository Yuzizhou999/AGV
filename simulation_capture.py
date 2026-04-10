from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import (
    EPISODE_DURATION,
    HIGH_LEVEL_DECISION_INTERVAL,
    LOADING_POSITIONS,
    LOADING_TIME,
    LOW_LEVEL_CONTROL_INTERVAL,
    MAX_VEHICLES,
    TRACK_LENGTH,
    UNLOADING_POSITIONS,
    UNLOADING_TIME,
)
from environment import Cargo, Environment


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _station_labels(station_ids: List[int]) -> List[str]:
    return [f"OP{station_id}" for station_id in station_ids]


def _cargo_destination_text(cargo: Cargo) -> str:
    if cargo.target_unloading_station is not None:
        return f"OP{cargo.target_unloading_station}"
    allowed = sorted(int(station_id) for station_id in cargo.allowed_unloading_stations)
    return f"({', '.join(_station_labels(allowed))})"


def _locate_cargo(env: Environment, cargo: Cargo) -> Dict[str, Any]:
    if env.is_cargo_at_loading_station(cargo):
        return {
            "type": "loading_station",
            "station_id": cargo.loading_station,
            "slot_index": cargo.loading_slot,
            "vehicle_id": None,
            "vehicle_slot_index": None,
        }

    if cargo.assigned_vehicle is not None:
        vehicle = env.vehicles.get(cargo.assigned_vehicle)
        if vehicle is not None:
            for slot_index, cargo_id in enumerate(vehicle.slots):
                if cargo_id == cargo.id:
                    return {
                        "type": "vehicle",
                        "station_id": None,
                        "slot_index": None,
                        "vehicle_id": cargo.assigned_vehicle,
                        "vehicle_slot_index": slot_index,
                    }

    for vehicle_id, vehicle in env.vehicles.items():
        for slot_index, cargo_id in enumerate(vehicle.slots):
            if cargo_id == cargo.id:
                return {
                    "type": "vehicle",
                    "station_id": None,
                    "slot_index": None,
                    "vehicle_id": vehicle_id,
                    "vehicle_slot_index": slot_index,
                }

    return {
        "type": "unknown",
        "station_id": None,
        "slot_index": None,
        "vehicle_id": cargo.assigned_vehicle,
        "vehicle_slot_index": cargo.assigned_vehicle_slot,
    }


def _build_vehicle_operation(env: Environment, vehicle_id: int) -> Dict[str, Any]:
    vehicle = env.vehicles[vehicle_id]
    if not vehicle.is_loading_unloading:
        return {
            "mode": "ready",
            "label": "Ready",
            "cargo_id": None,
            "remaining_time": None,
        }

    for cargo in env.cargos.values():
        if cargo.assigned_vehicle == vehicle_id and cargo.loading_start_time is not None:
            remaining_time = max(0.0, LOADING_TIME - (env.current_time - cargo.loading_start_time))
            return {
                "mode": "loading",
                "label": f"Loading C{cargo.id}",
                "cargo_id": cargo.id,
                "loading_station": cargo.loading_station,
                "vehicle_slot_index": cargo.assigned_vehicle_slot,
                "remaining_time": _round(remaining_time),
            }

    for slot_index, cargo_id in enumerate(vehicle.slots):
        if cargo_id is None or cargo_id not in env.cargos:
            continue
        cargo = env.cargos[cargo_id]
        if cargo.unloading_start_time is not None:
            remaining_time = max(0.0, UNLOADING_TIME - (env.current_time - cargo.unloading_start_time))
            return {
                "mode": "unloading",
                "label": f"Unloading C{cargo.id}",
                "cargo_id": cargo.id,
                "unloading_station": cargo.target_unloading_station,
                "vehicle_slot_index": slot_index,
                "remaining_time": _round(remaining_time),
            }

    return {
        "mode": "waiting",
        "label": "Loading/Unloading (Waiting)",
        "cargo_id": None,
        "remaining_time": None,
    }


def _build_vehicle_slots(env: Environment, vehicle_id: int) -> List[Dict[str, Any]]:
    vehicle = env.vehicles[vehicle_id]
    slots: List[Dict[str, Any]] = []
    for slot_index, cargo_id in enumerate(vehicle.slots):
        slot_payload: Dict[str, Any] = {
            "slot_index": slot_index,
            "cargo_id": cargo_id,
            "occupied": cargo_id is not None,
            "destination_text": None,
            "allowed_unloading_stations": [],
            "target_unloading_station": None,
            "assigned_vehicle": vehicle_id if cargo_id is not None else None,
            "assigned_vehicle_slot": slot_index if cargo_id is not None else None,
        }
        if cargo_id is not None and cargo_id in env.cargos:
            cargo = env.cargos[cargo_id]
            allowed = sorted(int(station_id) for station_id in cargo.allowed_unloading_stations)
            slot_payload.update(
                {
                    "destination_text": _cargo_destination_text(cargo),
                    "allowed_unloading_stations": allowed,
                    "target_unloading_station": cargo.target_unloading_station,
                    "wait_time": _round(cargo.wait_time(env.current_time)),
                    "is_timeout": cargo.is_timeout(env.current_time),
                }
            )
        slots.append(slot_payload)
    return slots


def _build_vehicle_state(env: Environment, vehicle_id: int) -> Dict[str, Any]:
    vehicle = env.vehicles[vehicle_id]
    operation = _build_vehicle_operation(env, vehicle_id)
    slots = _build_vehicle_slots(env, vehicle_id)
    occupied_slots = sum(1 for slot in vehicle.slots if slot is not None)

    return {
        "id": vehicle_id,
        "position": _round(vehicle.position),
        "velocity": _round(vehicle.velocity),
        "slots": slots,
        "slot_count": occupied_slots,
        "cargo_label": f"Box x{occupied_slots}" if occupied_slots else None,
        "is_loading_unloading": bool(vehicle.is_loading_unloading),
        "operation": operation,
        "status_text": operation["label"],
    }


def _build_loading_station_state(env: Environment, station_id: int) -> Dict[str, Any]:
    station = env.loading_stations[station_id]
    slots: List[Dict[str, Any]] = []
    for slot_index, cargo_id in enumerate(station.slots):
        slot_payload: Dict[str, Any] = {
            "slot_index": slot_index,
            "cargo_id": cargo_id,
            "occupied": cargo_id is not None,
            "wait_time": None,
            "is_timeout": False,
            "assigned_vehicle": None,
            "assigned_vehicle_slot": None,
            "allowed_unloading_stations": [],
            "destination_text": None,
        }
        if cargo_id is not None and cargo_id in env.cargos:
            cargo = env.cargos[cargo_id]
            allowed = sorted(int(station_id) for station_id in cargo.allowed_unloading_stations)
            slot_payload.update(
                {
                    "wait_time": _round(cargo.wait_time(env.current_time)),
                    "is_timeout": cargo.is_timeout(env.current_time),
                    "assigned_vehicle": cargo.assigned_vehicle,
                    "assigned_vehicle_slot": cargo.assigned_vehicle_slot,
                    "allowed_unloading_stations": allowed,
                    "destination_text": _cargo_destination_text(cargo),
                }
            )
        slots.append(slot_payload)

    return {
        "id": station_id,
        "position": _round(station.position),
        "occupied_slots": sum(1 for slot in station.slots if slot is not None),
        "slots": slots,
    }


def _build_unloading_station_state(env: Environment, station_id: int) -> Dict[str, Any]:
    station = env.unloading_stations[station_id]
    return {
        "id": station_id,
        "position": _round(station.position),
        "mode": "unlimited_receiving",
        "description": "Unlimited receiving (no slot state)",
    }


def _build_waiting_cargos(env: Environment) -> List[Dict[str, Any]]:
    waiting_cargos: List[Dict[str, Any]] = []
    for cargo in env.cargos.values():
        if cargo.completion_time is not None or not env.is_cargo_at_loading_station(cargo):
            continue
        waiting_cargos.append(
            {
                "id": cargo.id,
                "arrival_time": _round(cargo.arrival_time),
                "wait_time": _round(cargo.wait_time(env.current_time)),
                "is_timeout": cargo.is_timeout(env.current_time),
                "loading_station": cargo.loading_station,
                "loading_slot": cargo.loading_slot,
                "assigned_vehicle": cargo.assigned_vehicle,
                "assigned_vehicle_slot": cargo.assigned_vehicle_slot,
                "allowed_unloading_stations": sorted(
                    int(station_id) for station_id in cargo.allowed_unloading_stations
                ),
                "target_unloading_station": cargo.target_unloading_station,
                "destination_text": _cargo_destination_text(cargo),
            }
        )

    waiting_cargos.sort(key=lambda cargo: (not cargo["is_timeout"], -cargo["wait_time"], cargo["id"]))
    return waiting_cargos


def _build_active_cargos(env: Environment) -> List[Dict[str, Any]]:
    active_cargos: List[Dict[str, Any]] = []
    for cargo in sorted(env.cargos.values(), key=lambda item: item.id):
        location = _locate_cargo(env, cargo)
        active_cargos.append(
            {
                "id": cargo.id,
                "arrival_time": _round(cargo.arrival_time),
                "completion_time": _round(cargo.completion_time),
                "wait_time": _round(cargo.wait_time(env.current_time)),
                "is_timeout": cargo.is_timeout(env.current_time),
                "loading_station": cargo.loading_station,
                "loading_slot": cargo.loading_slot,
                "assigned_vehicle": cargo.assigned_vehicle,
                "assigned_vehicle_slot": cargo.assigned_vehicle_slot,
                "allowed_unloading_stations": sorted(
                    int(station_id) for station_id in cargo.allowed_unloading_stations
                ),
                "target_unloading_station": cargo.target_unloading_station,
                "destination_text": _cargo_destination_text(cargo),
                "loading_start_time": _round(cargo.loading_start_time),
                "unloading_start_time": _round(cargo.unloading_start_time),
                "picked_up_time": _round(cargo.picked_up_time),
                "location": location,
            }
        )
    return active_cargos


def build_frame(
    env: Environment,
    recent_alert_limit: int = 10,
    recent_completed_limit: int = 5,
) -> Dict[str, Any]:
    waiting_cargos = _build_waiting_cargos(env)
    active_cargos = _build_active_cargos(env)
    on_vehicle_cargos = sum(
        1
        for vehicle in env.vehicles.values()
        for cargo_id in vehicle.slots
        if cargo_id is not None
    )
    avg_wait = env.total_wait_time / max(1, env.completed_cargos) if env.completed_cargos else 0.0

    return {
        "time": _round(env.current_time),
        "system": {
            "completed_cargos": env.completed_cargos,
            "timed_out_cargos": env.timed_out_cargos,
            "avg_wait_time": _round(avg_wait),
            "current_cargos": len(env.cargos),
            "generated_cargos": env.cargo_counter,
            "waiting_cargos": len(waiting_cargos),
            "on_vehicle_cargos": on_vehicle_cargos,
            "safety_warning_count": env.safety_warning_count,
            "collision_alert_count": env.collision_alert_count,
            "safety_violations": [int(vehicle_id) for vehicle_id in env.safety_violations],
        },
        "vehicles": [_build_vehicle_state(env, vehicle_id) for vehicle_id in sorted(env.vehicles.keys())],
        "loading_stations": [
            _build_loading_station_state(env, station_id)
            for station_id in sorted(env.loading_stations.keys())
        ],
        "unloading_stations": [
            _build_unloading_station_state(env, station_id)
            for station_id in sorted(env.unloading_stations.keys())
        ],
        "waiting_cargos": waiting_cargos,
        "active_cargos": active_cargos,
        "recent_completed_cargos": env.completed_cargo_list[-recent_completed_limit:],
        "recent_alerts": env.alerts[-recent_alert_limit:],
    }


def build_summary(
    env: Environment,
    *,
    sample_interval: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    avg_wait_time = env.total_wait_time / max(1, env.completed_cargos) if env.completed_cargos else 0.0
    avg_completion_time = 0.0
    if env.completed_cargo_list:
        avg_completion_time = sum(
            cargo["completion_time"] - cargo["arrival_time"]
            for cargo in env.completed_cargo_list
        ) / len(env.completed_cargo_list)

    summary = {
        "kind": metadata.get("kind", "simulation"),
        "job_id": metadata.get("job_id"),
        "episode_id": metadata.get("episode_id"),
        "episode_index": metadata.get("episode_index"),
        "created_at": metadata.get("created_at", datetime.now().isoformat(timespec="seconds")),
        "seed": metadata.get("seed"),
        "deterministic_inference": metadata.get("deterministic_inference"),
        "controller_mode": metadata.get("controller_mode"),
        "model_mapping": metadata.get("model_mapping", {}),
        "run_label": metadata.get("run_label"),
        "episode_duration": metadata.get("episode_duration", EPISODE_DURATION),
        "simulation_control_interval": LOW_LEVEL_CONTROL_INTERVAL,
        "high_level_decision_interval": HIGH_LEVEL_DECISION_INTERVAL,
        "visualization_sampling_interval": sample_interval,
        "track_length": TRACK_LENGTH,
        "loading_positions": LOADING_POSITIONS,
        "unloading_positions": UNLOADING_POSITIONS,
        "completed_cargos": env.completed_cargos,
        "timed_out_cargos": env.timed_out_cargos,
        "average_wait_time": _round(avg_wait_time),
        "average_completion_time": _round(avg_completion_time),
        "active_cargo_count": len(env.cargos),
        "generated_cargos": env.cargo_counter,
        "safety_warning_count": env.safety_warning_count,
        "collision_alert_count": env.collision_alert_count,
        "alerts": env.alerts,
        "completed_cargo_records": env.completed_cargo_list,
    }

    for key, value in metadata.items():
        if key not in summary:
            summary[key] = value

    return summary


def build_statistics_sample(env: Environment) -> Dict[str, Any]:
    waiting_cargo_count = sum(
        1 for cargo in env.cargos.values() if cargo.completion_time is None and env.is_cargo_at_loading_station(cargo)
    )
    on_vehicle_cargo_count = sum(
        1
        for vehicle in env.vehicles.values()
        for cargo_id in vehicle.slots
        if cargo_id is not None
    )
    avg_wait = env.total_wait_time / max(1, env.completed_cargos) if env.completed_cargos else 0.0
    return {
        "time": _round(env.current_time),
        "completed": env.completed_cargos,
        "timeout": env.timed_out_cargos,
        "avg_wait_time": _round(avg_wait),
        "cargo_count": len(env.cargos),
        "waiting_cargos": waiting_cargo_count,
        "on_vehicle_cargos": on_vehicle_cargo_count,
        "safety_warning_count": env.safety_warning_count,
        "collision_alert_count": env.collision_alert_count,
        "vehicle_positions": {
            int(vehicle_id): _round(vehicle.position)
            for vehicle_id, vehicle in env.vehicles.items()
        },
        "vehicle_velocities": {
            int(vehicle_id): _round(vehicle.velocity)
            for vehicle_id, vehicle in env.vehicles.items()
        },
    }


class SimulationEpisodeRecorder:
    def __init__(self, env: Environment, sample_interval: float = 5.0):
        self.env = env
        self.sample_interval = max(float(sample_interval), LOW_LEVEL_CONTROL_INTERVAL)
        self.next_sample_time = 0.0
        self.frames: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {
            "time": [],
            "completed": [],
            "timeout": [],
            "avg_wait_time": [],
            "cargo_count": [],
            "waiting_cargos": [],
            "on_vehicle_cargos": [],
            "safety_warning_count": [],
            "collision_alert_count": [],
            "vehicle_positions": {vehicle_id: [] for vehicle_id in range(MAX_VEHICLES)},
            "vehicle_velocities": {vehicle_id: [] for vehicle_id in range(MAX_VEHICLES)},
        }
        self.capture(force=True)

    def _append_statistics(self, sample: Dict[str, Any]) -> None:
        self.statistics["time"].append(sample["time"])
        self.statistics["completed"].append(sample["completed"])
        self.statistics["timeout"].append(sample["timeout"])
        self.statistics["avg_wait_time"].append(sample["avg_wait_time"])
        self.statistics["cargo_count"].append(sample["cargo_count"])
        self.statistics["waiting_cargos"].append(sample["waiting_cargos"])
        self.statistics["on_vehicle_cargos"].append(sample["on_vehicle_cargos"])
        self.statistics["safety_warning_count"].append(sample["safety_warning_count"])
        self.statistics["collision_alert_count"].append(sample["collision_alert_count"])

        for vehicle_id in range(MAX_VEHICLES):
            self.statistics["vehicle_positions"][vehicle_id].append(
                sample["vehicle_positions"].get(vehicle_id)
            )
            self.statistics["vehicle_velocities"][vehicle_id].append(
                sample["vehicle_velocities"].get(vehicle_id)
            )

    def capture(self, force: bool = False) -> bool:
        current_time = float(self.env.current_time)
        if not force and current_time + 1e-9 < self.next_sample_time:
            return False

        frame = build_frame(self.env)
        if not self.frames or self.frames[-1]["time"] != frame["time"]:
            self.frames.append(frame)
            self._append_statistics(build_statistics_sample(self.env))

        if force:
            self.next_sample_time = current_time + self.sample_interval
        else:
            while self.next_sample_time <= current_time + 1e-9:
                self.next_sample_time += self.sample_interval
        return True

    def finalize(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.capture(force=True)
        return {
            "frames": self.frames,
            "summary": build_summary(
                self.env,
                sample_interval=self.sample_interval,
                metadata=metadata,
            ),
            "statistics": self.statistics,
        }


def write_statistics_plot(statistics: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = np.array(statistics["time"], dtype=float)
    if time.size == 0:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.set_title("No statistics collected")
        axis.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    axes[0, 0].plot(time, statistics["completed"], color="#4b9f5f", linewidth=2, label="Completed")
    axes[0, 0].plot(time, statistics["timeout"], color="#bf5a36", linewidth=2, label="Timeout")
    axes[0, 0].set_title("Cargo Completion and Timeout")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(time, statistics["avg_wait_time"], color="#176087", linewidth=2)
    axes[0, 1].set_title("Average Wait Time")
    axes[0, 1].grid(True, alpha=0.3)

    vehicle_positions = statistics.get("vehicle_positions", {})
    vehicle_velocities = statistics.get("vehicle_velocities", {})

    for vehicle_id in range(MAX_VEHICLES):
        axes[1, 0].plot(
            time,
            vehicle_positions.get(vehicle_id, vehicle_positions.get(str(vehicle_id), [])),
            linewidth=1.7,
            label=f"Vehicle {vehicle_id}",
        )
    axes[1, 0].set_title("Vehicle Position Trajectory")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    for vehicle_id in range(MAX_VEHICLES):
        axes[1, 1].plot(
            time,
            vehicle_velocities.get(vehicle_id, vehicle_velocities.get(str(vehicle_id), [])),
            linewidth=1.7,
            label=f"Vehicle {vehicle_id}",
        )
    axes[1, 1].set_title("Vehicle Velocity Trajectory")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(time, statistics["cargo_count"], color="#6e4f1f", linewidth=2, label="Active")
    axes[2, 0].plot(time, statistics["waiting_cargos"], color="#8b6a36", linewidth=2, label="Waiting")
    axes[2, 0].plot(time, statistics["on_vehicle_cargos"], color="#20303a", linewidth=2, label="On vehicle")
    axes[2, 0].set_title("Cargo Count Breakdown")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(
        time,
        statistics["safety_warning_count"],
        color="#c48d28",
        linewidth=2,
        label="Warnings",
    )
    axes[2, 1].plot(
        time,
        statistics["collision_alert_count"],
        color="#7d2f24",
        linewidth=2,
        label="Collisions",
    )
    axes[2, 1].set_title("Safety Alerts")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    for axis in axes.ravel():
        axis.set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_episode_artifacts(
    output_dir: Path,
    episode_payload: Dict[str, Any],
    *,
    include_statistics_plot: bool = True,
) -> Dict[str, str]:
    from ui.html_visualization import _json_safe, write_simulation_report

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_payload = _json_safe(episode_payload)

    episode_path = output_dir / "episode.json"
    summary_path = output_dir / "summary.json"
    statistics_path = output_dir / "statistics.json"
    report_path = output_dir / "simulation_report.html"

    episode_path.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(safe_payload["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    statistics_path.write_text(
        json.dumps(safe_payload["statistics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_simulation_report(report_path, episode=safe_payload)

    result = {
        "output_dir": str(output_dir),
        "episode_path": str(episode_path),
        "summary_path": str(summary_path),
        "statistics_path": str(statistics_path),
        "html_path": str(report_path),
    }

    if include_statistics_plot:
        statistics_plot_path = output_dir / "statistics.png"
        write_statistics_plot(safe_payload["statistics"], statistics_plot_path)
        result["statistics_plot_path"] = str(statistics_plot_path)

    return result
