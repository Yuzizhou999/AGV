import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import (
    EPISODE_DURATION,
    HIGH_LEVEL_DECISION_INTERVAL,
    LOADING_POSITIONS,
    LOW_LEVEL_CONTROL_INTERVAL,
    MAX_VEHICLES,
    TRACK_LENGTH,
    UNLOADING_POSITIONS,
)
from custom_ppo_controller import CustomPPOController
from environment import Environment
from heuristic_high_level import HeuristicHighLevelController
from html_visualization import _json_safe, write_simulation_report


class DualPPOVisualizationRunner:
    def __init__(
        self,
        model_paths,
        output_dir,
        seed=42,
        simulation_duration=EPISODE_DURATION,
        sample_interval=5.0,
        controller_factory=CustomPPOController,
        device="cpu",
    ):
        self.model_paths = {int(vehicle_id): str(path) for vehicle_id, path in model_paths.items()}
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.simulation_duration = simulation_duration
        self.sample_interval = sample_interval
        self.controller_factory = controller_factory
        self.device = device

        self._validate_model_paths()

        self.env = Environment(seed=seed)
        self.high_level_controller = HeuristicHighLevelController(self.env)
        self.low_level_controller = controller_factory(
            self.env,
            model_paths=self.model_paths,
            device=device,
            total_episodes=1,
        )
        if hasattr(self.low_level_controller, "reset_episode"):
            self.low_level_controller.reset_episode()

    def _validate_model_paths(self):
        missing = [path for path in self.model_paths.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")

    def _capture_frame(self):
        return {
            "time": round(self.env.current_time, 4),
            "vehicles": [
                {
                    "id": vehicle_id,
                    "position": round(vehicle.position, 4),
                    "velocity": round(vehicle.velocity, 4),
                    "slot_count": sum(1 for slot in vehicle.slots if slot is not None),
                    "is_loading_unloading": bool(vehicle.is_loading_unloading),
                }
                for vehicle_id, vehicle in self.env.vehicles.items()
            ],
            "loading_stations": [
                {
                    "id": station_id,
                    "occupied_slots": sum(1 for slot in station.slots if slot is not None),
                }
                for station_id, station in self.env.loading_stations.items()
            ],
            "completed_cargos": self.env.completed_cargos,
            "timed_out_cargos": self.env.timed_out_cargos,
            "active_cargos": len(self.env.cargos),
            "safety_warning_count": self.env.safety_warning_count,
            "collision_alert_count": self.env.collision_alert_count,
            "recent_alerts": self.env.alerts[-5:],
        }

    def _record_statistics(self, stats):
        stats["time"].append(round(self.env.current_time, 4))
        stats["completed"].append(self.env.completed_cargos)
        stats["timeout"].append(self.env.timed_out_cargos)
        stats["cargo_count"].append(len(self.env.cargos))
        avg_wait = self.env.total_wait_time / max(1, self.env.completed_cargos)
        stats["avg_wait_time"].append(avg_wait)
        for vehicle_id, vehicle in self.env.vehicles.items():
            stats["vehicle_positions"][vehicle_id].append(vehicle.position)
            stats["vehicle_velocities"][vehicle_id].append(vehicle.velocity)

    def _write_statistics_plot(self, statistics, output_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        time = np.array(statistics["time"])

        axes[0, 0].plot(time, statistics["completed"], color="green", linewidth=2)
        axes[0, 0].plot(time, statistics["timeout"], color="red", linewidth=2)
        axes[0, 0].set_title("Cargo Completion and Timeout")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time, statistics["avg_wait_time"], color="blue", linewidth=2)
        axes[0, 1].set_title("Average Wait Time")
        axes[0, 1].grid(True, alpha=0.3)

        for vehicle_id in range(MAX_VEHICLES):
            axes[1, 0].plot(
                time,
                statistics["vehicle_positions"][vehicle_id],
                linewidth=1.5,
                label=f"Vehicle {vehicle_id}",
            )
        axes[1, 0].set_title("Vehicle Position Trajectory")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time, statistics["cargo_count"], color="purple", linewidth=2)
        axes[1, 1].set_title("System Cargo Count")
        axes[1, 1].grid(True, alpha=0.3)

        for axis in axes.ravel():
            axis.set_xlabel("Time (s)")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _build_summary(self):
        avg_wait_time = self.env.total_wait_time / max(1, self.env.completed_cargos)
        avg_completion_time = 0.0
        if self.env.completed_cargo_list:
            avg_completion_time = sum(
                cargo["completion_time"] - cargo["arrival_time"]
                for cargo in self.env.completed_cargo_list
            ) / len(self.env.completed_cargo_list)

        return {
            "seed": self.seed,
            "deterministic_inference": True,
            "model_mapping": {str(k): v for k, v in self.model_paths.items()},
            "episode_duration": self.simulation_duration,
            "simulation_control_interval": LOW_LEVEL_CONTROL_INTERVAL,
            "high_level_decision_interval": HIGH_LEVEL_DECISION_INTERVAL,
            "visualization_sampling_interval": self.sample_interval,
            "track_length": TRACK_LENGTH,
            "loading_positions": LOADING_POSITIONS,
            "unloading_positions": UNLOADING_POSITIONS,
            "completed_cargos": self.env.completed_cargos,
            "timed_out_cargos": self.env.timed_out_cargos,
            "average_wait_time": avg_wait_time,
            "average_completion_time": avg_completion_time,
            "safety_warning_count": self.env.safety_warning_count,
            "collision_alert_count": self.env.collision_alert_count,
            "alerts": self.env.alerts,
            "completed_cargo_records": self.env.completed_cargo_list,
        }

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        obs = self.env.reset(seed=self.seed)
        if hasattr(self.low_level_controller, "reset_episode"):
            self.low_level_controller.reset_episode()

        frames = [self._capture_frame()]
        statistics = {
            "time": [],
            "completed": [],
            "timeout": [],
            "avg_wait_time": [],
            "cargo_count": [],
            "vehicle_positions": {i: [] for i in range(MAX_VEHICLES)},
            "vehicle_velocities": {i: [] for i in range(MAX_VEHICLES)},
        }
        self._record_statistics(statistics)

        next_high_level_decision = 0.0
        next_sample_time = self.sample_interval
        high_level_action = None

        while self.env.current_time < self.simulation_duration:
            if self.env.current_time >= next_high_level_decision:
                high_level_action = self.high_level_controller.compute_action(obs)
                next_high_level_decision = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL

            low_level_actions = self.low_level_controller.compute_actions(deterministic=True)
            next_obs, env_reward, done = self.env.step(high_level_action, low_level_actions)

            if hasattr(self.low_level_controller, "compute_and_store_rewards"):
                self.low_level_controller.compute_and_store_rewards(done=done, env_task_reward=env_reward)

            obs = next_obs
            self._record_statistics(statistics)

            if self.env.current_time >= next_sample_time:
                frames.append(self._capture_frame())
                next_sample_time += self.sample_interval

            if done:
                break

        if not frames or frames[-1]["time"] != round(self.env.current_time, 4):
            frames.append(self._capture_frame())

        summary = self._build_summary()
        html_path = self.output_dir / "simulation_report.html"
        statistics_path = self.output_dir / "statistics.png"
        summary_path = self.output_dir / "summary.json"

        write_simulation_report(html_path, frames=frames, summary=summary)
        self._write_statistics_plot(statistics, statistics_path)
        summary_path.write_text(
            json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "output_dir": str(self.output_dir),
            "html_path": str(html_path),
            "statistics_path": str(statistics_path),
            "summary_path": str(summary_path),
            "frames": frames,
            "statistics": statistics,
        }


def build_default_output_dir(base_dir=None):
    base_path = Path(base_dir or "outputs/dual_ppo_visualization")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / timestamp


def main():
    default_output_dir = build_default_output_dir()
    runner = DualPPOVisualizationRunner(
        model_paths={
            0: "models/rl_low_level_best_v0.pth",
            1: "models/rl_low_level_best_v1.pth",
        },
        output_dir=default_output_dir,
        seed=42,
        simulation_duration=EPISODE_DURATION,
        sample_interval=5.0,
    )
    result = runner.run()
    print(f"HTML report: {result['html_path']}")
    print(f"Statistics plot: {result['statistics_path']}")
    print(f"Summary JSON: {result['summary_path']}")


if __name__ == "__main__":
    main()
