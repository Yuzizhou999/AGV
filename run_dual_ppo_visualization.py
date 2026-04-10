from datetime import datetime
from pathlib import Path

from config import (
    EPISODE_DURATION,
    HIGH_LEVEL_DECISION_INTERVAL,
)
from custom_ppo_controller import CustomPPOController
from environment import Environment
from heuristic_high_level import HeuristicHighLevelController
from simulation_capture import (
    SimulationEpisodeRecorder,
    write_episode_artifacts,
)


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
        summary_metadata=None,
    ):
        self.model_paths = {int(vehicle_id): str(path) for vehicle_id, path in model_paths.items()}
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.simulation_duration = simulation_duration
        self.sample_interval = sample_interval
        self.controller_factory = controller_factory
        self.device = device
        self.summary_metadata = dict(summary_metadata or {})

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

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        obs = self.env.reset(seed=self.seed)
        if hasattr(self.low_level_controller, "reset_episode"):
            self.low_level_controller.reset_episode()
        recorder = SimulationEpisodeRecorder(self.env, sample_interval=self.sample_interval)

        next_high_level_decision = 0.0
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
            recorder.capture()

            if done:
                break

        summary_metadata = {
            "kind": "test",
            "seed": self.seed,
            "deterministic_inference": True,
            "model_mapping": {str(k): v for k, v in self.model_paths.items()},
            "episode_duration": self.simulation_duration,
            "controller_mode": "heuristic_high_level + custom_ppo_low_level",
        }
        summary_metadata.update(self.summary_metadata)
        episode_payload = recorder.finalize(summary_metadata)
        artifact_paths = write_episode_artifacts(self.output_dir, episode_payload)

        return {
            "output_dir": str(self.output_dir),
            "html_path": artifact_paths["html_path"],
            "episode_path": artifact_paths["episode_path"],
            "statistics_path": artifact_paths["statistics_plot_path"],
            "summary_path": artifact_paths["summary_path"],
            "frames": episode_payload["frames"],
            "statistics": episode_payload["statistics"],
            "summary": episode_payload["summary"],
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
