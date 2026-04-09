import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from custom_ppo_controller import CustomPPOController, MAX_VEHICLES
from environment import Environment
from html_visualization import write_simulation_report
from ppo_agent import PPOAgent
from run_dual_ppo_visualization import DualPPOVisualizationRunner


class TestCustomPPOControllerModelPaths(unittest.TestCase):
    def test_model_paths_load_exact_requested_file_per_vehicle(self):
        env = SimpleNamespace(vehicles={})
        created_agents = []
        model_paths = {
            0: "models/vehicle-zero-explicit.pth",
            1: "models/vehicle-one-explicit.pth",
        }

        def build_agent(*args, **kwargs):
            agent = Mock()
            created_agents.append(agent)
            return agent

        with patch("custom_ppo_controller.PPOAgent", side_effect=build_agent), patch(
            "custom_ppo_controller.os.path.exists",
            side_effect=lambda path: path in model_paths.values(),
        ):
            controller = CustomPPOController(env, model_paths=model_paths)

        self.assertEqual(len(created_agents), MAX_VEHICLES)
        self.assertEqual(controller.agents[0].load.call_args_list, [((model_paths[0],), {})])
        self.assertEqual(controller.agents[1].load.call_args_list, [((model_paths[1],), {})])

        for vehicle_id in range(2, MAX_VEHICLES):
            self.assertEqual(controller.agents[vehicle_id].load.call_count, 0)


class TestPPOAgentCheckpointLoading(unittest.TestCase):
    def test_load_supports_saved_checkpoint_on_current_torch(self):
        agent = PPOAgent(obs_dim=15, action_dim=1, device="cpu", total_episodes=1)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            checkpoint_path = handle.name

        try:
            torch.save(
                {
                    "policy_state_dict": agent.policy.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                },
                checkpoint_path,
            )

            reloaded = PPOAgent(obs_dim=15, action_dim=1, device="cpu", total_episodes=1)
            reloaded.load(checkpoint_path)
        finally:
            import os

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


class TestHtmlVisualization(unittest.TestCase):
    def test_write_simulation_report_outputs_controls_and_frame_payload(self):
        frames = [
            {
                "time": 0.0,
                "vehicles": [
                    {"id": 0, "position": 0.0, "velocity": 0.0, "slot_count": 0, "is_loading_unloading": False},
                    {"id": 1, "position": 50.0, "velocity": 0.0, "slot_count": 1, "is_loading_unloading": True},
                ],
                "loading_stations": [{"id": 0, "occupied_slots": 1}, {"id": 1, "occupied_slots": 0}],
                "completed_cargos": 0,
                "timed_out_cargos": 0,
                "active_cargos": 1,
                "recent_alerts": [{"time": 0.0, "level": "warning", "vehicle_ids": [0, 1], "distance": 1.0}],
            }
        ]
        summary = {"completed_cargos": 0, "timed_out_cargos": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "simulation_report.html"
            write_simulation_report(output_path, frames=frames, summary=summary)

            contents = output_path.read_text(encoding="utf-8")

        self.assertIn("Play", contents)
        self.assertIn("timeSlider", contents)
        self.assertIn("FRAME_DATA", contents)
        self.assertIn("summaryMetrics", contents)
        self.assertIn("Recent Alerts", contents)
        self.assertIn("recentAlerts", contents)


class TestDualPPOVisualizationRunner(unittest.TestCase):
    def test_runner_exports_metadata_and_uses_deterministic_actions(self):
        controllers = []

        class DummyController:
            def __init__(self, env, model_paths=None, device="cpu", total_episodes=1, **kwargs):
                self.env = env
                self.model_paths = model_paths
                self.compute_calls = []
                controllers.append(self)

            def reset_episode(self):
                return None

            def compute_actions(self, deterministic=False):
                self.compute_calls.append(deterministic)
                return {vehicle_id: 0.0 for vehicle_id in self.env.vehicles.keys()}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_paths = {}
            for vehicle_id in range(MAX_VEHICLES):
                model_path = tmpdir_path / f"vehicle_{vehicle_id}.pth"
                model_path.write_text("dummy", encoding="utf-8")
                model_paths[vehicle_id] = str(model_path)

            runner = DualPPOVisualizationRunner(
                model_paths=model_paths,
                output_dir=tmpdir_path / "export",
                seed=123,
                simulation_duration=10.0,
                sample_interval=5.0,
                controller_factory=DummyController,
            )
            result = runner.run()

            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            html_exists = Path(result["html_path"]).exists()
            statistics_exists = Path(result["statistics_path"]).exists()

        self.assertTrue(all(controllers[0].compute_calls))
        self.assertTrue(html_exists)
        self.assertTrue(statistics_exists)
        self.assertEqual(summary["seed"], 123)
        self.assertTrue(summary["deterministic_inference"])
        self.assertEqual(summary["visualization_sampling_interval"], 5.0)
        self.assertEqual(summary["simulation_control_interval"], 0.5)
        self.assertEqual(summary["high_level_decision_interval"], 1.0)
        self.assertEqual(summary["model_mapping"], {str(k): v for k, v in model_paths.items()})
        self.assertGreater(len(result["statistics"]["time"]), len(result["frames"]))
        self.assertIn("safety_warning_count", summary)
        self.assertIn("collision_alert_count", summary)
        self.assertIn("alerts", summary)

    def test_runner_is_reproducible_for_same_seed_and_controller(self):
        class DummyController:
            def __init__(self, env, **kwargs):
                self.env = env

            def reset_episode(self):
                return None

            def compute_actions(self, deterministic=False):
                return {vehicle_id: 0.0 for vehicle_id in self.env.vehicles.keys()}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_paths = {}
            for vehicle_id in range(MAX_VEHICLES):
                model_path = tmpdir_path / f"vehicle_{vehicle_id}.pth"
                model_path.write_text("dummy", encoding="utf-8")
                model_paths[vehicle_id] = str(model_path)

            first = DualPPOVisualizationRunner(
                model_paths=model_paths,
                output_dir=tmpdir_path / "first",
                seed=321,
                simulation_duration=10.0,
                sample_interval=5.0,
                controller_factory=DummyController,
            ).run()
            second = DualPPOVisualizationRunner(
                model_paths=model_paths,
                output_dir=tmpdir_path / "second",
                seed=321,
                simulation_duration=10.0,
                sample_interval=5.0,
                controller_factory=DummyController,
            ).run()

        self.assertEqual(first["frames"], second["frames"])


class TestCollisionAlerts(unittest.TestCase):
    def test_environment_records_safety_warning_when_distance_below_threshold(self):
        env = Environment(seed=42)
        env.vehicles[0].position = 10.0
        env.vehicles[1].position = 11.0

        env._update_proximity_alerts()

        self.assertGreaterEqual(len(env.alerts), 1)
        self.assertEqual(env.alerts[-1]["level"], "warning")

    def test_environment_records_collision_when_positions_overlap(self):
        env = Environment(seed=42)
        env.vehicles[0].position = 10.0
        env.vehicles[1].position = 10.0

        env._update_proximity_alerts()

        self.assertGreaterEqual(len(env.alerts), 1)
        self.assertEqual(env.alerts[-1]["level"], "collision")


if __name__ == "__main__":
    unittest.main()
