import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from custom_ppo_controller import CustomPPOController, MAX_VEHICLES
from environment import Cargo, Environment
from ui.html_visualization import write_simulation_report
from ppo_agent import PPOAgent
from run_dual_ppo_visualization import DualPPOVisualizationRunner
from simulation_capture import build_frame
from ui.simulation_web import discover_model_sets


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
                "system": {"completed_cargos": 0, "timed_out_cargos": 0, "avg_wait_time": 0.0, "current_cargos": 1},
                "vehicles": [
                    {
                        "id": 0,
                        "position": 0.0,
                        "velocity": 0.0,
                        "slot_count": 0,
                        "is_loading_unloading": False,
                        "slots": [],
                        "operation": {"mode": "ready", "label": "Ready"},
                        "status_text": "Ready",
                    },
                    {
                        "id": 1,
                        "position": 50.0,
                        "velocity": 0.0,
                        "slot_count": 1,
                        "is_loading_unloading": True,
                        "slots": [],
                        "operation": {"mode": "loading", "label": "Loading C1", "remaining_time": 5.0},
                        "status_text": "Loading C1",
                    },
                ],
                "loading_stations": [{"id": 0, "position": 20.0, "occupied_slots": 1, "slots": []}, {"id": 1, "position": 60.0, "occupied_slots": 0, "slots": []}],
                "unloading_stations": [{"id": 0, "position": 30.0, "description": "Unlimited receiving"}],
                "waiting_cargos": [],
                "active_cargos": [],
                "recent_completed_cargos": [],
                "recent_alerts": [{"time": 0.0, "level": "warning", "vehicle_ids": [0, 1], "distance": 1.0}],
            }
        ]
        summary = {"completed_cargos": 0, "timed_out_cargos": 0, "kind": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "simulation_report.html"
            write_simulation_report(output_path, frames=frames, summary=summary)

            contents = output_path.read_text(encoding="utf-8")

        self.assertIn("训练任务", contents)
        self.assertIn("timeSlider", contents)
        self.assertIn("window.__SIMULATION_BOOTSTRAP__", contents)
        self.assertIn("Episode Library", contents)
        self.assertIn("最近告警", contents)


class TestSimulationCapture(unittest.TestCase):
    def test_build_frame_contains_visualizer_fields(self):
        env = Environment(seed=42)
        cargo = Cargo(
            id=0,
            arrival_time=0.0,
            loading_station=0,
            loading_slot=0,
            allowed_unloading_stations={0, 2},
            assigned_vehicle=0,
            assigned_vehicle_slot=0,
        )
        env.cargos[cargo.id] = cargo
        env.loading_stations[0].slots[0] = cargo.id
        env.completed_cargo_list.append(
            {
                "id": 99,
                "arrival_time": 0.0,
                "completion_time": 20.0,
                "wait_time": 12.0,
                "loading_station": 0,
                "unloading_station": 2,
                "vehicle_id": 1,
            }
        )

        frame = build_frame(env)

        self.assertIn("system", frame)
        self.assertIn("vehicles", frame)
        self.assertIn("loading_stations", frame)
        self.assertIn("recent_completed_cargos", frame)
        self.assertIn("waiting_cargos", frame)
        self.assertIn("active_cargos", frame)
        self.assertIn("operation", frame["vehicles"][0])
        self.assertIn("slots", frame["vehicles"][0])
        self.assertEqual(frame["loading_stations"][0]["slots"][0]["assigned_vehicle"], 0)
        self.assertEqual(frame["waiting_cargos"][0]["destination_text"], "(OP0, OP2)")


class TestModelDiscovery(unittest.TestCase):
    def test_discover_model_sets_groups_complete_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "alpha_v0.pth").write_text("x", encoding="utf-8")
            (root / "alpha_v1.pth").write_text("x", encoding="utf-8")
            (root / "beta_v0.pth").write_text("x", encoding="utf-8")

            discovered = discover_model_sets(root)

        alpha = next(item for item in discovered if item["id"] == "alpha")
        beta = next(item for item in discovered if item["id"] == "beta")
        self.assertTrue(alpha["complete"])
        self.assertFalse(beta["complete"])
        self.assertEqual(sorted(alpha["vehicle_ids"]), [0, 1])


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
        self.assertGreaterEqual(len(result["statistics"]["time"]), len(result["frames"]))
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
