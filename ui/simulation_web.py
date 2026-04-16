from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

if __package__ in (None, ""):
    import sys

    workspace_root = Path(__file__).resolve().parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

from config import EPISODE_DURATION, MAX_VEHICLES
from ui.dashboard_metadata import (
    config_snapshot_summary,
    enrich_episode_payload,
    summarize_episode_record,
    summarize_job_record,
)


MODEL_SET_PATTERN = re.compile(r"(?P<prefix>.+)_v(?P<vehicle_id>\d+)\.pth$")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def discover_model_sets(models_dir: Path) -> List[Dict[str, Any]]:
    models_dir = Path(models_dir)
    groups: Dict[str, Dict[str, Any]] = {}
    if not models_dir.exists():
        return []

    for model_path in models_dir.rglob("*.pth"):
        match = MODEL_SET_PATTERN.match(model_path.name)
        if not match:
            continue
        relative_parent = model_path.parent.relative_to(models_dir)
        prefix = match.group("prefix")
        if str(relative_parent) != ".":
            model_id = f"{relative_parent.as_posix()}/{prefix}"
        else:
            model_id = prefix
        vehicle_id = int(match.group("vehicle_id"))
        group = groups.setdefault(
            model_id,
            {
                "id": model_id,
                "label": prefix,
                "directory": "." if str(relative_parent) == "." else relative_parent.as_posix(),
                "paths": {},
                "display_paths": {},
                "vehicle_ids": [],
                "updated_at": None,
                "complete": False,
            },
        )
        group["paths"][str(vehicle_id)] = str(model_path.resolve())
        group["display_paths"][str(vehicle_id)] = str(model_path.relative_to(models_dir).as_posix())
        group["vehicle_ids"].append(vehicle_id)
        file_mtime = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds")
        if group["updated_at"] is None or file_mtime > group["updated_at"]:
            group["updated_at"] = file_mtime

    discovered = []
    for group in groups.values():
        unique_vehicle_ids = sorted(set(group["vehicle_ids"]))
        group["vehicle_ids"] = unique_vehicle_ids
        group["complete"] = unique_vehicle_ids == list(range(MAX_VEHICLES))
        discovered.append(group)

    discovered.sort(
        key=lambda item: (item["complete"], item["updated_at"] or "", item["id"]),
        reverse=True,
    )
    return discovered


def discover_saved_episodes(runs_dir: Path, workspace: Optional[Path] = None) -> List[Dict[str, Any]]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    workspace = Path(workspace or runs_dir).resolve()

    episodes: List[Dict[str, Any]] = []
    for summary_path in runs_dir.rglob("summary.json"):
        episode_path = summary_path.parent / "episode.json"
        if not episode_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        episode_dir = summary_path.parent
        episode_summary = summarize_episode_record(summary, episode_dir, workspace)
        if not episode_summary.get("created_at"):
            episode_summary["created_at"] = datetime.fromtimestamp(
                summary_path.stat().st_mtime
            ).isoformat(timespec="seconds")
        episodes.append(episode_summary)

    episodes.sort(key=lambda item: (item["created_at"], item["episode_id"]), reverse=True)
    return episodes


class SimulationWebApp:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.models_dir = self.workspace / "models"
        self.runs_dir = self.workspace / "outputs" / "web_runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _job_dir(self, job_id: str) -> Path:
        return self.runs_dir / job_id

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self.lock:
            jobs = [summarize_job_record(job, self.workspace) for job in self.jobs.values()]
        jobs.sort(key=lambda item: item["created_at"], reverse=True)
        return jobs

    def get_state(self) -> Dict[str, Any]:
        return {
            "api_enabled": True,
            "workspace": str(self.workspace),
            "jobs": self.list_jobs(),
            "models": discover_model_sets(self.models_dir),
            "episodes": discover_saved_episodes(self.runs_dir, self.workspace),
            "config_snapshot": config_snapshot_summary(),
            "server_time": _now_iso(),
        }

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        for episode_path in self.runs_dir.rglob("episode.json"):
            if episode_path.parent.name != episode_id:
                continue
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            return enrich_episode_payload(
                payload,
                workspace=self.workspace,
                episode_dir=episode_path.parent,
                report_path=episode_path.parent / "simulation_report.html",
            )
        raise FileNotFoundError(f"Episode not found: {episode_id}")

    def resolve_model_set(self, model_id: str) -> Dict[str, Any]:
        for model_set in discover_model_sets(self.models_dir):
            if model_set["id"] == model_id:
                if not model_set["complete"]:
                    raise ValueError(f"Model set is incomplete: {model_id}")
                return model_set
        raise FileNotFoundError(f"Model set not found: {model_id}")

    def _create_job(self, kind: str, config: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        job_id = f"{kind}_{timestamp}_{suffix}"
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "progress": 0.0,
            "message": "Queued",
            "config": config,
            "output_dir": str(self._job_dir(job_id)),
            "completed_episodes": 0,
            "episode_count": int(config.get("num_episodes", 1)),
            "latest_episode_id": None,
            "generated_model_sets": [],
        }
        with self.lock:
            self.jobs[job_id] = job
        return job

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)

    def start_train_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        num_episodes = max(1, int(config.get("num_episodes", 5)))
        sample_interval = max(float(config.get("sample_interval", 5.0)), 0.5)
        use_gpu = _torch_cuda_available()
        job = self._create_job(
            "train",
            {
                "num_episodes": num_episodes,
                "sample_interval": sample_interval,
                "episode_duration": EPISODE_DURATION,
                "device": "cuda" if use_gpu else "cpu",
                "controller_mode": "heuristic_high_level + custom_ppo_low_level",
                "use_rl_low_level": True,
                "use_custom_ppo": True,
                "save_final_models": True,
                "data_source": "seeded_simulation",
            },
        )
        thread = threading.Thread(
            target=self._run_train_job,
            args=(job["id"], num_episodes, sample_interval, use_gpu),
            daemon=True,
        )
        thread.start()
        return job

    def start_test_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        model_id = str(config.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id is required")
        model_set = self.resolve_model_set(model_id)
        num_episodes = max(1, int(config.get("num_episodes", 1)))
        sample_interval = max(float(config.get("sample_interval", 5.0)), 0.5)
        base_seed = int(config.get("seed", 42))
        simulation_duration = float(config.get("simulation_duration", EPISODE_DURATION))

        job = self._create_job(
            "test",
            {
                "model_id": model_id,
                "num_episodes": num_episodes,
                "sample_interval": sample_interval,
                "seed": base_seed,
                "simulation_duration": simulation_duration,
                "deterministic_inference": True,
                "controller_mode": "heuristic_high_level + custom_ppo_low_level",
                "data_source": "seeded_simulation",
            },
        )
        thread = threading.Thread(
            target=self._run_test_job,
            args=(job["id"], model_set, num_episodes, sample_interval, base_seed, simulation_duration),
            daemon=True,
        )
        thread.start()
        return job

    def _run_train_job(self, job_id: str, num_episodes: int, sample_interval: float, use_gpu: bool) -> None:
        from train import TrainingManager, setup_logger

        job_dir = self._job_dir(job_id)
        episodes_dir = job_dir / "episodes"
        logger = setup_logger(str(job_dir / "logs"))
        self._update_job(
            job_id,
            status="running",
            started_at=_now_iso(),
            message="Training in progress",
        )

        def on_episode_complete(event: Dict[str, Any]) -> None:
            artifact = event.get("artifact") or {}
            self._update_job(
                job_id,
                progress=event["episode_index"] / max(num_episodes, 1),
                completed_episodes=event["episode_index"],
                latest_episode_id=artifact.get("episode_id"),
                message=f"Episode {event['episode_index']} completed",
            )

        try:
            manager = TrainingManager(
                num_episodes=num_episodes,
                use_gpu=use_gpu,
                enable_visualization=False,
                use_rl_low_level=True,
                use_custom_ppo=True,
                logger=logger,
                model_output_dir=str(self.models_dir),
                model_output_prefix=job_id,
                save_final_models=True,
                stats_output_path=str(job_dir / "training_stats.json"),
                episode_artifact_dir=str(episodes_dir),
                episode_sample_interval=sample_interval,
                run_label=job_id,
                on_episode_complete=on_episode_complete,
            )
            manager.train()
            self._update_job(
                job_id,
                status="completed",
                finished_at=_now_iso(),
                progress=1.0,
                completed_episodes=num_episodes,
                generated_model_sets=[f"{job_id}_best", f"{job_id}_final"],
                training_stats_path=str((job_dir / "training_stats.json").resolve()),
                message="Training finished",
            )
        except Exception as exc:
            logger.exception("Training job failed")
            self._update_job(
                job_id,
                status="failed",
                finished_at=_now_iso(),
                message=f"{type(exc).__name__}: {exc}",
                error=traceback.format_exc(),
            )

    def _run_test_job(
        self,
        job_id: str,
        model_set: Dict[str, Any],
        num_episodes: int,
        sample_interval: float,
        base_seed: int,
        simulation_duration: float,
    ) -> None:
        job_dir = self._job_dir(job_id)
        episodes_dir = job_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        self._update_job(
            job_id,
            status="running",
            started_at=_now_iso(),
            message="Test in progress",
        )

        try:
            from run_dual_ppo_visualization import DualPPOVisualizationRunner

            for episode_index in range(num_episodes):
                seed = base_seed + episode_index
                episode_id = f"{job_id}_ep{episode_index + 1:04d}"
                episode_dir = episodes_dir / episode_id
                runner = DualPPOVisualizationRunner(
                    model_paths={int(key): value for key, value in model_set["paths"].items()},
                    output_dir=episode_dir,
                    seed=seed,
                    simulation_duration=simulation_duration,
                    sample_interval=sample_interval,
                    summary_metadata={
                        "job_id": job_id,
                        "episode_id": episode_id,
                        "episode_index": episode_index + 1,
                        "run_label": job_id,
                        "selected_model_id": model_set["id"],
                    },
                )
                runner.run()
                self._update_job(
                    job_id,
                    progress=(episode_index + 1) / max(num_episodes, 1),
                    completed_episodes=episode_index + 1,
                    latest_episode_id=episode_id,
                    message=f"Episode {episode_index + 1} completed",
                )

            self._update_job(
                job_id,
                status="completed",
                finished_at=_now_iso(),
                progress=1.0,
                generated_model_sets=[model_set["id"]],
                training_stats_path=None,
                message="Test finished",
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_now_iso(),
                message=f"{type(exc).__name__}: {exc}",
                error=traceback.format_exc(),
            )


class DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, app: SimulationWebApp):
        super().__init__(server_address, request_handler_class)
        self.app = app


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/state":
            _json_response(self, 200, self.server.app.get_state())
            return

        if route.startswith("/api/episodes/"):
            episode_id = route.split("/")[-1]
            try:
                payload = self.server.app.load_episode(episode_id)
            except FileNotFoundError as exc:
                _json_response(self, 404, {"error": str(exc)})
                return
            _json_response(self, 200, payload)
            return

        if route in {"/", "/index.html", "/simulation_report.html"}:
            route = "/ui/simulation_report.html"

        self._serve_static_file(route)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "Invalid JSON body"})
            return

        try:
            if route == "/api/train":
                job = self.server.app.start_train_job(payload)
                _json_response(self, 202, job)
                return
            if route == "/api/test":
                job = self.server.app.start_test_job(payload)
                _json_response(self, 202, job)
                return
        except (ValueError, FileNotFoundError) as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(
                self,
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            return

        _json_response(self, 404, {"error": f"Unknown route: {route}"})

    def _serve_static_file(self, route: str) -> None:
        relative = unquote(route.lstrip("/"))
        target = (self.server.app.workspace / relative).resolve()

        if not str(target).startswith(str(self.server.app.workspace)):
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(str(target))
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + "; charset=utf-8" if target.suffix in {".html", ".js", ".json", ".css"} else (content_type or "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AGV simulation web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    app = SimulationWebApp(workspace)
    server = DashboardHTTPServer((args.host, args.port), DashboardRequestHandler, app)
    print(f"Dashboard running at http://{args.host}:{args.port}/simulation_report.html")
    print(f"UI source lives at http://{args.host}:{args.port}/ui/simulation_report.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
