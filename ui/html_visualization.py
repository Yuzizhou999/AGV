import json
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _build_episode_payload(
    *,
    episode: Optional[Dict[str, Any]] = None,
    frames=None,
    summary=None,
    statistics=None,
) -> Dict[str, Any]:
    if episode is not None:
        return _json_safe(episode)
    return {
        "frames": _json_safe(frames or []),
        "summary": _json_safe(summary or {}),
        "statistics": _json_safe(statistics or {}),
    }


def _load_template() -> str:
    template_path = Path(__file__).with_name("simulation_report.html")
    if not template_path.exists():
        raise FileNotFoundError(f"simulation_report.html template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def write_simulation_report(
    output_path,
    frames=None,
    summary=None,
    statistics=None,
    episode: Optional[Dict[str, Any]] = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_episode_payload(
        episode=episode,
        frames=frames,
        summary=summary,
        statistics=statistics,
    )
    bootstrap = {
        "mode": "embedded",
        "episode": payload,
        "generated_report": True,
    }

    template = _load_template()
    bootstrap_literal = json.dumps(bootstrap, ensure_ascii=False)
    html = template.replace(
        "window.__SIMULATION_BOOTSTRAP__ = null;",
        f"window.__SIMULATION_BOOTSTRAP__ = {bootstrap_literal};",
        1,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
