import json
from pathlib import Path


def _json_safe(value):
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


def write_simulation_report(output_path, frames, summary):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_frames = _json_safe(frames)
    safe_summary = _json_safe(summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dual PPO Simulation Report</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffaf0;
      --ink: #1e2a2f;
      --accent: #176087;
      --accent-soft: #c7dfea;
      --track: #d8c9a7;
      --load: #4b9f5f;
      --unload: #4d79c7;
      --warn: #bf5a36;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f8f4ea 0%, #efe5d0 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }}
    .title {{
      margin: 0;
      font-size: 30px;
    }}
    .subtitle {{
      margin: 6px 0 0;
      color: #4d5b63;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 20px;
    }}
    .panel {{
      background: rgba(255, 250, 240, 0.9);
      border: 1px solid rgba(23, 96, 135, 0.15);
      border-radius: 18px;
      box-shadow: 0 16px 40px rgba(61, 53, 39, 0.08);
      padding: 18px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
    }}
    button, select {{
      border: none;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-size: 14px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: radial-gradient(circle at 50% 50%, #fffaf0 0%, #efe4cd 100%);
      border-radius: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .metric {{
      background: white;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid rgba(23, 96, 135, 0.12);
    }}
    .metric-label {{
      display: block;
      font-size: 12px;
      color: #5c6b73;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-value {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
      font-weight: 700;
    }}
    .vehicle-list {{
      display: grid;
      gap: 10px;
    }}
    .alert-list {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}
    .vehicle-card {{
      background: white;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid rgba(23, 96, 135, 0.12);
    }}
    .alert-card {{
      background: white;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid rgba(191, 90, 54, 0.18);
    }}
    .section-title {{
      margin: 18px 0 10px;
      font-size: 15px;
      color: #4d5b63;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 13px;
      color: #4d5b63;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
    }}
    .legend .vehicle0::before {{ background: #d4513d; }}
    .legend .vehicle1::before {{ background: #c48d28; }}
    .legend .loading::before {{ background: var(--load); }}
    .legend .unloading::before {{ background: var(--unload); }}
    .legend .busy::before {{ background: var(--warn); }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1 class="title">Dual PPO 8h Simulation Playback</h1>
        <p class="subtitle">Heuristic high-level control with per-vehicle custom PPO low-level weights</p>
      </div>
      <div id="summaryMetrics"></div>
    </div>
    <div class="layout">
      <section class="panel">
        <div class="controls">
          <button id="playButton">Play</button>
          <input id="timeSlider" type="range" min="0" max="{max(len(frames) - 1, 0)}" step="1" value="0" />
          <span id="timeLabel">00:00:00</span>
          <select id="speedSelect">
            <option value="1">1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </div>
        <svg id="trackCanvas" viewBox="-90 -90 180 180" aria-label="Track playback"></svg>
        <div class="legend">
          <span class="vehicle0">Vehicle 0</span>
          <span class="vehicle1">Vehicle 1</span>
          <span class="loading">Loading station</span>
          <span class="unloading">Unloading station</span>
          <span class="busy">Loading / unloading</span>
        </div>
      </section>
      <aside class="panel">
        <div class="metrics">
          <div class="metric"><span class="metric-label">Completed</span><span class="metric-value" id="completedValue">0</span></div>
          <div class="metric"><span class="metric-label">Timed Out</span><span class="metric-value" id="timeoutValue">0</span></div>
          <div class="metric"><span class="metric-label">Active Cargos</span><span class="metric-value" id="activeValue">0</span></div>
          <div class="metric"><span class="metric-label">Sampled Frames</span><span class="metric-value" id="frameCountValue">{len(frames)}</span></div>
          <div class="metric"><span class="metric-label">Warnings</span><span class="metric-value" id="warningValue">0</span></div>
          <div class="metric"><span class="metric-label">Collisions</span><span class="metric-value" id="collisionValue">0</span></div>
        </div>
        <h3 class="section-title">Recent Alerts</h3>
        <div class="alert-list" id="recentAlerts"></div>
        <h3 class="section-title">Vehicles</h3>
        <div class="vehicle-list" id="vehicleList"></div>
      </aside>
    </div>
  </div>
  <script>
    const FRAME_DATA = {json.dumps(safe_frames, ensure_ascii=False)};
    const SUMMARY_DATA = {json.dumps(safe_summary, ensure_ascii=False)};
    const trackLength = SUMMARY_DATA.track_length || 100.0;
    const loadingPositions = SUMMARY_DATA.loading_positions || [];
    const unloadingPositions = SUMMARY_DATA.unloading_positions || [];
    const slider = document.getElementById("timeSlider");
    const playButton = document.getElementById("playButton");
    const speedSelect = document.getElementById("speedSelect");
    const timeLabel = document.getElementById("timeLabel");
    const completedValue = document.getElementById("completedValue");
    const timeoutValue = document.getElementById("timeoutValue");
    const activeValue = document.getElementById("activeValue");
    const warningValue = document.getElementById("warningValue");
    const collisionValue = document.getElementById("collisionValue");
    const recentAlerts = document.getElementById("recentAlerts");
    const vehicleList = document.getElementById("vehicleList");
    const summaryMetrics = document.getElementById("summaryMetrics");
    const trackCanvas = document.getElementById("trackCanvas");
    let currentIndex = 0;
    let playing = false;
    let timerId = null;

    function formatTime(totalSeconds) {{
      const seconds = Math.max(0, Math.floor(totalSeconds));
      const hh = String(Math.floor(seconds / 3600)).padStart(2, "0");
      const mm = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
      const ss = String(seconds % 60).padStart(2, "0");
      return `${{hh}}:${{mm}}:${{ss}}`;
    }}

    function toPoint(position) {{
      const radius = 60;
      const angle = (-2 * Math.PI * position / trackLength) + Math.PI / 2;
      return {{
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
      }};
    }}

    function drawTrack(frame) {{
      trackCanvas.innerHTML = "";
      const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      track.setAttribute("cx", "0");
      track.setAttribute("cy", "0");
      track.setAttribute("r", "60");
      track.setAttribute("fill", "none");
      track.setAttribute("stroke", "#b5964b");
      track.setAttribute("stroke-width", "6");
      trackCanvas.appendChild(track);

      loadingPositions.forEach((position, index) => {{
        const point = toPoint(position);
        const node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        node.setAttribute("cx", point.x);
        node.setAttribute("cy", point.y);
        node.setAttribute("r", "5");
        node.setAttribute("fill", "#4b9f5f");
        trackCanvas.appendChild(node);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", point.x + 6);
        label.setAttribute("y", point.y - 6);
        label.setAttribute("font-size", "5");
        label.textContent = `L${{index}}`;
        trackCanvas.appendChild(label);
      }});

      unloadingPositions.forEach((position, index) => {{
        const point = toPoint(position);
        const node = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        node.setAttribute("x", point.x - 4);
        node.setAttribute("y", point.y - 4);
        node.setAttribute("width", "8");
        node.setAttribute("height", "8");
        node.setAttribute("rx", "2");
        node.setAttribute("fill", "#4d79c7");
        trackCanvas.appendChild(node);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", point.x + 6);
        label.setAttribute("y", point.y + 10);
        label.setAttribute("font-size", "5");
        label.textContent = `U${{index}}`;
        trackCanvas.appendChild(label);
      }});

      (frame.vehicles || []).forEach((vehicle, index) => {{
        const point = toPoint(vehicle.position);
        const body = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        body.setAttribute("cx", point.x);
        body.setAttribute("cy", point.y);
        body.setAttribute("r", "6");
        body.setAttribute("fill", index === 0 ? "#d4513d" : "#c48d28");
        body.setAttribute("stroke", vehicle.is_loading_unloading ? "#bf5a36" : "#20303a");
        body.setAttribute("stroke-width", vehicle.is_loading_unloading ? "3" : "1.5");
        trackCanvas.appendChild(body);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", point.x);
        label.setAttribute("y", point.y + 1.8);
        label.setAttribute("font-size", "4");
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("fill", "white");
        label.textContent = `V${{vehicle.id}}`;
        trackCanvas.appendChild(label);
      }});
    }}

    function renderVehicleCards(frame) {{
      vehicleList.innerHTML = "";
      (frame.vehicles || []).forEach((vehicle) => {{
        const card = document.createElement("div");
        card.className = "vehicle-card";
        card.innerHTML = `
          <strong>Vehicle ${{vehicle.id}}</strong><br/>
          Position: ${{vehicle.position.toFixed(2)}}<br/>
          Velocity: ${{vehicle.velocity.toFixed(2)}}<br/>
          Slots occupied: ${{vehicle.slot_count}}<br/>
          State: ${{vehicle.is_loading_unloading ? "Busy" : "Ready"}}
        `;
        vehicleList.appendChild(card);
      }});
    }}

    function renderAlerts(frame) {{
      recentAlerts.innerHTML = "";
      const alerts = frame.recent_alerts || [];
      if (alerts.length === 0) {{
        const empty = document.createElement("div");
        empty.className = "alert-card";
        empty.textContent = "No recent alerts in this frame.";
        recentAlerts.appendChild(empty);
        return;
      }}

      alerts.forEach((alert) => {{
        const card = document.createElement("div");
        card.className = "alert-card";
        card.innerHTML = `
          <strong>${{String(alert.level).toUpperCase()}}</strong><br/>
          Time: ${{formatTime(alert.time || 0)}}<br/>
          Vehicles: V${{(alert.vehicle_ids || []).join(" / V")}}<br/>
          Distance: ${{Number(alert.distance || 0).toFixed(3)}}
        `;
        recentAlerts.appendChild(card);
      }});
    }}

    function renderSummary(frame) {{
      timeLabel.textContent = formatTime(frame.time || 0);
      completedValue.textContent = frame.completed_cargos ?? 0;
      timeoutValue.textContent = frame.timed_out_cargos ?? 0;
      activeValue.textContent = frame.active_cargos ?? 0;
      warningValue.textContent = frame.safety_warning_count ?? 0;
      collisionValue.textContent = frame.collision_alert_count ?? 0;
      summaryMetrics.innerHTML = `
        <strong>Seed:</strong> ${{SUMMARY_DATA.seed}}<br/>
        <strong>Sampling:</strong> ${{SUMMARY_DATA.visualization_sampling_interval}}s<br/>
        <strong>Deterministic:</strong> ${{SUMMARY_DATA.deterministic_inference}}
      `;
    }}

    function renderFrame(index) {{
      currentIndex = index;
      slider.value = String(index);
      const frame = FRAME_DATA[index] || FRAME_DATA[0] || {{
        time: 0,
        vehicles: [],
        completed_cargos: 0,
        timed_out_cargos: 0,
        active_cargos: 0,
      }};
      drawTrack(frame);
      renderVehicleCards(frame);
      renderAlerts(frame);
      renderSummary(frame);
    }}

    function stopPlayback() {{
      if (timerId !== null) {{
        window.clearInterval(timerId);
        timerId = null;
      }}
      playing = false;
      playButton.textContent = "Play";
    }}

    function startPlayback() {{
      stopPlayback();
      const speed = Number(speedSelect.value);
      timerId = window.setInterval(() => {{
        if (currentIndex >= FRAME_DATA.length - 1) {{
          stopPlayback();
          return;
        }}
        renderFrame(currentIndex + 1);
      }}, 1000 / speed);
      playing = true;
      playButton.textContent = "Pause";
    }}

    playButton.addEventListener("click", () => {{
      if (playing) {{
        stopPlayback();
      }} else {{
        startPlayback();
      }}
    }});

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      renderFrame(Number(event.target.value));
    }});

    renderFrame(0);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
