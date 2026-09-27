"""Interactive visualizations for the Kevin Sun AI Swimming Health App.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from typing import Any
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _elapsed_minutes(timestamps: pd.Series) -> pd.Series:
    return (timestamps - timestamps.min()).dt.total_seconds() / 60.0


def _downsample(frame: pd.DataFrame, maximum_rows: int = 2500) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame.copy()
    step = max(1, len(frame) // maximum_rows)
    return frame.iloc[::step].copy()


def sensor_timeline_figure(session_data: pd.DataFrame) -> go.Figure:
    frame = _downsample(session_data.sort_values("timestamp"))
    frame["elapsed_min"] = _elapsed_minutes(frame["timestamp"])
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    if frame["heart_rate_bpm"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=frame["elapsed_min"],
                y=frame["heart_rate_bpm"],
                mode="lines",
                name="Heart rate",
                hovertemplate="%{x:.2f} min<br>%{y:.1f} bpm<extra></extra>",
            ),
            secondary_y=False,
        )
    if frame["spo2_pct"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=frame["elapsed_min"],
                y=frame["spo2_pct"],
                mode="lines",
                name="SpO2 sensor",
                hovertemplate="%{x:.2f} min<br>%{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title="Health-sensor timeline",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figure.update_xaxes(title_text="Elapsed time (minutes)")
    figure.update_yaxes(title_text="Heart rate (bpm)", secondary_y=False)
    figure.update_yaxes(title_text="SpO2 sensor (%)", secondary_y=True)
    return figure


def motion_timeline_figure(session_data: pd.DataFrame) -> go.Figure:
    frame = _downsample(session_data.sort_values("timestamp"))
    frame["elapsed_min"] = _elapsed_minutes(frame["timestamp"])
    plot_frame = frame[["elapsed_min", "acc_mag", "gyro_mag"]].melt(
        id_vars="elapsed_min",
        value_vars=["acc_mag", "gyro_mag"],
        var_name="signal",
        value_name="value",
    )
    plot_frame["signal"] = plot_frame["signal"].map(
        {"acc_mag": "Acceleration magnitude", "gyro_mag": "Gyroscope magnitude"}
    )
    figure = px.line(
        plot_frame,
        x="elapsed_min",
        y="value",
        facet_row="signal",
        title="Watch motion signals",
        labels={"elapsed_min": "Elapsed time (minutes)", "value": "Sensor magnitude"},
    )
    figure.update_yaxes(matches=None)
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    figure.update_layout(
        showlegend=False,
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return figure


def stroke_timeline_figure(
    session_data: pd.DataFrame,
    window_results: pd.DataFrame,
) -> go.Figure:
    frame = window_results.sort_values("window_start").copy()
    start_time = session_data["timestamp"].min()
    frame["elapsed_min"] = (frame["window_start"] - start_time).dt.total_seconds() / 60.0
    figure = px.scatter(
        frame,
        x="elapsed_min",
        y="predicted_stroke",
        color="prediction_confidence",
        size="prediction_confidence",
        size_max=18,
        range_color=[0, 1],
        title="AI stroke-classification timeline",
        labels={
            "elapsed_min": "Elapsed time (minutes)",
            "predicted_stroke": "Predicted stroke",
            "prediction_confidence": "Confidence",
        },
        hover_data={
            "window_start": True,
            "window_end": True,
            "stroke_rate_spm": ":.1f",
            "heart_rate_mean_bpm": ":.1f",
        },
    )
    figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=420)
    return figure


def stroke_distribution_figure(report: dict[str, Any]) -> go.Figure:
    distribution = report.get("predicted_stroke_distribution_pct", {})
    frame = pd.DataFrame(
        {
            "stroke": list(distribution.keys()),
            "percent": list(distribution.values()),
        }
    )
    if frame.empty:
        frame = pd.DataFrame({"stroke": ["Unavailable"], "percent": [100.0]})
    figure = px.pie(
        frame,
        names="stroke",
        values="percent",
        hole=0.58,
        title="Predicted stroke mix",
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    return figure


def score_figure(report: dict[str, Any]) -> go.Figure:
    summary = report["summary"]
    frame = pd.DataFrame(
        {
            "metric": ["Technique consistency", "Fatigue trend", "Data quality"],
            "score": [
                summary["technique_consistency_score_0_to_100"],
                summary["fatigue_trend_score_0_to_100"],
                summary["data_quality_score_0_to_100"],
            ],
        }
    )
    figure = px.bar(
        frame,
        x="score",
        y="metric",
        orientation="h",
        text="score",
        range_x=[0, 100],
        title="Session scores",
        labels={"score": "Score (0-100)", "metric": ""},
    )
    figure.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    figure.update_layout(margin=dict(l=10, r=30, t=55, b=10), height=390)
    return figure


def heart_rate_zone_figure(report: dict[str, Any]) -> go.Figure:
    zones = report.get("heart_rate_zones", {})
    frame = pd.DataFrame(
        {
            "zone": [f"Zone {index}" for index in range(1, 6)],
            "minutes": [float(zones.get(f"zone_{index}_minutes", 0.0)) for index in range(1, 6)],
        }
    )
    figure = px.bar(
        frame,
        x="zone",
        y="minutes",
        text="minutes",
        title="Estimated heart-rate zone time",
        labels={"zone": "Heart-rate zone", "minutes": "Minutes"},
    )
    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    return figure


def animated_swimmer_html(
    session_data: pd.DataFrame,
    window_results: pd.DataFrame,
    seconds: int = 8,
    fps: int = 12,
) -> str:
    """Return a self-contained HTML canvas animation driven by IMU predictions."""

    ordered = session_data.sort_values("timestamp").copy()
    if ordered.empty:
        return "<p>No animation data available.</p>"

    duration_samples = max(2, min(len(ordered), seconds * 20))
    clip = ordered.iloc[:duration_samples].copy()
    frame_count = max(2, seconds * fps)
    indexes = np.linspace(0, len(clip) - 1, frame_count).astype(int)
    frames = clip.iloc[indexes].reset_index(drop=True)

    labels = window_results[
        ["window_start", "predicted_stroke", "prediction_confidence"]
    ].sort_values("window_start")
    frames = pd.merge_asof(
        frames.sort_values("timestamp"),
        labels,
        left_on="timestamp",
        right_on="window_start",
        direction="nearest",
    )
    frames["predicted_stroke"] = frames["predicted_stroke"].fillna("unknown")
    frames["prediction_confidence"] = frames["prediction_confidence"].fillna(0.0)

    payload = []
    for row in frames.itertuples(index=False):
        payload.append(
            {
                "gx": float(getattr(row, "gyro_x", 0.0) or 0.0),
                "gy": float(getattr(row, "gyro_y", 0.0) or 0.0),
                "gz": float(getattr(row, "gyro_z", 0.0) or 0.0),
                "ay": float(getattr(row, "acc_y", 0.0) or 0.0),
                "hr": (
                    float(getattr(row, "heart_rate_bpm"))
                    if np.isfinite(getattr(row, "heart_rate_bpm", np.nan))
                    else None
                ),
                "stroke": str(getattr(row, "predicted_stroke", "unknown")),
                "confidence": float(getattr(row, "prediction_confidence", 0.0) or 0.0),
            }
        )

    data_json = json.dumps(payload, separators=(",", ":"))
    frame_interval = int(round(1000 / max(fps, 1)))

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, Arial, sans-serif; }}
  .wrap {{ border: 1px solid rgba(125,125,125,.22); border-radius: 18px; overflow: hidden;
           background: linear-gradient(160deg, #071a2b 0%, #0a4668 55%, #0c678a 100%);
           box-shadow: 0 14px 32px rgba(0,0,0,.18); }}
  canvas {{ width: 100%; height: 350px; display: block; }}
  .caption {{ color: rgba(255,255,255,.80); font-size: 12px; padding: 0 18px 14px; }}
</style>
</head>
<body>
<div class="wrap">
  <canvas id="swim" width="1000" height="350"></canvas>
  <div class="caption">Watch-driven educational animation — not a biomechanical reconstruction.</div>
</div>
<script>
const frames = {data_json};
const canvas = document.getElementById('swim');
const ctx = canvas.getContext('2d');
let i = 0;
function line(x1,y1,x2,y2,w,alpha=1) {{
  ctx.save(); ctx.globalAlpha=alpha; ctx.lineWidth=w; ctx.lineCap='round';
  ctx.strokeStyle='#eefbff'; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); ctx.restore();
}}
function circle(x,y,r,fill=false) {{
  ctx.save(); ctx.lineWidth=4; ctx.strokeStyle='#eefbff'; ctx.fillStyle='#ffdd8a';
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); fill ? ctx.fill() : ctx.stroke(); ctx.restore();
}}
function draw() {{
  const f = frames[i % frames.length];
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const grad = ctx.createLinearGradient(0,0,0,canvas.height);
  grad.addColorStop(0,'#082138'); grad.addColorStop(1,'#0b7093');
  ctx.fillStyle=grad; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='rgba(255,255,255,.18)'; ctx.lineWidth=2;
  for (let y=105; y<345; y+=42) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.bezierCurveTo(250,y-8,750,y+8,1000,y); ctx.stroke(); }}
  ctx.strokeStyle='rgba(255,255,255,.55)'; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(0,80); ctx.lineTo(1000,80); ctx.stroke();

  const phase = i / Math.max(frames.length-1,1) * Math.PI * 4;
  const bob = Math.sin(phase*2)*7 + Math.tanh(f.gy/160)*6;
  const cx=500, cy=205+bob;
  const shoulder={{x:430,y:cy-18}}, hip={{x:585,y:cy+10}}, neck={{x:405,y:cy-22}};
  const a1=Math.PI + .62*Math.tanh(f.gz/150) + .38*Math.sin(phase);
  const a2=Math.PI - .62*Math.tanh(f.gx/130) - .38*Math.sin(phase);
  const arm=118;
  const h1={{x:shoulder.x+arm*Math.cos(a1),y:shoulder.y+arm*Math.sin(a1)}};
  const h2={{x:shoulder.x+arm*Math.cos(a2),y:shoulder.y+arm*Math.sin(a2)}};
  const kick=.24*Math.tanh(f.ay/.5)+.18*Math.sin(phase*2);
  const footX=735;
  const lf={{x:footX,y:cy+25+kick*70}}, rf={{x:footX,y:cy+25-kick*70}};

  line(neck.x,neck.y,hip.x,hip.y,14);
  line(shoulder.x,shoulder.y,h1.x,h1.y,8);
  line(shoulder.x,shoulder.y,h2.x,h2.y,8,.90);
  line(hip.x,hip.y,lf.x,lf.y,9);
  line(hip.x,hip.y,rf.x,rf.y,9,.90);
  circle(373,cy-18,26,false);
  circle(h1.x,h1.y,8,true);

  ctx.fillStyle='#ffffff'; ctx.font='700 26px Inter, Arial';
  ctx.fillText('AI Swimming Motion View',35,42);
  ctx.font='600 20px Inter, Arial';
  ctx.fillText('Stroke: '+f.stroke.replaceAll('_',' '),35,135);
  ctx.fillText('Confidence: '+Math.round(f.confidence*100)+'%',35,168);
  ctx.fillText('Watch HR: '+(f.hr===null?'N/A':Math.round(f.hr)+' bpm'),35,201);
  ctx.font='500 15px Inter, Arial'; ctx.fillStyle='rgba(255,255,255,.76)';
  ctx.fillText('Author: Kevin Sun  |  Mentor: Dr. Qingyang Xiao',35,315);
  i=(i+1)%frames.length;
}}
draw(); setInterval(draw,{frame_interval});
</script>
</body>
</html>
"""
