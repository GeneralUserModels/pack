import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import time
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Session Monitor", layout="wide")

sessions_dir = Path(__file__).parent.parent / "logs"
TIME_WINDOW = 30.0


def get_sessions():
    """Get all available session directories."""
    if not sessions_dir.exists():
        return []
    return reversed(sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()]))


def get_last_timestamp(file_path: Path):
    """Return the timestamp of the last non-empty line, or None."""
    if not file_path.exists():
        return None
    last_line = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

    if not last_line:
        return None

    try:
        entry = json.loads(last_line)
        return float(entry.get("timestamp", 0.0)) or None
    except Exception:
        return None


def read_jsonl(file_path: Path, time_window: float = TIME_WINDOW, ref_ts: float = None):
    if not file_path.exists():
        return []

    # If ref_ts is None, determine it from this file's last entry (old behavior)
    if ref_ts is None:
        last_ts = get_last_timestamp(file_path)
        if last_ts is None:
            return []
        cutoff_time = last_ts - float(time_window)
    else:
        cutoff_time = float(ref_ts) - float(time_window)

    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                try:
                    ts = float(entry.get("timestamp", 0.0))
                except Exception:
                    ts = 0.0
                if ts >= cutoff_time:
                    entries.append(entry)
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return []

    return entries


def create_timeline_chart(entries, title, value_key=None, time_window: float = TIME_WINDOW, reference_time: float = None):
    fig = go.Figure()

    if not reference_time:
        # default fallback: now
        reference_time = time.time()

    if entries:
        # Prepare timestamps as floats
        timestamps = []
        for e in entries:
            try:
                timestamps.append(float(e.get("timestamp", 0.0)))
            except Exception:
                timestamps.append(0.0)

        # compute relative times relative to reference_time (negative => seconds before newest)
        relative_times = [(t - reference_time) for t in timestamps]

        # Create vertical lines for each event
        for rt in relative_times:
            fig.add_trace(go.Scatter(
                x=[rt, rt],
                y=[0, 1],
                mode='lines',
                line=dict(width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add shaded area from first to last entry (within this chart)
        if len(relative_times) > 1:
            first_time = min(relative_times)
            last_time = max(relative_times)
            fig.add_trace(go.Scatter(
                x=[first_time, last_time, last_time, first_time],
                y=[0, 0, 1, 1],
                fill='toself',
                fillcolor='rgba(65, 105, 225, 0.12)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add value markers if provided
        if value_key:
            values = [e.get(value_key, 0) for e in entries]
            hover_text = [f"Time: {rt:.2f}s<br>Value: {v:.3f}"
                          for rt, v in zip(relative_times, values)]

            fig.add_trace(go.Scatter(
                x=relative_times,
                y=[0.5] * len(relative_times),
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

    # Update layout to always show [-time_window, 0] relative to the reference_time
    # and include the reference timestamp in the title for clarity.
    ref_dt = datetime.fromtimestamp(reference_time).isoformat()
    fig.update_layout(
        title=f"{title} (Last {int(time_window)}s) — ref: {ref_dt} — {len(entries)} events",
        xaxis=dict(
            title="Time (seconds from newest event)",
            range=[-time_window, 0],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=False,
            range=[0, 1],
            showgrid=False
        ),
        height=200,
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='white'
    )

    return fig


def read_jsonl(file_path: Path, time_window: float = TIME_WINDOW, ref_ts: float = None):
    if not file_path.exists():
        return []

    if ref_ts is None:
        last_ts = get_last_timestamp(file_path)
        if last_ts is None:
            return []
        cutoff_time = last_ts - float(time_window)
    else:
        cutoff_time = float(ref_ts) - float(time_window)

    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                try:
                    ts = float(entry.get("timestamp", 0.0))
                except Exception:
                    ts = 0.0
                if ts >= cutoff_time:
                    entries.append(entry)
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return []

    return entries

# --- New: read aggregated clusters ---


def read_aggregated_clusters(file_path: Path, time_window: float = TIME_WINDOW, ref_ts: float = None):
    """
    Parse aggregated_actions.jsonl into a list of cluster dicts:
    { "start": float_epoch, "end": float_epoch, "type": str, "count": int }
    Only returns clusters that overlap the time window relative to ref_ts (or last timestamp in file if ref_ts None).
    """
    if not file_path.exists():
        return []

    # determine cutoff_time based on ref_ts or file's last timestamp
    if ref_ts is None:
        # try last timestamp in this aggregated file (fall back to None)
        last_ts = get_last_timestamp(file_path)
        if last_ts is None:
            return []
        cutoff_time = last_ts - float(time_window)
    else:
        cutoff_time = float(ref_ts) - float(time_window)

    clusters = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                start = obj.get("cluster_start", None)
                end = obj.get("cluster_end", None)
                if start is None or end is None:
                    evs = obj.get("events", [])
                    times = [e.get("timestamp", e.get("time")) for e in evs if (e.get("timestamp", e.get("time")) is not None)]
                    if times:
                        start = start if start is not None else min(times)
                        end = end if end is not None else max(times)

                if start is None or end is None:
                    continue

                # filter by time window: include clusters if any part >= cutoff_time
                if end < cutoff_time:
                    continue

                cluster_type = obj.get("cluster_type", obj.get("action_type", "cluster"))
                count = obj.get("cluster_event_count", None)
                clusters.append({
                    "start": float(start),
                    "end": float(end),
                    "type": str(cluster_type),
                    "count": count if count is not None else len(obj.get("events", [])),
                })
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return []

    clusters.sort(key=lambda x: x["start"])
    return clusters


# --- Helper: color palette and utilities ---
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def hex_to_rgba(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_type_color_map(cluster_types: List[str]) -> Dict[str, str]:
    """Return deterministic mapping cluster_type -> hex color (cycles through PALETTE)."""
    unique = sorted(set(cluster_types))
    mapping = {}
    for i, t in enumerate(unique):
        mapping[t] = PALETTE[i % len(PALETTE)]
    return mapping

# --- New: Plotly chart for clusters (overlapping single row) ---


def create_clusters_timeline_chart(clusters: List[dict], title: str, time_window: float = TIME_WINDOW, reference_time: float = None):
    """
    clusters: list of {"start": epoch, "end": epoch, "type": str, "count": int}
    reference_time: epoch seconds to compute relative seconds (defaults to now)
    """
    fig = go.Figure()
    if reference_time is None:
        reference_time = time.time()

    if not clusters:
        # empty placeholder
        fig.update_layout(
            title=f"{title} — no clusters in the window",
            xaxis=dict(range=[-time_window, 0], title="Time (seconds from newest event)"),
            yaxis=dict(visible=False),
            height=200,
            margin=dict(l=50, r=50, t=70, b=50),
            plot_bgcolor='white'
        )
        return fig

    # stable color mapping per cluster_type
    types = [c["type"] for c in clusters]
    color_map = build_type_color_map(types)

    # compute relative times (seconds)
    rel_starts = [(c["start"] - reference_time) for c in clusters]
    rel_ends = [(c["end"] - reference_time) for c in clusters]

    full_span = max(rel_ends) - min(rel_starts) if rel_ends and rel_starts else time_window
    # clamp full_span to at least time_window to get reasonable offsets for labels
    if full_span < 1e-6:
        full_span = time_window

    # shapes for rectangles and vertical lines
    shapes = []
    hover_x = []
    hover_y = []
    hover_texts = []

    for c, rs, re in zip(clusters, rel_starts, rel_ends):
        color = color_map.get(c["type"], PALETTE[0])
        fillcolor = hex_to_rgba(color, 0.35)
        outline_color = hex_to_rgba(color, 1.0)

        # small positive width for zero-length clusters
        if re <= rs:
            re = rs + 1e-3

        # rectangle occupies y 0.2..0.8
        shapes.append({
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": rs, "x1": re,
            "y0": 0.2, "y1": 0.8,
            "fillcolor": fillcolor,
            "line": {"width": 1, "color": outline_color},
            "layer": "below",
        })

        # vertical start + end lines (same color)
        shapes.append({
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": rs, "x1": rs,
            "y0": 0.15, "y1": 0.85,
            "line": {"color": outline_color, "width": 2},
        })
        shapes.append({
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": re, "x1": re,
            "y0": 0.15, "y1": 0.85,
            "line": {"color": outline_color, "width": 2},
        })

        # hover point (middle) so user can hover for info
        mid = 0.5 * (rs + re)
        hover_x.append(mid)
        hover_y.append(0.5)
        start_ts = datetime.fromtimestamp(c["start"]).isoformat()
        end_ts = datetime.fromtimestamp(c["end"]).isoformat()
        hover_texts.append(f"Type: {c['type']}<br>Count: {c['count']}<br>start: {start_ts}<br>end: {end_ts}")

        # annotation label (placed just to the right)
        label_x = re + 0.02 * full_span
        fig.add_annotation(
            x=label_x, y=0.5,
            xref="x", yref="y",
            text=f"{c['type']} ({c['count']})",
            showarrow=False,
            font=dict(size=10),
            align="left"
        )

    # add shapes to layout
    fig.update_layout(shapes=shapes)

    # add invisible scatter for hover
    fig.add_trace(go.Scatter(
        x=hover_x,
        y=hover_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),  # invisible
        hoverinfo="text",
        text=hover_texts,
        showlegend=False
    ))

    # x axis range [-time_window, 0]
    fig.update_xaxes(title_text="Time (seconds from newest event)", range=[-time_window, 0], gridcolor='lightgray')
    fig.update_yaxes(visible=False, range=[0, 1])

    ref_dt = datetime.fromtimestamp(reference_time).isoformat()
    fig.update_layout(
        title=f"{title} (Last {int(time_window)}s) — ref: {ref_dt} — {len(clusters)} clusters",
        height=220,
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='white'
    )

    return fig

# --- Main flow (session selection, file paths) ---


# Title
st.title("📊 Realtime Session Monitor")

# Session selector
sessions = get_sessions()
if not sessions:
    st.error(f"No sessions found in {sessions_dir}")
    st.stop()

selected_session = st.selectbox("Select Session", sessions)

# Auto-refresh control
col1, col2 = st.columns([3, 1])
with col1:
    refresh_rate = st.slider("Refresh rate (seconds)", 0.1, 5.0, 1.0, 0.1)
with col2:
    auto_refresh = st.checkbox("Auto-refresh", value=True)

session_path = sessions_dir / selected_session

input_events_file = session_path / "input_events.jsonl"
screenshots_file = session_path / "screenshots.jsonl"
ssim_file = session_path / "ssim.jsonl"
aggregated_file = session_path / "aggregated_actions.jsonl"  # <<-- new

last_ts_candidates = [
    get_last_timestamp(input_events_file),
    get_last_timestamp(screenshots_file),
    get_last_timestamp(ssim_file),
    get_last_timestamp(aggregated_file)
]
global_ref_ts = max([ts for ts in last_ts_candidates if ts is not None], default=time.time())

input_events = read_jsonl(input_events_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)
screenshots = read_jsonl(screenshots_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)
ssim_data = read_jsonl(ssim_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)
aggregated_clusters = read_aggregated_clusters(aggregated_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)

chart_container = st.container()

with chart_container:
    st.subheader("Input Events")
    fig1 = create_timeline_chart(input_events, "Input Events", time_window=TIME_WINDOW, reference_time=global_ref_ts)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Screenshots")
    fig2 = create_timeline_chart(screenshots, "Screenshots", time_window=TIME_WINDOW, reference_time=global_ref_ts)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SSIM Values")
    fig3 = create_timeline_chart(ssim_data, "SSIM Values", value_key='ssim_value', time_window=TIME_WINDOW, reference_time=global_ref_ts)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Aggregated Clusters (overlapping)")
    fig4 = create_clusters_timeline_chart(aggregated_clusters, "Aggregated Actions (clusters)", time_window=TIME_WINDOW, reference_time=global_ref_ts)
    st.plotly_chart(fig4, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Input Events (30s)", len(input_events))
with col2:
    st.metric("Screenshots (30s)", len(screenshots))
with col3:
    if ssim_data:
        avg_ssim = sum(e.get('ssim_value', 0) for e in ssim_data) / len(ssim_data)
        st.metric("Avg SSIM (30s)", f"{avg_ssim:.3f}")
    else:
        st.metric("Avg SSIM (30s)", "N/A")

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
