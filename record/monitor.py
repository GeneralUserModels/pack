import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import time
from datetime import datetime

st.set_page_config(page_title="Session Monitor", layout="wide")

sessions_dir = Path(__file__).parent.parent / "logs"
TIME_WINDOW = 30.0  # seconds (used consistently)


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

last_ts_candidates = [
    get_last_timestamp(input_events_file),
    get_last_timestamp(screenshots_file),
    get_last_timestamp(ssim_file)
]
global_ref_ts = max([ts for ts in last_ts_candidates if ts is not None], default=time.time())

input_events = read_jsonl(input_events_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)
screenshots = read_jsonl(screenshots_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)
ssim_data = read_jsonl(ssim_file, time_window=TIME_WINDOW, ref_ts=global_ref_ts)

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
