from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

from aggregate.logs import calculate_breaks, aggregate_logs
from modules.raw_log import RawLogEvents


PERCENTILE = 95

EVENT_Y = {
    "mouse_move": 1,
    "mouse_down": 2,
    "mouse_scroll": 3,
    "keyboard_press": 4,
}

SPECIAL_KEYS = {
    'enter': '⏎', 'return': '⏎', 'backspace': '⌫', 'space': '␣',
    'tab': '⇥', 'shift': '⇧', 'ctrl': '⌃', 'alt': '⎇',
    'cmd': '⌘', 'esc': '⎋', 'escape': '⎋',
    'up': '↑', 'down': '↓', 'left': '←', 'right': '→',
}

EVENT_COLORS = {
    "mouse_move": "#1f77b4",
    "mouse_down": "#ff7f0e",
    "mouse_scroll": "#2ca02c",
    "keyboard_press": "#d62728"
}


def extract_key(details: dict, pretty=False) -> str:
    """Extract key from event details, with optional pretty formatting."""
    key = details.get('key', '')
    if not key:
        return ''
    key = key.replace('Key.', '')
    return SPECIAL_KEYS.get(key, key) if pretty else key


def prepare_plot_data(logs: RawLogEvents):
    """Prepare data for plotting from RawLogEvents."""
    timestamps = []
    y_positions = []
    labels = []

    for event in logs:
        if event.event_type not in EVENT_Y:
            continue
        ts = datetime.strptime(event.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        timestamps.append(ts)
        y_positions.append(EVENT_Y[event.event_type])

        if event.event_type == 'keyboard_press':
            labels.append(extract_key(event.details, pretty=True))
        else:
            labels.append('')

    return timestamps, y_positions, labels


def create_interactive_plot(logs: RawLogEvents):
    timestamps, breaks, durations, thresholds = calculate_breaks(logs, PERCENTILE)
    aggregated = aggregate_logs(logs, breaks)

    plot_timestamps, y_positions, labels = prepare_plot_data(logs)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[
            'Event Timeline with Break Spans',
            'Event Count Over Time',
            'Break Duration Distribution',
            'Aggregated AggregatedLog Spans'
        ]
    )

    for event_type in EVENT_Y:
        event_timestamps = [ts for ts, y in zip(plot_timestamps, y_positions) if y == EVENT_Y[event_type]]
        event_labels = [lbl for ts, y, lbl in zip(plot_timestamps, y_positions, labels) if y == EVENT_Y[event_type]]

        fig.add_trace(
            go.Scatter(
                x=event_timestamps,
                y=[EVENT_Y[event_type]] * len(event_timestamps),
                mode='markers+text',
                marker=dict(color=EVENT_COLORS[event_type], size=6),
                text=event_labels,
                textposition="top center",
                textfont=dict(size=8),
                name=event_type,
                showlegend=True,
                hovertemplate=f"<b>{event_type}</b><br>Time: %{{x}}<br>Key: %{{text}}<extra></extra>"
            ),
            row=1, col=1
        )

    for event_type, spans in breaks.items():
        for start_time, end_time in spans:
            if event_type not in EVENT_Y:
                continue
            fig.add_shape(
                type="rect",
                x0=start_time, x1=end_time,
                y0=EVENT_Y[event_type] - 0.4, y1=EVENT_Y[event_type] + 0.4,
                fillcolor=EVENT_COLORS[event_type],
                opacity=0.2,
                line_width=0,
                row=1, col=1
            )

    df = pd.DataFrame({
        'timestamp': plot_timestamps,
        'event': [list(EVENT_Y.keys())[list(EVENT_Y.values()).index(y)] for y in y_positions]
    })
    df['timestamp_bin'] = pd.cut(df['timestamp'], bins=100)

    hist_data = df.groupby(['timestamp_bin', 'event'], observed=False).size().unstack(fill_value=0)

    bin_centers = [interval.mid for interval in hist_data.index]

    for event_type in EVENT_Y:
        if event_type in hist_data.columns:
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist_data[event_type],
                    name=f"{event_type} count",
                    marker_color=EVENT_COLORS[event_type],
                    showlegend=False,
                    hovertemplate=f"<b>{event_type}</b><br>Count: %{{y}}<extra></extra>"
                ),
                row=2, col=1
            )

        # fig.add_trace(
        #     go.Histogram(
        #         x=all_durations,
        #         nbinsx=50,
        #         name="Duration Distribution",
        #         marker_color="lightblue",
        #         showlegend=False,
        #         hovertemplate="Duration: %{x:.2f}s<br>Count: %{y}<extra></extra>"
        #     ),
        #     row=3, col=1
        # )

    for event_type, thresh in thresholds.items():
        if event_type not in EVENT_Y:
            continue
        if thresh > 0:
            fig.add_vline(
                x=thresh,
                line_dash="dash",
                line_color=EVENT_COLORS[event_type],
                annotation_text=f"{event_type} {PERCENTILE}th",
                row=3, col=1
            )

    for idx, log in enumerate(aggregated):
        start = datetime.strptime(log.start_timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        end = datetime.strptime(log.end_timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        keys = ''.join([SPECIAL_KEYS.get(k.replace("Key.", ""), k.replace("Key.", "")) for k in log.keys_pressed if k])

        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[idx + 1, idx + 1],
                mode='lines+markers+text',
                line=dict(color='blue', width=8),
                marker=dict(size=8, color='blue'),
                text=['', keys],
                textposition="middle right",
                textfont=dict(size=10),
                name=f"Log {idx + 1}",
                showlegend=False,
                hovertemplate=f"<b>Log {idx + 1}</b><br>Start: {start}<br>End: {end}<br>Keys: {keys}<extra></extra>"
            ),
            row=4, col=1
        )

    fig.update_layout(
        height=1200,
        title_text="Interactive Event Timeline Dashboard",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Event Type", row=1, col=1,
                     tickmode='array', tickvals=list(EVENT_Y.values()),
                     ticktext=list(EVENT_Y.keys()))
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_yaxes(title_text="Log ID", row=4, col=1,
                     tickmode='array', tickvals=list(range(1, len(aggregated) + 1)),
                     ticktext=[f"Log {i}" for i in range(1, len(aggregated) + 1)])

    fig.update_xaxes(title_text="Time", row=4, col=1)

    fig.update_layout(
        xaxis4=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    fig.update_layout(
        dragmode='zoom',
        selectdirection='h'
    )

    return fig, aggregated


def plot_interactive(events_path: Path):
    logs = RawLogEvents().load(events_path)
    logs.sort()

    fig, aggregated = create_interactive_plot(logs)

    fig.show()

    fig.write_html(events_path.with_suffix(f'.{PERCENTILE}_perc_interactive.html'))
    print(f"Interactive plot saved to: {events_path.with_suffix(f'.{PERCENTILE}_perc_interactive.html')}")

    return aggregated


if __name__ == '__main__':
    # path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-11_04-03-47-306009' / 'events.jsonl'
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-13_15-59-04-565176' / 'events.jsonl'
    plot_interactive(path)
