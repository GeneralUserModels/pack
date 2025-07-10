import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import pandas as pd

from observers.logs import EventLog

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


def load_events(events_path: Path, event_y: dict) -> list:
    events = []
    with open(events_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('event') in event_y:
                events.append(data)
    return events


def extract_key(details: dict, pretty=False) -> str:
    key = details.get('key', '')
    if not key:
        return
    key = key.replace('Key.', '')
    return SPECIAL_KEYS.get(key, key) if pretty else key


def calculate_breaks(events: list, event_y: dict, percentile: int):
    timestamps, y_positions, labels = [], [], []
    per_event_ts = {evt: [] for evt in event_y}

    for ev in events:
        ts = datetime.strptime(ev['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
        evt = ev['event']
        timestamps.append(ts)
        y_positions.append(event_y[evt])
        per_event_ts[evt].append(ts)
        labels.append(extract_key(ev.get('details', {}), pretty=(evt == 'keyboard_press')))

    breaks, durations, thresholds = {}, {}, {}
    for evt, ts_list in per_event_ts.items():
        if evt == "mouse_down":
            continue
        if len(ts_list) < 2:
            breaks[evt], durations[evt], thresholds[evt] = [], [], 0
            continue
        dt = np.array([t.timestamp() for t in ts_list])
        deltas = np.diff(dt)
        thresh = np.percentile(deltas, percentile)
        thresholds[evt] = thresh
        idx = np.where(deltas >= thresh)[0]
        breaks[evt] = [(ts_list[i], ts_list[i + 1]) for i in idx]
        durations[evt] = deltas.tolist()

    MOUSE_DOWN_DEBOUNCE = 500  # milliseconds
    MOUSE_DOWN_POS_OFFSET = 0.01  # % of screen height, width
    mouse_down_events = [e for e in events if e["event"] == "mouse_down"]

    processed_mouse_events = []
    last_pos, last_time = None, None

    def remove_lst_double_click(ts, pos, events):
        for i in range(len(events) - 1, -1, -1):
            evt_ts = datetime.strptime(events[i]['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
            if evt_ts == ts and events[i]['details'].get('position') == pos:
                del events[i]
        return events

    for evt in mouse_down_events:
        ts = datetime.strptime(evt['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
        button = evt['details'].get('button', 'unknown')

        if "left" not in button.lower():
            processed_mouse_events.append(evt)
            continue

        pos = evt.get('details', {}).get('position', [0, 0])
        monitor = evt.get('monitor', {})
        monitor_dim = [monitor.get("width", 1920), monitor.get("height", 1080)]

        new_evt = evt.copy()
        new_evt['details'] = evt['details'].copy()

        if last_pos is None or last_time is None:
            last_pos, last_time = pos, ts
            processed_mouse_events.append(new_evt)
            continue

        time_diff = (ts - last_time).total_seconds() * 1000

        if (time_diff <= MOUSE_DOWN_DEBOUNCE and
            abs(pos[0] - last_pos[0]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[0] and
                abs(pos[1] - last_pos[1]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[1]):

            new_evt['details']['double_click'] = True
            processed_mouse_events.append(new_evt)
            processed_mouse_events = remove_lst_double_click(last_time, last_pos, processed_mouse_events)
        else:
            last_pos, last_time = pos, ts
            processed_mouse_events.append(new_evt)

    mouse_down_timestamps = []
    for evt in processed_mouse_events:
        ts = datetime.strptime(evt['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
        mouse_down_timestamps.append(ts)

    mouse_down_breaks = []
    if len(mouse_down_timestamps) > 1:
        for i in range(len(mouse_down_timestamps) - 1):
            mouse_down_breaks.append((mouse_down_timestamps[i], mouse_down_timestamps[i + 1]))

    breaks["mouse_down"] = mouse_down_breaks
    durations["mouse_down"] = []
    thresholds["mouse_down"] = 0

    events_dict = {(e['timestamp'], e['event']): e for e in events}
    for proc_evt in processed_mouse_events:
        key = (proc_evt['timestamp'], proc_evt['event'])
        if key in events_dict:
            events_dict[key]['details'] = proc_evt['details']

    return timestamps, y_positions, labels, breaks, durations, thresholds


def _aggregate_event(events_slice: list) -> EventLog:
    cursor_pos = [ev.get("details", {}).get('position', 0) for ev in events_slice if ev.get("details", {}).get('position', 0)]
    return EventLog(
        start_timestamp=events_slice[0]['timestamp'],
        end_timestamp=events_slice[-1]['timestamp'],
        monitor=events_slice[0]['monitor'],
        start_screenshot_path=events_slice[0].get('screenshot_path'),
        end_screenshot_path=events_slice[-1].get('screenshot_path'),
        start_cursor_pos=cursor_pos[0] if cursor_pos else None,
        end_cursor_pos=cursor_pos[-1] if cursor_pos else None,
        click_positions=[ev.get("details", {}) for ev in events_slice if ev.get("event") == "mouse_down" and ev.get("details", {}).get('position')],
        scroll_directions=list([ev.get('details', {}).get("scroll") for ev in events_slice if ev.get('event') == 'mouse_scroll' and ev.get('details', {}).get("scroll")]),
        keys_pressed=[extract_key(ev['details']) for ev in events_slice if ev['event'] == 'keyboard_press' and extract_key(ev['details'])],
        events=[{"timestamp": ev["timestamp"], "event_type": ev['event'], "details": ev.get("details", {})} for ev in events_slice]
    )


def aggregate_logs(events: list, breaks: dict):
    events_sorted = sorted(events, key=lambda e: e['timestamp'])
    edges = sorted({
        ts.timestamp()
        for spans in breaks.values()
        for start, end in spans
        for ts in (start, end)
    })

    logs = []
    start_idx = 0

    for edge in edges:
        for rel_idx, event in enumerate(events_sorted[start_idx:], start=start_idx):
            ev_time = datetime.strptime(event['timestamp'], "%Y-%m-%d_%H-%M-%S-%f").timestamp()
            switched_screen = json.dumps(event.get("monitor", {})) != json.dumps(events_sorted[start_idx]['monitor'])
            if ev_time == edge or switched_screen:
                cut_idx = rel_idx
                break
        else:
            cut_idx = len(events_sorted) - 1
            switched_screen = False

        last_event = events_sorted[cut_idx]
        evt_type = last_event['event']

        if evt_type == 'mouse_down' and not switched_screen:
            button = last_event['details'].get('button')
            for lookahead in events_sorted[cut_idx + 1:]:
                # TODO: Threshold
                if lookahead['event'] == 'mouse_up' and lookahead['details'].get('button') == button:
                    cut_idx = events_sorted.index(lookahead)
                    break

        elif evt_type == 'keyboard_press' and not switched_screen:
            key = extract_key(last_event['details'], pretty=False)
            for lookahead in events_sorted[cut_idx + 1:]:
                # TODO: Threshold
                if lookahead['event'] == 'keyboard_release' and extract_key(lookahead['details'], pretty=False) == key:
                    cut_idx = events_sorted.index(lookahead)
                    break

        logs.append(_aggregate_event(events_sorted[start_idx:cut_idx + 1]))
        start_idx = cut_idx + 1

        if start_idx >= len(events_sorted):
            break

    if start_idx < len(events_sorted):
        logs.append(_aggregate_event(events_sorted[start_idx:]))

    return logs


def create_interactive_plot(events_path: Path):
    events = load_events(events_path, EVENT_Y)
    timestamps, y_positions, labels, breaks, durations, thresholds = calculate_breaks(events, EVENT_Y, PERCENTILE)
    aggregated = aggregate_logs(events, breaks)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[
            'Event Timeline with Break Spans',
            'Event Count Over Time',
            'Break Duration Distribution',
            'Aggregated EventLog Spans'
        ]
    )

    for event_type in EVENT_Y:
        event_timestamps = [ts for ts, y in zip(timestamps, y_positions) if y == EVENT_Y[event_type]]
        event_labels = [lbl for ts, y, lbl in zip(timestamps, y_positions, labels) if y == EVENT_Y[event_type]]

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
            fig.add_shape(
                type="rect",
                x0=start_time, x1=end_time,
                y0=EVENT_Y[event_type] - 0.4, y1=EVENT_Y[event_type] + 0.4,
                fillcolor=EVENT_COLORS[event_type],
                opacity=0.2,
                line_width=0,
                row=1, col=1
            )

    df = pd.DataFrame({'timestamp': timestamps, 'event': [list(EVENT_Y.keys())[list(EVENT_Y.values()).index(y)] for y in y_positions]})
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

    all_durations = []
    duration_events = []

    for event_type, durs in durations.items():
        all_durations.extend(durs)
        duration_events.extend([event_type] * len(durs))

    fig.add_trace(
        go.Histogram(
            x=all_durations,
            nbinsx=50,
            name="Duration Distribution",
            marker_color="lightblue",
            showlegend=False,
            hovertemplate="Duration: %{x:.2f}s<br>Count: %{y}<extra></extra>"
        ),
        row=3, col=1
    )

    for event_type, thresh in thresholds.items():
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
        keys = ''.join([SPECIAL_KEYS.get(k, k) for k in log.keys_pressed])

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
    fig, aggregated = create_interactive_plot(events_path)

    fig.show()

    with open(events_path.with_suffix(f'.agg_{PERCENTILE}.json'), 'w') as f:
        json.dump([log.to_dict() for log in aggregated], f, indent=2)

    fig.write_html(events_path.with_suffix(f'.{PERCENTILE}_perc_interactive.html'))
    print(f"Interactive plot saved to: {events_path.with_suffix(f'.{PERCENTILE}_perc_interactive.html')}")

    return aggregated


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-03_01-04-03-001589' / 'events.jsonl'
    agg = plot_interactive(path)
