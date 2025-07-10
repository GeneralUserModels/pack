import json
from pathlib import Path
from datetime import datetime
import numpy as np

from observers.logs import AggregatedLog


def extract_key(details: dict, pretty=False) -> str:
    key = details.get('key', '')
    if not key:
        return
    key = key.replace('Key.', '')
    return key


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


def _aggregate_event(events_slice: list) -> AggregatedLog:
    cursor_pos = [ev.get("details", {}).get('position', 0) for ev in events_slice if ev.get("details", {}).get('position', 0)]
    return AggregatedLog(
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


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-03_01-04-03-001589' / 'events.jsonl'
    agg = plot_interactive(path)
