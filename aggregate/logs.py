import json
from pathlib import Path
from datetime import datetime

import numpy as np

from modules import AggregatedLog, RawLogEvents

PERCENTILE = 95

MAX_RELEASE_EVENT_DURATION = 1  # seconds, used to filter out noise in keyboard events
MOUSE_DOWN_DEBOUNCE = 500  # milliseconds
MOUSE_DOWN_POS_OFFSET = 0.01  # % of screen height, width


def extract_key(details: dict, pretty=False) -> str:
    key = details.get('key', '')
    if not key:
        return
    key = key.replace('Key.', '')
    return key


# TODO: Refactor
def calculate_breaks(events: list, percentile: int):
    timestamps = []
    per_event_ts = {}

    for ev in events:
        ts = datetime.strptime(ev.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        evt = ev.event_type
        timestamps.append(ts)
        per_event_ts.setdefault(evt, []).append(ts)

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

    mouse_down_events = [e for e in events if e.event_type == "mouse_down"]

    processed_mouse_events = []
    last_pos, last_time = None, None

    def remove_lst_double_click(ts, pos, events):
        for i in range(len(events) - 1, -1, -1):
            evt_ts = datetime.strptime(events[i].timestamp, "%Y-%m-%d_%H-%M-%S-%f")
            if evt_ts == ts and events[i].cursor_pos == pos:
                del events[i]
        return events

    for evt in mouse_down_events:
        ts = datetime.strptime(evt.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        button = evt.details.get('button', 'unknown')

        if "left" not in button.lower():
            processed_mouse_events.append(evt)
            continue

        pos = evt.cursor_pos
        monitor_dim = [evt.monitor.get("width", 1920), evt.monitor.get("height", 1080)]

        new_evt = evt.copy()

        if last_pos is None or last_time is None:
            last_pos, last_time = pos, ts
            processed_mouse_events.append(new_evt)
            continue

        time_diff = (ts - last_time).total_seconds() * 1000

        if (time_diff <= MOUSE_DOWN_DEBOUNCE and not None in last_pos + pos and
            abs(pos[0] - last_pos[0]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[0] and
                abs(pos[1] - last_pos[1]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[1]):

            new_evt.details['double_click'] = True
            processed_mouse_events.append(new_evt)
            processed_mouse_events = remove_lst_double_click(last_time, last_pos, processed_mouse_events)
        else:
            last_pos, last_time = pos, ts
            processed_mouse_events.append(new_evt)

    mouse_down_timestamps = []
    for evt in processed_mouse_events:
        ts = datetime.strptime(evt.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        mouse_down_timestamps.append(ts)

    mouse_down_breaks = []
    if len(mouse_down_timestamps) > 1:
        for i in range(len(mouse_down_timestamps) - 1):
            mouse_down_breaks.append((mouse_down_timestamps[i], mouse_down_timestamps[i + 1]))

    breaks["mouse_down"] = mouse_down_breaks
    durations["mouse_down"] = []
    thresholds["mouse_down"] = 0

    return timestamps, breaks, durations, thresholds


def sort_and_fill_edges(breaks, gap_threshold=60):
    edges = sorted({
        ts.timestamp()
        for spans in breaks.values()
        for start, end in spans
        for ts in (start, end)
    })
    result = []

    for i in range(len(edges) - 1):
        result.append(edges[i])

        gap = edges[i + 1] - edges[i]
        if gap > gap_threshold:
            current = edges[i] + gap_threshold
            while current < edges[i + 1]:
                result.append(current)
                current += gap_threshold

    result.append(edges[-1])
    return result


def _debounce_press_event(events, debounce_event, max_duration):
    def get_event_detail(event):
        if event.event_type == 'mouse_down':
            return event.details.get('button')
        elif event.event_type == 'keyboard_press':
            return extract_key(event.details)

    cut_idx = None
    for lookahead in events:
        switched_screen = json.dumps(debounce_event.monitor) != json.dumps(lookahead.monitor)
        if switched_screen or max_duration > lookahead.strp_timestamp - debounce_event.strp_timestamp:
            break
        if lookahead.event_type == debounce_event.event_type and get_event_detail(lookahead) == get_event_detail(debounce_event):
            cut_idx = events.index(lookahead)
            break
    return cut_idx


def aggregate_logs(events: RawLogEvents, breaks: dict):
    events.sort()
    edges = sort_and_fill_edges(breaks)

    logs = []
    start_idx = 0

    for edge in edges:
        for rel_idx, event in enumerate(events[start_idx:], start=start_idx):
            ev_time = event.strp_timestamp
            next_ev_time = events[rel_idx + 1].strp_timestamp if rel_idx + 1 < len(events) else float('inf')
            switched_screen = json.dumps(event.monitor) != json.dumps(events[start_idx].monitor)
            if ev_time <= edge and next_ev_time > edge:
                cut_idx = rel_idx
                break
            if switched_screen:
                cut_idx = max(start_idx, rel_idx - 1)
                break
        else:
            cut_idx = len(events) - 1
            switched_screen = False

        last_event = events[cut_idx]
        if last_event.event_type not in ['mouse_down', 'keyboard_press']:
            _cut_idx = _debounce_press_event(events, last_event, MAX_RELEASE_EVENT_DURATION)
            cut_idx = _cut_idx if _cut_idx is not None else cut_idx

        logs.append(AggregatedLog.from_raw_log_events(events[start_idx:cut_idx + 1]))
        start_idx = cut_idx + 1

        if start_idx >= len(events):
            break

    if start_idx < len(events):
        logs.append(AggregatedLog.from_raw_log_events(events[start_idx:]))

    return logs


def main(path, percentile=95):
    logs = RawLogEvents().load(path / 'events.jsonl')
    logs.sort()
    debounced_logs = RawLogEvents()
    debounced_logs.events = [log for log in logs if log.screenshot_path]
    logs.events = [log for log in logs if log.event_type != 'poll']
    timestamps, breaks, durations, thresholds = calculate_breaks(logs, percentile)
    return aggregate_logs(debounced_logs, breaks)


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-13_15-59-04-565176'

    aggregated_logs = main(path, PERCENTILE)
    with open(path.parent / f'aggregated_logs_{PERCENTILE}.json', 'w') as f:
        json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)
