import json
from pathlib import Path
from datetime import datetime

import numpy as np

from modules import AggregatedLog, RawLogEvents

PERCENTILE = 95

MAX_RELEASE_EVENT_DURATION = 1.0  # seconds, used to filter out noise in keyboard events
MOUSE_DOWN_DEBOUNCE = 500  # milliseconds
MOUSE_DOWN_POS_OFFSET = 0.01  # % of screen height, width


def extract_key(details: dict, pretty=False) -> str:
    key = details.get('key', '')
    if not key:
        return
    key = key.replace('Key.', '')
    return key


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
        if evt in ["mouse_down", "poll"]:
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
    return timestamps, breaks, durations, thresholds


def debounce_double_clicks(events):
    logs = RawLogEvents()
    processed_events = []
    for i, event in enumerate(events):
        if event.event_type != "mouse_down" or "left" not in event.details.get('button', '').lower():
            processed_events.append(event)
            continue
        ts = datetime.strptime(event.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        pos = event.cursor_pos
        monitor_dim = [event.monitor.get("width", 1920), event.monitor.get("height", 1080)]

        for new_event in events[i + 1:]:
            if new_event.event_type != "mouse_down":
                continue
            new_ts = datetime.strptime(new_event.timestamp, "%Y-%m-%d_%H-%M-%S-%f")
            if (new_ts - ts).total_seconds() * 1000 > MOUSE_DOWN_DEBOUNCE:
                processed_events.append(event)
                break
            new_pos = new_event.cursor_pos

            if abs(new_pos[0] - pos[0]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[0] and \
               abs(new_pos[1] - pos[1]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[1]:
                new_event.details['double_click'] = True
                processed_events.append(new_event)

                for j in range(i + 1, len(events)):
                    if events[j].event_type == "mouse_up" and events[j].details.get('button', '').lower() == 'left':
                        del events[j]
                break
            else:
                processed_events.append(event)
                break
    logs.events = processed_events
    return logs


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


def _debounce_press_event(events, debounce_event, max_duration, offset):
    def get_event_detail(event):
        if event.event_type in ['mouse_down', 'mouse_up']:
            return event.details.get('button')
        elif event.event_type in ['keyboard_press', 'keyboard_release']:
            return extract_key(event.details)

    cut_idx = None
    for lookahead in events:
        switched_screen = json.dumps(debounce_event.monitor) != json.dumps(lookahead.monitor)
        if switched_screen or (lookahead.strp_timestamp - debounce_event.strp_timestamp) > max_duration or lookahead.event_type in ["mouse_down", "keyboard_press"]:
            break

        if debounce_event.event_type == 'mouse_down':
            if (lookahead.event_type == 'mouse_up' and
                    get_event_detail(lookahead) == get_event_detail(debounce_event)):

                pos = debounce_event.cursor_pos
                new_pos = lookahead.cursor_pos
                monitor_dim = [debounce_event.monitor.get("width", 1920),
                               debounce_event.monitor.get("height", 1080)]

                if (abs(new_pos[0] - pos[0]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[0] and
                        abs(new_pos[1] - pos[1]) <= MOUSE_DOWN_POS_OFFSET * monitor_dim[1]):
                    cut_idx = events.index(lookahead) + offset
                    break

        elif debounce_event.event_type == 'keyboard_press':
            if (lookahead.event_type == 'keyboard_release' and
                    get_event_detail(lookahead) == get_event_detail(debounce_event)):
                cut_idx = events.index(lookahead) + offset
                break

    return cut_idx


def aggregate_logs(events: RawLogEvents, breaks: dict):
    events.sort()
    edges = sort_and_fill_edges(breaks)
    events = debounce_double_clicks(events)

    logs = []
    start_idx = 0
    cut_idx = 0

    for edge in edges:
        if edge < events[start_idx].strp_timestamp:
            continue
        for rel_idx, event in enumerate(events[start_idx:], start=start_idx):
            ev_time = event.strp_timestamp
            next_ev_time = events[rel_idx + 1].strp_timestamp if rel_idx + 1 < len(events) else float('inf')
            switched_screen = json.dumps(event.monitor) != json.dumps(events[start_idx].monitor)
            if ev_time <= edge < next_ev_time:
                cut_idx = rel_idx
                break
            if switched_screen:
                cut_idx = max(start_idx, rel_idx - 1)
                break
            if event.event_type == 'mouse_down':
                cut_idx = rel_idx
                break
        else:
            cut_idx = len(events) - 1
            switched_screen = False

        last_event = events[cut_idx]
        if last_event.event_type in ['mouse_down', 'keyboard_press']:
            _cut_idx = _debounce_press_event(events[cut_idx + 1:], last_event, MAX_RELEASE_EVENT_DURATION, cut_idx + 1)
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
    timestamps, breaks, durations, thresholds = calculate_breaks(logs, percentile)
    return aggregate_logs(debounced_logs, breaks)


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-13_15-59-04-565176'

    aggregated_logs = main(path, PERCENTILE)
    with open(path.parent / f'aggregated_logs_{PERCENTILE}.json', 'w') as f:
        json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)
