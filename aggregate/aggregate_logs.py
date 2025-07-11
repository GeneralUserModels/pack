import json
from pathlib import Path
from datetime import datetime
import numpy as np

from modules import AggregatedLog, RawLogEvents

PERCENTILE = 95

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

    events_dict = {(e.timestamp, e.event_type): e for e in events}
    for proc_evt in processed_mouse_events:
        key = (proc_evt.timestamp, proc_evt.event_type)
        if key in events_dict:
            events_dict[key].details = proc_evt.details

    return timestamps, breaks, durations, thresholds


def aggregate_logs(events: RawLogEvents, breaks: dict):
    events.sort()
    edges = sorted({
        ts.timestamp()
        for spans in breaks.values()
        for start, end in spans
        for ts in (start, end)
    })

    logs = []
    start_idx = 0

    for edge in edges:
        for rel_idx, event in enumerate(events[start_idx:], start=start_idx):
            ev_time = datetime.strptime(event.timestamp, "%Y-%m-%d_%H-%M-%S-%f").timestamp()
            switched_screen = json.dumps(event.monitor) != json.dumps(events[start_idx].monitor)
            if ev_time == edge or switched_screen:
                cut_idx = rel_idx
                break
        else:
            cut_idx = len(events) - 1
            switched_screen = False

        last_event = events[cut_idx]
        evt_type = last_event.event_type

        if evt_type == 'mouse_down' and not switched_screen:
            button = last_event.details.get('button')
            for lookahead in events[cut_idx + 1:]:
                # TODO: Threshold
                if lookahead.event_type == 'mouse_up' and lookahead.details.get('button') == button:
                    cut_idx = events.index(lookahead)
                    break

        elif evt_type == 'keyboard_press' and not switched_screen:
            key = extract_key(last_event.details)
            for lookahead in events[cut_idx + 1:]:
                # TODO: Threshold
                if lookahead.event_type == 'keyboard_release' and extract_key(lookahead.details) == key:
                    cut_idx = events.index(lookahead)
                    break

        logs.append(AggregatedLog.from_raw_log_events(events[start_idx:cut_idx + 1]))
        start_idx = cut_idx + 1

        if start_idx >= len(events):
            break

    if start_idx < len(events):
        logs.append(AggregatedLog.from_raw_log_events(events[start_idx:]))

    return logs


def main(path, percentile=95):
    logs = RawLogEvents().load(path)
    logs.sort()
    timestamps, breaks, durations, thresholds = calculate_breaks(logs, percentile)
    return aggregate_logs(logs, breaks)


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-11_02-51-30-768112' / 'events.jsonl'
    aggregated_logs = main(path, PERCENTILE)
    with open(path.parent / f'aggregated_logs_{PERCENTILE}.json', 'w') as f:
        json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)
