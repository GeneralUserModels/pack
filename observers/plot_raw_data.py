import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

from observers.logs import EventLog

PERCENTILE = 95

EVENT_Y = {
    "mouse_move": 1,
    "mouse_down": 2,
    "mouse_scroll": 4,
    "keyboard_press": 5,
}

SPECIAL_KEYS = {
    'enter': '⏎', 'return': '⏎', 'backspace': '⌫', 'space': '␣',
    'tab': '⇥', 'shift': '⇧', 'ctrl': '⌃', 'alt': '⎇',
    'cmd': '⌘', 'esc': '⎋', 'escape': '⎋',
    'up': '↑', 'down': '↓', 'left': '←', 'right': '→',
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

    return timestamps, y_positions, labels, breaks, durations, thresholds


def _aggregate_event(events_slice: list) -> EventLog:
    return EventLog(
        start_timestamp=events_slice[0]['timestamp'],
        end_timestamp=events_slice[-1]['timestamp'],
        monitor=events_slice[0]['monitor'],
        start_screenshot_path=events_slice[0].get('screenshot_path'),
        end_screenshot_path=events_slice[-1].get('screenshot_path'),
        start_cursor_pos=events_slice[0].get('cursor_pos'),
        end_cursor_pos=events_slice[-1].get('cursor_pos'),
        click_positions=[ev.get('click_position') for ev in events_slice if 'click_position' in ev],
        scroll_directions=list({ev.get('scroll_direction') for ev in events_slice if 'scroll_direction' in ev}),
        keys_pressed=[extract_key(ev['details'], pretty=True) for ev in events_slice if ev['event'] == 'keyboard_press' and extract_key(ev['details'], pretty=True)]
    )


def aggregate_logs(events: list, breaks: dict):
    events_sorted = sorted(events, key=lambda e: e['timestamp'])
    edges = sorted(set([b.timestamp() for spans in breaks.values() for bstart, bend in spans for b in (bstart, bend)]))
    logs = []
    start_idx = 0
    for edge in edges:
        for idx in range(start_idx, len(events_sorted)):
            ev_time = datetime.strptime(events_sorted[idx]['timestamp'], "%Y-%m-%d_%H-%M-%S-%f").timestamp()
            if ev_time == edge:
                logs.append(_aggregate_event(events_sorted[start_idx:idx + 1]))
                start_idx = idx + 1
                break
    if start_idx < len(events_sorted):
        logs.append(_aggregate_event(events_sorted[start_idx:]))
    return logs


def plot(events_path: Path):
    events = load_events(events_path, EVENT_Y)
    timestamps, y_positions, labels, breaks, durations, thresholds = calculate_breaks(events, EVENT_Y, PERCENTILE)
    aggregated = aggregate_logs(events, breaks)

    date_nums = mdates.date2num(timestamps)

    cmap = plt.get_cmap('tab10', len(EVENT_Y))
    event_colors = {evt: cmap(i) for i, evt in enumerate(EVENT_Y)}

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(12, 12), sharex=False,
        gridspec_kw={'height_ratios': [2, 1, 1, 1]}
    )

    for ts, y, lbl in zip(timestamps, y_positions, labels):
        ax1.scatter(ts, y, color=event_colors[list(EVENT_Y.keys())[list(EVENT_Y.values()).index(y)]], s=10)
        if lbl:
            ax1.text(ts, y + 0.1, lbl, fontsize=8, ha='center')
    for evt, spans in breaks.items():
        for s, e in spans:
            ax1.axvspan(s, e, color=event_colors[evt], alpha=0.2)
    ax1.set_yticks(list(EVENT_Y.values()))
    ax1.set_yticklabels(list(EVENT_Y.keys()))
    ax1.set_title('Event Timeline with Break Spans')
    ax1.grid(True)
    ax1.legend(handles=[mpatches.Patch(color=event_colors[e], label=e) for e in EVENT_Y],
               title='Event Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    bins = 500
    edges = np.linspace(date_nums.min(), date_nums.max(), bins)
    ax2.hist(date_nums, bins=edges, color='gray')
    for evt, spans in breaks.items():
        for s, e in spans:
            ax2.axvspan(mdates.date2num(s), mdates.date2num(e), color=event_colors[evt], alpha=0.2)
    ax2.set_title('Log Count Histogram with Break Spans')
    ax2.set_ylabel('Count')
    ax2.grid(True)

    all_dur = [durations[evt] for evt in EVENT_Y]
    labs = list(EVENT_Y.keys())
    ax3.hist(all_dur, bins=50, stacked=True, color=[event_colors[e] for e in EVENT_Y], label=labs)
    for evt, th in thresholds.items():
        ax3.axvline(th, linestyle='--', color=event_colors[evt], label=f'{evt} {PERCENTILE}th pct')
    ax3.set_title('Stacked Histogram of Break Durations')
    ax3.set_xlabel('Seconds')
    ax3.set_ylabel('Freq')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    y_locs = np.arange(len(aggregated)) + 0.5
    for idx, log in enumerate(aggregated):
        start = datetime.strptime(log.start_timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        end = datetime.strptime(log.end_timestamp, "%Y-%m-%d_%H-%M-%S-%f")
        ax4.broken_barh([(mdates.date2num(start), mdates.date2num(end) - mdates.date2num(start))],
                        (y_locs[idx] - 0.4, 0.8), facecolors='tab:blue')
        keys = ''.join(log.keys_pressed)
        ax4.text(mdates.date2num(start), y_locs[idx], keys, va='center', ha='left', fontsize=8)
    ax4.set_ylim(0, len(aggregated))
    ax4.set_yticks(y_locs)
    ax4.set_yticklabels([f'Log {i + 1}' for i in range(len(aggregated))])
    ax4.set_title('Aggregated EventLog Spans with Keys')
    ax4.set_xlabel('Time')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.grid(True)

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    path = Path(__file__).parent.parent / 'logs' / 'session_2025-07-04_15-31-18-439993' / 'events.jsonl'
    plot(path)
