import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def get_events(session="session_2025-07-04_09-31-33-777536"):
    events_path = Path(__file__).parent.parent / "logs" / session / "events.jsonl"

    events = []
    with open(events_path, 'r') as f:
        for line in f:
            events.append(json.loads(line))
    return events


def analyze_breaks_by_event_type(events, event_types=None, percentile=90):
    grouped_events = defaultdict(list)
    for event in events:
        if event_types and event['event'] not in event_types:
            continue
        grouped_events[event['event']].append(event)

    num_types = len(grouped_events)

    fig, axes = plt.subplots(num_types, 1, figsize=(10, 4 * num_types), squeeze=False)
    fig.suptitle(f'Break Duration Histograms by Event Type (Top {100 - percentile}% in Red)', fontsize=12)

    for idx, (etype, evs) in enumerate(sorted(grouped_events.items())):
        evs.sort(key=lambda x: x['timestamp'])

        breaks = []
        last_ts = datetime.strptime(evs[0]['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
        for event in evs[1:]:
            ts = datetime.strptime(event['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
            delta = (ts - last_ts).total_seconds()
            breaks.append(delta)
            last_ts = ts

        breaks = np.array(breaks)
        perc_thresh = np.percentile(breaks, percentile)

        print(f"{etype}: {len([b for b in breaks if b < 0.1])} very short breaks (<0.1s)")
        print(f"{etype}: {percentile}th percentile = {perc_thresh:.2f}s")

        counts, bin_edges = np.histogram(breaks, bins=500)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        colors = ['blue' if x <= perc_thresh else 'red' for x in bin_centers]

        ax = axes[idx][0]
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), align='center', color=colors)
        ax.set_title(f'{etype} - Break Duration Histogram')
        ax.set_xlabel('Break Duration (seconds)')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    plt.subplots_adjust(top=0.92, hspace=0.6)
    plt.show()


def analyze_breaks(events, event_types=None):
    breaks = []

    events.sort(key=lambda x: x['timestamp'])
    last_ts = datetime.strptime(events[0]['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")

    for event in events[1:]:
        if event_types and event['event'] not in event_types:
            continue
        ts = datetime.strptime(event['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
        breaks.append(timedelta(seconds=(ts - last_ts).total_seconds()).total_seconds())
        last_ts = ts

    print(len([b for b in breaks if b < 0.1]))
    plt.figure(figsize=(10, 6))
    plt.hist(breaks, bins=500, edgecolor='black')
    plt.title('Break Duration Histogram')
    plt.xlabel('Break Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


if __name__ == "__main__":
    events = get_events()
    event_types = [
        "poll",
        "mouse_move",
        "mouse_down",
        "mouse_up",
        "mouse_scroll",
        "keyboard_press",
        "keyboard_release"
    ]
    analyze_breaks(events, event_types)  # , ["poll", "keyboard_release", "mouse_move"])
    analyze_breaks_by_event_type(events, event_types)  # , ["poll", "keyboard_release", "mouse_move"])
