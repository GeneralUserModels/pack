import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

PERCENTILE = 95
events_path = Path(__file__).parent.parent / "logs" / "session_2025-07-04_13-28-45-580384" / "events.jsonl"

event_y = {
    "poll": 0,
    "mouse_move": 1,
    "mouse_down": 2,
    "mouse_up": 3,
    "mouse_scroll": 4,
    "keyboard_press": 5,
    "keyboard_release": 6
}

break_filter_events = [
    "keyboard_press"
]

special_keys = {
    'enter': '⏎',
    'return': '⏎',
    'backspace': '⌫',
    'space': '␣',
    'tab': '⇥',
    'shift': '⇧',
    'ctrl': '⌃',
    'alt': '⎇',
    'cmd': '⌘',
    'esc': '⎋',
    'escape': '⎋',
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→',
}

# Load events
events = []
with open(events_path, 'r') as f:
    for line in f:
        line = json.loads(line)
        if line.get("event") in event_y:
            events.append(line)

timestamps = []
break_timestamps = []
y_positions = []
labels = []
for event in events:
    ts = datetime.strptime(event['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
    evt = event['event']
    timestamps.append(ts)
    if evt in break_filter_events:
        break_timestamps.append(ts)
    y_positions.append(event_y[evt])

    if evt == 'keyboard_press':
        key = event['details'].get('key', '').replace('Key.', '')
        labels.append(special_keys.get(key, key))
    else:
        labels.append('')

deltas = np.diff([t.timestamp() for t in break_timestamps])
threshold = np.percentile(deltas, PERCENTILE)
break_indices = np.where(deltas >= threshold)[0]
break_spans = [(break_timestamps[i], break_timestamps[i + 1]) for i in break_indices]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1]})

ax1.scatter(timestamps, y_positions, c='blue')
for ts, y, label in zip(timestamps, y_positions, labels):
    if y == event_y['keyboard_press'] and label:
        ax1.text(ts, y + 0.1, label, fontsize=9, ha='center')

for start, end in break_spans:
    ax1.axvline(start, color='grey', linestyle='--', linewidth=0.8)
    ax1.axvline(end, color='grey', linestyle='--', linewidth=0.8)
    ax1.axvspan(start, end, color='grey', alpha=0.3)

ax1.set_yticks(list(event_y.values()))
ax1.set_yticklabels(list(event_y.keys()))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
ax1.set_title(f'Event Timeline with Breaks (top {PERCENTILE}percentile)')
ax1.set_ylabel('Event Type')
ax1.grid(True)

ax2.hist(timestamps, bins=500)
ax2.set_title('Log Count Histogram (500 bins)')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Count')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax2.grid(True)

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()
