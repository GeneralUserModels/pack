import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

events_path = Path(__file__).parent.parent / "logs"

event_y = {
    "poll": 0,
    "mouse_move": 1,
    "mouse_down": 2,
    "mouse_up": 3,
    "mouse_scroll": 4,
    "keyboard_press": 5,
    "keyboard_release": 6
}


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

events = []
with open(events_path, 'r') as f:
    for line in f:
        events.append(json.loads(line))

timestamps = []
y_positions = []
labels = []

for event in events:
    ts = datetime.strptime(event['timestamp'], "%Y-%m-%d_%H-%M-%S-%f")
    event_type = event['event']
    y_val = event_y[event_type]

    timestamps.append(ts)
    y_positions.append(y_val)

    if event_type == "keyboard_press":
        key = event['details'].get('key', '')
        if key:
            key = key.replace("Key.", "")
        key = special_keys.get(key, key)
        labels.append(key)
    else:
        labels.append('')

plt.figure(figsize=(12, 6))
plt.scatter(timestamps, y_positions, c='blue')

for ts, y, label in zip(timestamps, y_positions, labels):
    if y == event_y['keyboard_press'] and label:
        plt.text(ts, y + 0.1, label, fontsize=9, ha='center')

plt.yticks(list(event_y.values()), list(event_y.keys()))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
plt.gcf().autofmt_xdate()
plt.title("Event Timeline")
plt.xlabel("Timestamp")
plt.ylabel("Event Type")
plt.grid(True)
plt.tight_layout()

plt.show()
