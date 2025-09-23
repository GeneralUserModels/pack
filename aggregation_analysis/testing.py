import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Dict, Any

FNAME = "./aggregation_analysis/regression_dataset.json"
CLICK_PAIR_MAX_GAP = 1.0
BURST_MAX_GAP_MOVE = 0.5
BURST_MAX_GAP_SCROLL = 0.7
BURST_MAX_GAP_KEYBOARD = 0.3
VERBOSE = True


def load_data(fname=FNAME):
    if os.path.exists(fname):
        with open(fname, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {fname}")
        return data
    else:
        print(f"File not found: {fname}")
        return []


def cluster_event_times(times: List[Dict], max_gap: float) -> List[Tuple[float, float, List[Dict]]]:
    """Simple clustering based only on time gaps, ignoring SSIM"""
    if not times:
        return []
    times = sorted(times, key=lambda e: e["time"])
    clusters = []
    cur_start = times[0]["time"]
    cur_end = times[0]["time"]
    cur_list = [times[0]]

    for t in times[1:]:
        if t["time"] - cur_end <= max_gap:
            cur_end = t["time"]
            cur_list.append(t)
        else:
            clusters.append((cur_start, cur_end, cur_list))
            cur_start = t["time"]
            cur_end = t["time"]
            cur_list = [t]
    clusters.append((cur_start, cur_end, cur_list))
    return clusters


def find_screenshot_before(target_time: float, screenshots: List[Dict]) -> Dict:
    """Find the last screenshot with timestamp < target_time"""
    for i in range(len(screenshots) - 1, -1, -1):
        if screenshots[i]["timestamp"] < target_time:
            return screenshots[i]
    return screenshots[0] if screenshots else None


def find_screenshot_after(target_time: float, screenshots: List[Dict]) -> Dict:
    """Find the first screenshot with timestamp > target_time"""
    for screenshot in screenshots:
        if screenshot["timestamp"] > target_time:
            return screenshot
    return screenshots[-1] if screenshots else None


# Load and process data
data = load_data()

# Build screenshot index list
screenshots = []
for i, item in enumerate(data):
    inp = item.get("input", {})
    screenshots.append({
        "index": i,
        "timestamp": float(inp.get("timestamp", 0.0)),
        "ssim": float(inp.get("ssim_similarity", 1.0)),
        "monitor_id": inp.get("monitor_id"),
        "events": inp.get("events", []),
        "label": bool(item.get("output", False))
    })
screenshots = sorted(screenshots, key=lambda s: s["timestamp"])

# Build events list
events: List[Dict[str, Any]] = []
for s in screenshots:
    ts = s["timestamp"]
    for ev in s["events"]:
        rel = float(ev.get("relative_time", 0.0)) if ev.get("relative_time") is not None else 0.0
        abs_t = ts + rel
        events.append({
            "time": abs_t,
            "event_type": ev.get("event_type"),
            "details": ev.get("details", {}),
            "cursor_pos": ev.get("cursor_pos"),
            "monitor_id": ev.get("monitor_id", s["monitor_id"])
        })
events = sorted(events, key=lambda e: e["time"])

move_events = [e for e in events if e["event_type"] == "mouse_move"]
scroll_events = [e for e in events if e["event_type"] == "mouse_scroll"]
kbd_events = [e for e in events if e["event_type"] in ("keyboard_press", "keyboard_release")]
click_downs = [e for e in events if e["event_type"] == "mouse_down"]
click_ups = [e for e in events if e["event_type"] == "mouse_up"]

move_clusters = cluster_event_times(move_events, BURST_MAX_GAP_MOVE)
scroll_clusters = cluster_event_times(scroll_events, BURST_MAX_GAP_SCROLL)
kbd_clusters = cluster_event_times(kbd_events, BURST_MAX_GAP_KEYBOARD)

click_pairs = []
for d in click_downs:
    candidate_ups = [u for u in click_ups if u["time"] >= d["time"] and u.get("monitor_id") == d.get("monitor_id") and u["time"] - d["time"] <= CLICK_PAIR_MAX_GAP]
    if candidate_ups:
        u = candidate_ups[0]
        click_pairs.append((d["time"], u["time"]))

# Create actions with simple screenshot selection
actions = []

# Add click actions
for start_time, end_time in click_pairs:
    start_screenshot = find_screenshot_before(start_time, screenshots)
    end_screenshot = find_screenshot_after(end_time, screenshots)
    actions.append({
        "type": "click",
        "start": start_time,
        "end": end_time,
        "start_screenshot": start_screenshot,
        "end_screenshot": end_screenshot,
        "start_screenshot_time": start_screenshot["timestamp"] if start_screenshot else None,
        "end_screenshot_time": end_screenshot["timestamp"] if end_screenshot else None
    })

# Add move actions
for start_time, end_time, event_list in move_clusters:
    start_screenshot = find_screenshot_before(start_time, screenshots)
    end_screenshot = find_screenshot_after(end_time, screenshots)
    actions.append({
        "type": "move",
        "start": start_time,
        "end": end_time,
        "start_screenshot": start_screenshot,
        "end_screenshot": end_screenshot,
        "start_screenshot_time": start_screenshot["timestamp"] if start_screenshot else None,
        "end_screenshot_time": end_screenshot["timestamp"] if end_screenshot else None,
        "event_count": len(event_list)
    })

# Add scroll actions
for start_time, end_time, event_list in scroll_clusters:
    start_screenshot = find_screenshot_before(start_time, screenshots)
    end_screenshot = find_screenshot_after(end_time, screenshots)
    actions.append({
        "type": "scroll",
        "start": start_time,
        "end": end_time,
        "start_screenshot": start_screenshot,
        "end_screenshot": end_screenshot,
        "start_screenshot_time": start_screenshot["timestamp"] if start_screenshot else None,
        "end_screenshot_time": end_screenshot["timestamp"] if end_screenshot else None,
        "event_count": len(event_list)
    })

# Add keyboard actions
for start_time, end_time, event_list in kbd_clusters:
    start_screenshot = find_screenshot_before(start_time, screenshots)
    end_screenshot = find_screenshot_after(end_time, screenshots)
    actions.append({
        "type": "keyboard",
        "start": start_time,
        "end": end_time,
        "start_screenshot": start_screenshot,
        "end_screenshot": end_screenshot,
        "start_screenshot_time": start_screenshot["timestamp"] if start_screenshot else None,
        "end_screenshot_time": end_screenshot["timestamp"] if end_screenshot else None,
        "event_count": len(event_list)
    })

actions = sorted(actions, key=lambda a: a["start"])


def create_timeline_plot():
    if not screenshots:
        print("No screenshots to plot")
        return

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Simplified Action Clustering Timeline', fontsize=14, fontweight='bold')

    # Get time range
    min_time = min(s["timestamp"] for s in screenshots)
    max_time = max(s["timestamp"] for s in screenshots)
    time_range = max_time - min_time

    # 1. Screenshots timeline
    ax1 = axes[0]
    screenshot_times = [s["timestamp"] for s in screenshots]
    screenshot_labels = [s["label"] for s in screenshots]

    # Plot screenshots as vertical lines, colored by label
    print(f"Plotting {len(screenshot_times)} {len(screenshot_labels)} screenshots")
    for i, (t, label) in enumerate(zip(screenshot_times, screenshot_labels)):
        color = 'red' if label else 'blue'
        ax1.axvline(x=t, color=color, alpha=0.6, linewidth=1)

    ax1.set_ylabel('Screenshots')
    ax1.set_title('Screenshots (Red=Positive Label, Blue=Negative)')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])

    # 2. SSIM values (for reference, but not used in clustering)
    ax2 = axes[1]
    ssim_times = [s["timestamp"] for s in screenshots]
    ssim_values = [s["ssim"] for s in screenshots]

    ax2.plot(ssim_times, ssim_values, 'o-', markersize=3, linewidth=1, color='green', alpha=0.7)
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Similarity Values (Not Used in Clustering)')
    ax2.grid(True, alpha=0.3)

    # 3. Individual events
    ax3 = axes[2]
    event_types = list(set(e["event_type"] for e in events))
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    event_colors = dict(zip(event_types, colors))

    for i, event_type in enumerate(event_types):
        event_times = [e["time"] for e in events if e["event_type"] == event_type]
        y_pos = [i] * len(event_times)
        ax3.scatter(event_times, y_pos, c=[event_colors[event_type]], s=10, alpha=0.7, label=event_type)

    ax3.set_ylabel('Event Types')
    ax3.set_title('Individual Events')
    ax3.set_yticks(range(len(event_types)))
    ax3.set_yticklabels(event_types, fontsize=8)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 4. Click clusters
    ax4 = axes[3]
    for i, (start, end) in enumerate(click_pairs):
        rect = patches.Rectangle((start, -0.4), end - start, 0.8,
                                 linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
        ax4.add_patch(rect)

    ax4.set_ylabel('Click Actions')
    ax4.set_title('Click Pairs (Mouse Down → Mouse Up)')
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_yticks([])

    # 5. Non-click clusters
    ax5 = axes[4]
    cluster_types = [
        ('Mouse Move', move_clusters, 'blue'),
        ('Scroll', scroll_clusters, 'green'),
        ('Keyboard', kbd_clusters, 'orange')
    ]

    y_offset = 0
    for name, clusters, color in cluster_types:
        for start, end, _ in clusters:
            rect = patches.Rectangle((start, y_offset - 0.3), end - start, 0.6,
                                     linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
            ax5.add_patch(rect)
        y_offset += 1

    ax5.set_ylabel('Non-Click Clusters')
    ax5.set_title('Event Clusters (Blue=Move, Green=Scroll, Orange=Keyboard)')
    ax5.set_ylim(-0.5, 2.5)
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Move', 'Scroll', 'Keyboard'])

    # 6. Actions with screenshot boundaries
    ax6 = axes[5]

    # Show all action clusters
    for action in actions:
        color_map = {'click': 'red', 'move': 'blue', 'scroll': 'green', 'keyboard': 'orange'}
        color = color_map.get(action['type'], 'gray')

        # Draw action interval
        rect = patches.Rectangle((action['start'], -0.3), action['end'] - action['start'], 0.6,
                                 linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
        ax6.add_patch(rect)

        # Mark screenshot boundaries
        if action['start_screenshot_time'] is not None:
            ax6.axvline(x=action['start_screenshot_time'], color=color, alpha=0.8, linewidth=2, linestyle='--')
        if action['end_screenshot_time'] is not None:
            ax6.axvline(x=action['end_screenshot_time'], color=color, alpha=0.8, linewidth=2, linestyle='-.')

    ax6.set_ylabel('Actions + Screenshots')
    ax6.set_title('Actions with Screenshot Boundaries (Dashed=Start, Dash-dot=End)')
    ax6.set_ylim(-0.5, 0.5)
    ax6.set_yticks([])
    ax6.set_xlabel('Time (seconds)')

    # Add gap threshold annotations
    textstr = f'Gap Thresholds (Time Only):\nMove: {BURST_MAX_GAP_MOVE}s\nScroll: {BURST_MAX_GAP_SCROLL}s\nKeyboard: {BURST_MAX_GAP_KEYBOARD}s\nClick Pair: {CLICK_PAIR_MAX_GAP}s'
    ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    os.makedirs("./aggregation_analysis", exist_ok=True)
    plt.savefig("./aggregation_analysis/simplified_timeline.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\nTimeline Analysis Summary:")
    print(f"Total screenshots: {len(screenshots)}")
    print(f"Total events: {len(events)}")
    print(f"Total actions: {len(actions)}")
    print(f"  - Click actions: {len([a for a in actions if a['type'] == 'click'])}")
    print(f"  - Move actions: {len([a for a in actions if a['type'] == 'move'])}")
    print(f"  - Scroll actions: {len([a for a in actions if a['type'] == 'scroll'])}")
    print(f"  - Keyboard actions: {len([a for a in actions if a['type'] == 'keyboard'])}")
    print(f"Time range: {min_time:.3f}s to {max_time:.3f}s ({time_range:.3f}s duration)")


# Create the visualization
create_timeline_plot()

# Print action details
print("\nAction Details:")
for i, action in enumerate(actions):
    print(f"{i + 1}. {action['type'].upper()}: {action['start']:.3f}s → {action['end']:.3f}s (dur: {action['end'] - action['start']:.3f}s)")
    print(f"   Start screenshot: {action['start_screenshot_time']:.3f}s (gap: {action['start'] - action['start_screenshot_time']:.3f}s)")
    print(f"   End screenshot: {action['end_screenshot_time']:.3f}s (gap: {action['end_screenshot_time'] - action['end']:.3f}s)")
    if 'event_count' in action:
        print(f"   Events in cluster: {action['event_count']}")
    print()

# Save actions to file
os.makedirs("./aggregation_analysis", exist_ok=True)
with open("./aggregation_analysis/simplified_actions.json", "w") as f:
    json.dump(actions, f, indent=2, default=str)
