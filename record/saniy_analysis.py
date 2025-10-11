#!/usr/bin/env python3
"""
plot_aggregations_allow_singlequotes.py

Same behaviour as the previous script, but the JSONL reader now tolerates
single-quoted strings by falling back to ast.literal_eval when json.loads fails.

Usage:
    python plot_aggregations_allow_singlequotes.py /path/to/folder
    python plot_aggregations_allow_singlequotes.py /path/to/folder --out out.png
"""

from pathlib import Path
import argparse
import json
import sys
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------------------
# Mappings / config
# ---------------------------
INNER_TO_CATEGORY = {
    "mouse_down": "click",
    "mouse_up": "click",
    "mouse_move": "move",
    "mouse_scroll": "scroll",
    "key_press": "key",
    "key_release": "key",
}

CATEGORIES = ["click", "move", "scroll", "key"]
UNMATCHED_CATEGORY = "unmatched"
ALL_PLOTTING_CATEGORIES = CATEGORIES + [UNMATCHED_CATEGORY]

CATEGORY_COLORS = {
    "click": "#ffb3b3",
    "move":  "#b3d1ff",
    "scroll":"#b3ffc9",
    "key":   "#f0e68c",
    "unmatched": "#d3d3d3",
}

# ---------------------------
# Helpers for timestamp matching
# ---------------------------
def ts_to_key(ts):
    try:
        return int(round(float(ts) * 1_000_000))
    except Exception:
        return None

def ts_to_dt(ts):
    return datetime.fromtimestamp(float(ts))

# ---------------------------
# Read JSONL files (with fallback to ast.literal_eval)
# ---------------------------
def read_jsonl(path: Path):
    objs = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # Try strict JSON first
            try:
                obj = json.loads(line)
                objs.append(obj)
                continue
            except json.JSONDecodeError:
                pass
            # Fall back to Python literal parsing (accepts single quotes, True/False/None)
            try:
                obj = ast.literal_eval(line)
                # ensure the result is a mapping-like object (dict)
                if isinstance(obj, dict):
                    objs.append(obj)
                else:
                    print(f"Warning: line {i} parsed but is not a dict; skipping. Path={path}", file=sys.stderr)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: skipping invalid JSON on line {i} of {path}: {e}", file=sys.stderr)
    return objs

# ---------------------------
# Collect inner events from aggregations.jsonl
# ---------------------------
def collect_inner_events_from_aggregations(objects):
    events = {cat: [] for cat in CATEGORIES}
    keyset = set()
    for obj in objects:
        for ev in obj.get("events", []) or []:
            ev_type = ev.get("event_type")
            cat = None
            if ev_type:
                cat = INNER_TO_CATEGORY.get(str(ev_type).strip().lower())
            if cat:
                ts = ev.get("timestamp")
                try:
                    fts = float(ts)
                except Exception:
                    # sometimes timestamp might be a string with quotes, try to clean
                    try:
                        fts = float(str(ts).strip())
                    except Exception:
                        continue
                events[cat].append(fts)
                k = ts_to_key(fts)
                if k is not None:
                    keyset.add(k)
    for cat in events:
        events[cat].sort()
    return events, keyset

# ---------------------------
# Collect outer intervals from aggregations.jsonl
# ---------------------------
def collect_outer_intervals(objects):
    intervals = {cat: [] for cat in CATEGORIES}
    n = len(objects)
    for i, obj in enumerate(objects):
        if not obj.get("is_start"):
            continue
        typ = obj.get("event_type")
        if typ not in CATEGORIES:
            continue
        start_ts = obj.get("timestamp")
        if start_ts is None:
            continue
        end_ts = None
        for j in range(i + 1, n):
            o2 = objects[j]
            if o2.get("event_type") == typ and not o2.get("is_start"):
                end_ts = o2.get("timestamp")
                break
        if end_ts is not None:
            try:
                intervals[typ].append((float(start_ts), float(end_ts)))
            except Exception:
                continue
        else:
            print(f"Warning: no matching end found for start object at ts={start_ts}, type={typ}", file=sys.stderr)
    for cat in intervals:
        intervals[cat].sort()
    return intervals

# ---------------------------
# Collect timestamps from events.jsonl
# ---------------------------
def collect_timestamps_from_events_file(objects):
    ts_list = []
    for obj in objects:
        if "timestamp" in obj and not isinstance(obj.get("timestamp"), dict):
            try:
                ts_list.append(float(obj["timestamp"]))
            except Exception:
                try:
                    ts_list.append(float(str(obj["timestamp"]).strip()))
                except Exception:
                    pass
        if isinstance(obj.get("events"), list):
            for ev in obj.get("events", []):
                if ev and ("timestamp" in ev):
                    try:
                        ts_list.append(float(ev["timestamp"]))
                    except Exception:
                        try:
                            ts_list.append(float(str(ev["timestamp"]).strip()))
                        except Exception:
                            pass
        if "event" in obj and isinstance(obj["event"], dict) and "timestamp" in obj["event"]:
            try:
                ts_list.append(float(obj["event"]["timestamp"]))
            except Exception:
                try:
                    ts_list.append(float(str(obj["event"]["timestamp"]).strip()))
                except Exception:
                    pass
    return sorted(ts_list)

# ---------------------------
# Plotting
# ---------------------------
def plot_all(events_by_cat, intervals_by_cat, unmatched_ts, out_path=None, show=True):
    num_rows = len(ALL_PLOTTING_CATEGORIES)
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, sharex=True, figsize=(14, 2.2 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = [axes]
    xfmt = mdates.DateFormatter("%H:%M:%S.%f")

    for ax, cat in zip(axes, ALL_PLOTTING_CATEGORIES):
        ax.set_title(cat if cat != UNMATCHED_CATEGORY else f"{UNMATCHED_CATEGORY} (events in events.jsonl not in aggregations)")
        if cat in events_by_cat:
            xs = events_by_cat.get(cat, [])
            if xs:
                x_dt = [ts_to_dt(ts) for ts in xs]
                jitter = (np.random.rand(len(x_dt)) - 0.5) * 0.2
                y_vals = np.zeros(len(x_dt)) + jitter
                ax.scatter(x_dt, y_vals, s=18, edgecolors="k", linewidths=0.3, zorder=5)
        if cat == UNMATCHED_CATEGORY:
            if unmatched_ts:
                x_dt_un = [ts_to_dt(ts) for ts in unmatched_ts]
                jitter = (np.random.rand(len(x_dt_un)) - 0.5) * 0.2
                y_vals = np.zeros(len(x_dt_un)) + jitter
                ax.scatter(x_dt_un, y_vals, s=22, marker="x", label="unmatched events", zorder=6)
            else:
                ax.plot([], [])
        if cat in intervals_by_cat:
            for (s_ts, e_ts) in intervals_by_cat.get(cat, []):
                s_dt = ts_to_dt(s_ts)
                e_dt = ts_to_dt(e_ts)
                ax.axvspan(s_dt, e_dt, alpha=0.25, color=CATEGORY_COLORS.get(cat, "#dddddd"), zorder=0)
        ax.set_yticks([])
        ax.set_ylim(-1.0, 1.0)

    axes[-1].xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate(rotation=45, ha="right")
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")
    if show:
        plt.show()
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot aggregations.jsonl and mark events.jsonl timestamps not present in aggregations.")
    parser.add_argument("folder", type=Path, help="Folder that contains aggregations.jsonl and events.jsonl")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Optional output PNG file")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful for headless runs)")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder does not exist or is not a directory: {folder}", file=sys.stderr)
        sys.exit(2)

    agg_path = folder / "aggregations.jsonl"
    events_path = folder / "events.jsonl"

    if not agg_path.exists():
        print(f"Error: aggregations.jsonl not found in {folder}", file=sys.stderr)
        sys.exit(2)
    if not events_path.exists():
        print(f"Error: events.jsonl not found in {folder}", file=sys.stderr)
        sys.exit(2)

    agg_objs = read_jsonl(agg_path)
    events_objs = read_jsonl(events_path)

    if not agg_objs:
        print("Warning: no objects parsed from aggregations.jsonl", file=sys.stderr)
    if not events_objs:
        print("Warning: no objects parsed from events.jsonl", file=sys.stderr)

    events_by_cat, agg_keyset = collect_inner_events_from_aggregations(agg_objs)
    intervals_by_cat = collect_outer_intervals(agg_objs)
    raw_event_ts = collect_timestamps_from_events_file(events_objs)
    raw_event_keys = [(ts, ts_to_key(ts)) for ts in raw_event_ts]

    unmatched = []
    for ts, key in raw_event_keys:
        if key is None:
            unmatched.append(ts)
            continue
        if key not in agg_keyset:
            unmatched.append(ts)

    print(f"Aggregations inner events per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(events_by_cat.get(cat, []))} timestamps, {len(intervals_by_cat.get(cat, []))} intervals")
    print(f"Events.jsonl total timestamps found: {len(raw_event_ts)}")
    print(f"Unmatched events (not present in aggregations inner events): {len(unmatched)}")

    plot_all(events_by_cat, intervals_by_cat, unmatched, out_path=str(args.out) if args.out else None, show=not args.no_show)

if __name__ == "__main__":
    main()
