from collections import Counter
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import ast
from datetime import datetime
import re

INNER_TO_CATEGORY = {
    "mouse_down": "click",
    "mouse_up": "click",
    "mouse_move": "move",
    "mouse_scroll": "scroll",
    "key_press": "key",
    "key_release": "key",
}

CATEGORIES = ["click", "move", "scroll", "key"]
DUPLICATES_CATEGORY = "duplicates"
UNMATCHED_CATEGORY = "unmatched"
SCREENSHOT_CATEGORY = "screenshots"
# add screenshots row before duplicates/unmatched
ALL_PLOTTING_CATEGORIES = CATEGORIES + [SCREENSHOT_CATEGORY, DUPLICATES_CATEGORY, UNMATCHED_CATEGORY]

CATEGORY_COLORS = {'key': '#8B7FC7', 'click': '#FF6B6B', 'move': '#51CF66', 'scroll': '#FFD43B'}


def ts_to_key(ts):
    """
    Convert a timestamp (float or string) to an integer key for robust comparison.
    Use microsecond precision (multiply by 1e6 and round).
    """
    try:
        return int(round(float(ts) * 1_000_000))
    except Exception:
        return None


def ts_to_dt(ts):
    return datetime.fromtimestamp(float(ts))


def read_jsonl(path: Path):
    objs = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                objs.append(obj)
                continue
            except json.JSONDecodeError:
                pass
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    objs.append(obj)
                else:
                    print(f"Warning: line {i} parsed but is not a dict; skipping. Path={path}", file=sys.stderr)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: skipping invalid JSON on line {i} of {path}: {e}", file=sys.stderr)
    return objs


def collect_inner_events_and_counts(objects):
    """
    Return:
      - events_by_cat: dict category -> sorted list[timestamp floats]
      - agg_key_counts: Counter mapping integer timestamp key -> occurrence count in aggregations inner events
    """
    events = {cat: [] for cat in CATEGORIES}
    key_counts = Counter()
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
                    try:
                        fts = float(str(ts).strip())
                    except Exception:
                        continue
                events[cat].append(fts)
                k = ts_to_key(fts)
                if k is not None:
                    key_counts[k] += 1
    for cat in events:
        events[cat].sort()
    return events, key_counts


def collect_outer_intervals(objects):
    """
    Collect intervals and state markers using request_state (start/mid/end).
    Returns:
      - intervals: dict category -> list of (start_ts, end_ts) tuples
      - state_markers: dict category -> list of (timestamp, state) tuples where state is 'start', 'mid', or 'end'
    """
    intervals = {cat: [] for cat in CATEGORIES}
    state_markers = {cat: [] for cat in CATEGORIES}

    # Track open intervals by category
    open_intervals = {cat: None for cat in CATEGORIES}

    n = len(objects)
    for i, obj in enumerate(objects):
        typ = obj.get("event_type")
        if typ not in CATEGORIES:
            continue

        req_state = obj.get("request_state")
        if not req_state:
            continue

        ts = obj.get("timestamp")
        if ts is None:
            continue

        try:
            fts = float(ts)
        except Exception:
            continue

        # Add state marker for visualization
        state_markers[typ].append((fts, req_state))

        # Handle interval tracking
        if req_state == "start":
            # Start a new interval
            open_intervals[typ] = fts
        elif req_state == "end":
            # Close the interval if one is open
            if open_intervals[typ] is not None:
                intervals[typ].append((open_intervals[typ], fts))
                open_intervals[typ] = None

    # Sort intervals and markers
    for cat in intervals:
        intervals[cat].sort()
        state_markers[cat].sort()

    return intervals, state_markers


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
        # inner events list
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


def collect_screenshots_from_aggregations(objects):
    """
    Parse screenshot timestamps and reasons from aggregation objects' "screenshot_path".
    Returns list of dicts: [{'ts': float_timestamp, 'reason': reason_str, 'color': color_str}, ...]
    Expected filename pattern examples:
      1760702571.228687_reason_move_start.jpg
      1760702571.228687-some-other-pattern_reason_click.jpg
    The parser attempts to extract a leading float timestamp and the 'reason' piece before the extension.
    """
    screenshots = []
    # regex: capture float (digits + dot + digits), then optionally any chars, then _reason_ then reason (non-greedy) until dot (extension)
    pattern = re.compile(r"(?P<ts>\d+\.\d+)[^/]*?(?:_reason_|-reason-)?(?P<reason>[^.]+)\.")
    for obj in objects:
        sp = obj.get("screenshot_path")
        if not sp or not isinstance(sp, str):
            continue
        fname = Path(sp).name
        m = pattern.search(fname)
        if not m:
            # fallback: try to parse leading float timestamp only
            m2 = re.match(r"(?P<ts>\d+\.\d+)", fname)
            if m2:
                try:
                    ts = float(m2.group("ts"))
                except Exception:
                    continue
                reason = "unknown"
            else:
                continue
        else:
            try:
                ts = float(m.group("ts"))
            except Exception:
                continue
            reason = m.group("reason")
        # normalize reason: pick a category keyword if present in reason
        reason_lower = reason.lower()
        color = "#888888"  # default grey
        matched_cat = None
        for cat in CATEGORIES:
            if cat in reason_lower:
                matched_cat = cat
                color = CATEGORY_COLORS.get(cat, color)
                break
        # also try to match keys from INNER_TO_CATEGORY (e.g., mouse_* -> click)
        if matched_cat is None:
            for k, v in INNER_TO_CATEGORY.items():
                if k in reason_lower:
                    matched_cat = v
                    color = CATEGORY_COLORS.get(v, color)
                    break
        screenshots.append({'ts': ts, 'reason': reason, 'color': color})
    # sort by timestamp
    screenshots.sort(key=lambda s: s['ts'])
    return screenshots


def plot_all(events_by_cat, intervals_by_cat, state_markers_by_cat, duplicates_ts, unmatched_ts, screenshots, out_path=None):
    num_rows = len(ALL_PLOTTING_CATEGORIES)
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, sharex=True, figsize=(14, 2.2 * num_rows), constrained_layout=True)
    if num_rows == 1:
        axes = [axes]

    xfmt = mdates.DateFormatter("%H:%M:%S.%f")

    for ax, cat in zip(axes, ALL_PLOTTING_CATEGORIES):
        title = cat
        if cat == DUPLICATES_CATEGORY:
            title = f"{DUPLICATES_CATEGORY} (events.jsonl timestamps present more than once in aggregations)"
        elif cat == UNMATCHED_CATEGORY:
            title = f"{UNMATCHED_CATEGORY} (events.jsonl timestamps not in aggregations)"
        elif cat == SCREENSHOT_CATEGORY:
            title = f"{SCREENSHOT_CATEGORY} (screenshot timestamps from aggregations)"
        ax.set_title(title)

        # plot inner events (only for first 4 categories)
        if cat in events_by_cat:
            xs = events_by_cat.get(cat, [])
            if xs:
                x_dt = [ts_to_dt(ts) for ts in xs]
                jitter = (np.random.rand(len(x_dt)) - 0.5) * 0.2
                y_vals = np.zeros(len(x_dt)) + jitter
                ax.scatter(x_dt, y_vals, s=18, edgecolors="k", linewidths=0.3, zorder=5)

        # screenshots row -> vertical lines colored by reason/category
        if cat == SCREENSHOT_CATEGORY:
            if screenshots:
                for s in screenshots:
                    try:
                        s_dt = ts_to_dt(s['ts'])
                    except Exception:
                        continue
                    ax.axvline(s_dt, color=s.get('color', '#888888'), linestyle='-', linewidth=1.2, alpha=0.9, zorder=7)
                # optionally place small markers to make them more visible
                x_dt_markers = [ts_to_dt(s['ts']) for s in screenshots]
                y_markers = np.zeros(len(x_dt_markers)) + 0.0
                ax.scatter(x_dt_markers, y_markers, s=28, marker='|', linewidths=1.2, zorder=8)
            else:
                ax.plot([], [])

        # duplicates row
        if cat == DUPLICATES_CATEGORY:
            if duplicates_ts:
                x_dt_dup = [ts_to_dt(ts) for ts in duplicates_ts]
                jitter = (np.random.rand(len(x_dt_dup)) - 0.5) * 0.25
                y_vals = np.zeros(len(x_dt_dup)) + jitter
                ax.scatter(x_dt_dup, y_vals, s=36, marker="D", label="duplicates", zorder=6)
            else:
                ax.plot([], [])

        # unmatched row
        if cat == UNMATCHED_CATEGORY:
            if unmatched_ts:
                x_dt_un = [ts_to_dt(ts) for ts in unmatched_ts]
                jitter = (np.random.rand(len(x_dt_un)) - 0.5) * 0.25
                y_vals = np.zeros(len(x_dt_un)) + jitter
                ax.scatter(x_dt_un, y_vals, s=20, marker="x", label="unmatched", zorder=6)
            else:
                ax.plot([], [])

        # shade intervals for the main categories (start to end boxes)
        if cat in intervals_by_cat:
            for (s_ts, e_ts) in intervals_by_cat.get(cat, []):
                s_dt = ts_to_dt(s_ts)
                e_dt = ts_to_dt(e_ts)
                ax.axvspan(
                    s_dt, e_dt,
                    facecolor=CATEGORY_COLORS.get(cat, "#dddddd"),
                    edgecolor="black",
                    alpha=0.25,
                    linewidth=1.0,
                    zorder=0
                )

        # draw vertical black lines for state markers (start, mid, end)
        if cat in state_markers_by_cat:
            for (ts, state) in state_markers_by_cat.get(cat, []):
                try:
                    ts_dt = ts_to_dt(ts)
                except Exception:
                    continue
                ax.axvline(ts_dt, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=4)

        ax.set_yticks([])
        ax.set_ylim(-1.0, 1.0)

    axes[-1].xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate(rotation=45, ha="right")

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to: {out_path}")


def plot_summary_stats(directory: Path = Path("."), agg_path: Path = Path("raw_aggregations.jsonl"), events_path: Path = Path("events.jsonl"), summary_path: Path = Path("summary.png")):
    agg_objs = read_jsonl(agg_path)
    events_objs = read_jsonl(events_path)

    if not agg_objs:
        print("Warning: no objects parsed from raw_aggregations.jsonl", file=sys.stderr)
    if not events_objs:
        print("Warning: no objects parsed from events.jsonl", file=sys.stderr)

    events_by_cat, agg_key_counts = collect_inner_events_and_counts(agg_objs)
    intervals_by_cat, state_markers_by_cat = collect_outer_intervals(agg_objs)

    # collect screenshots (new)
    screenshots = collect_screenshots_from_aggregations(agg_objs)

    raw_event_ts = collect_timestamps_from_events_file(events_objs)
    raw_event_keys = [(ts, ts_to_key(ts)) for ts in raw_event_ts]

    unmatched = []
    duplicates = []
    matched_unique = []

    for ts, key in raw_event_keys:
        if key is None:
            unmatched.append(ts)
            continue
        cnt = agg_key_counts.get(key, 0)
        if cnt == 0:
            unmatched.append(ts)
        elif cnt == 1:
            matched_unique.append(ts)
        else:
            duplicates.append((ts, cnt))

    duplicates_sorted = sorted(duplicates, key=lambda x: x[0])
    dup_timestamps = [ts for ts, c in duplicates_sorted]
    dup_counts = {ts: c for ts, c in duplicates_sorted}
    unmatched_sorted = sorted(unmatched)

    print("Aggregations inner events per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(events_by_cat.get(cat, []))} timestamps, {len(intervals_by_cat.get(cat, []))} intervals, {len(state_markers_by_cat.get(cat, []))} state markers")
    print(f"Events.jsonl total timestamps found: {len(raw_event_ts)}")
    print(f"Matched uniquely (present exactly once in aggregations): {len(matched_unique)}")
    print(f"Duplicated in aggregations (present >1 times): {len(dup_timestamps)}")
    if dup_timestamps:
        print("Duplicated timestamps and counts (timestamp -> count):")
        for ts in dup_timestamps:
            print(f"  {ts} -> {dup_counts.get(ts)}")
    print(f"Unmatched events (not present in aggregations): {len(unmatched_sorted)}")

    plot_duplicates_ts = dup_timestamps
    plot_unmatched_ts = unmatched_sorted

    plot_all(events_by_cat, intervals_by_cat, state_markers_by_cat, plot_duplicates_ts, plot_unmatched_ts,
             screenshots, out_path=summary_path)


if __name__ == "__main__":
    # if run without args, uses default filenames; adjust as you need
    plot_summary_stats()
