import time
import json
import ast
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional
import warnings

from record.monitor.reader import TailReader

MAX_EVENTS = 5000
MAX_BURSTS = 300
MAX_SEGMENTS = 1000
EVENT_Y = {'key': 3, 'click': 2, 'move': 1, 'scroll': 0}
EVENT_COLOR = {'key': '#8B7FC7', 'click': '#FF6B6B', 'move': '#51CF66', 'scroll': '#FFD43B'}
MARKER_SIZE = 32
EVENT_THROTTLE_MS = 100

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"^Starting a Matplotlib GUI outside of the main thread will likely fail\.$"
)


class RealtimeVisualizer:
    def __init__(self, events_path: str, aggr_path: str, refresh_hz: int = 10, window_s: float = 30.0):
        import matplotlib.pyplot as plt

        self.events_reader = TailReader(events_path, from_start=True)
        self.aggr_reader = TailReader(aggr_path, from_start=True)

        self.events: Deque[Dict] = deque(maxlen=MAX_EVENTS)
        self.segments: Deque[Dict] = deque(maxlen=MAX_SEGMENTS)  # Store burst segments
        self.mid_markers: Deque[Dict] = deque(maxlen=MAX_BURSTS * 3)  # Store mid-state markers

        # Track pending starts and mids by burst_id
        self.pending_by_burst_id: Dict[str, List[Dict]] = {}

        self.start_time: Optional[float] = None
        self.last_shown_time: Dict[str, float] = defaultdict(lambda: -float('inf'))

        self.window_s = window_s
        self.interval_ms = int(1000.0 / refresh_hz)
        self.throttle_s = EVENT_THROTTLE_MS / 1000.0

        self._last_segment_count = 0
        self._last_event_count = 0

        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.suptitle("Real-time Input Event Visualizer", fontsize=16, fontweight='bold')

        self.ax.set_yticks(list(EVENT_Y.values()))
        self.ax.set_yticklabels(list(EVENT_Y.keys()), fontsize=11, fontweight='bold')
        self.ax.set_ylim(-0.5, max(EVENT_Y.values()) + 0.5)
        self.ax.set_ylabel("Event Type", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Time (s) relative", fontsize=12, fontweight='bold')

        self.ax.grid(True, axis='x', alpha=0.15, linestyle='--', linewidth=0.5)

        self.scatter = None
        self._burst_patches = []

    @staticmethod
    def _coarse_from_type(et: Optional[str]) -> str:
        if not et:
            return "unknown"
        if "key" in et:
            return "key"
        if "mouse" in et:
            if "move" in et:
                return "move"
            if "mouse_up" in et or "mouse_down" in et:
                return "click"
            if "scroll" in et:
                return "scroll"
            return "move"
        for k in EVENT_Y.keys():
            if k in et:
                return k
        return et

    def _parse_event_line(self, line: str) -> Optional[Dict]:
        """Try to parse an events.jsonl line. File uses python dict repr (single quotes)."""
        if not line:
            return None
        try:
            d = ast.literal_eval(line)
            et = d.get("event_type")
            coarse = self._coarse_from_type(et)
            d["coarse_type"] = coarse
            return d
        except Exception:
            try:
                d = json.loads(line)
                et = d.get("event_type")
                coarse = self._coarse_from_type(et)
                d["coarse_type"] = coarse
                return d
            except Exception:
                return None

    def _parse_aggregation_line(self, line: str) -> Optional[Dict]:
        """Parse aggregation json line (valid JSON expected)."""
        if not line:
            return None
        try:
            d = json.loads(line)
            return d
        except Exception:
            try:
                d = ast.literal_eval(line)
                return d
            except Exception:
                return None

    def _process_new_events(self, lines: List[str]):
        for line in lines:
            ev = self._parse_event_line(line)
            if not ev:
                continue
            ts = float(ev.get("timestamp", time.time()))
            if self.start_time is None:
                self.start_time = ts
            rel = ts - self.start_time
            coarse = ev.get("coarse_type", "unknown")

            if ts - self.last_shown_time[coarse] >= self.throttle_s or ev.get("event_type") not in ["mouse_move", "key_press", "key_release"]:
                self.last_shown_time[coarse] = ts
                item = {
                    "timestamp": ts,
                    "relative": rel,
                    "coarse": coarse,
                    "raw": ev
                }
                self.events.append(item)

    def _process_new_aggrs(self, lines: List[str]):
        """Aggregations are expected to have fields: timestamp, event_type, request_state ('start', 'mid', or 'end'), burst_id"""
        for line in lines:
            ag = self._parse_aggregation_line(line)
            if not ag:
                continue
            ts = float(ag.get("timestamp", time.time()))
            if self.start_time is None:
                self.start_time = ts
            etype_raw = ag.get("event_type", "unknown")
            etype = self._coarse_from_type(etype_raw)
            request_state = ag.get("request_state", "").lower()
            burst_id = ag.get("burst_id", None)

            if not burst_id:
                # Fallback if burst_id is missing - generate one
                burst_id = f"{etype}_{ts}"

            if request_state == "start":
                # Initialize the burst tracking
                if burst_id not in self.pending_by_burst_id:
                    self.pending_by_burst_id[burst_id] = []
                self.pending_by_burst_id[burst_id].append({
                    "timestamp": ts,
                    "state": "start",
                    "event_type": etype,
                    "raw": ag
                })

            elif request_state == "mid":
                # Add mid marker and create segment from last point to here
                if burst_id in self.pending_by_burst_id:
                    events = self.pending_by_burst_id[burst_id]
                    if events:
                        last_event = events[-1]
                        # Create segment from last point to this mid point
                        segment = {
                            "event_type": etype,
                            "start_ts": last_event["timestamp"],
                            "end_ts": ts,
                            "start_rel": last_event["timestamp"] - self.start_time,
                            "end_rel": ts - self.start_time,
                            "duration": ts - last_event["timestamp"],
                            "burst_id": burst_id
                        }
                        self.segments.append(segment)

                    # Add the mid event to tracking
                    self.pending_by_burst_id[burst_id].append({
                        "timestamp": ts,
                        "state": "mid",
                        "event_type": etype,
                        "raw": ag
                    })

                    # Add mid marker for visual separation
                    mid_marker = {
                        "event_type": etype,
                        "timestamp": ts,
                        "relative": ts - self.start_time,
                        "burst_id": burst_id,
                        "raw": ag
                    }
                    self.mid_markers.append(mid_marker)
                else:
                    # Mid without start - treat as standalone marker
                    mid_marker = {
                        "event_type": etype,
                        "timestamp": ts,
                        "relative": ts - self.start_time,
                        "burst_id": burst_id,
                        "raw": ag
                    }
                    self.mid_markers.append(mid_marker)

            elif request_state == "end":
                if burst_id in self.pending_by_burst_id:
                    events = self.pending_by_burst_id[burst_id]
                    if events:
                        last_event = events[-1]
                        # Create final segment from last point to end
                        segment = {
                            "event_type": etype,
                            "start_ts": last_event["timestamp"],
                            "end_ts": ts,
                            "start_rel": last_event["timestamp"] - self.start_time,
                            "end_rel": ts - self.start_time,
                            "duration": ts - last_event["timestamp"],
                            "burst_id": burst_id
                        }
                        self.segments.append(segment)

                    # Clean up this burst tracking
                    del self.pending_by_burst_id[burst_id]
                else:
                    # End without start - create a zero-duration segment
                    segment = {
                        "event_type": etype,
                        "start_ts": ts,
                        "end_ts": ts,
                        "start_rel": ts - self.start_time,
                        "end_rel": ts - self.start_time,
                        "duration": 0.0,
                        "burst_id": burst_id
                    }
                    self.segments.append(segment)

    def _read_and_update(self):
        ev_lines = self.events_reader.read_new_lines()
        ag_lines = self.aggr_reader.read_new_lines()
        if ev_lines:
            self._process_new_events(ev_lines)
        if ag_lines:
            self._process_new_aggrs(ag_lines)

    def _draw(self, frame):
        self._read_and_update()

        now_wall = time.time()
        now_ts = now_wall
        if self.events:
            now_ts = max(now_ts, self.events[-1]["timestamp"])
        if self.segments:
            now_ts = max(now_ts, max(s["end_ts"] for s in self.segments))

        if self.start_time is None:
            self.start_time = now_ts

        window_start_rel = (now_ts - self.start_time) - self.window_s
        window_end_rel = (now_ts - self.start_time)

        self._last_event_count = len(self.events)
        self._last_segment_count = len(self.segments)

        self.ax.clear()
        self.ax.set_yticks(list(EVENT_Y.values()))
        self.ax.set_yticklabels(list(EVENT_Y.keys()), fontsize=11, fontweight='bold')
        self.ax.set_ylim(-0.5, max(EVENT_Y.values()) + 0.5)
        self.ax.set_ylabel("Event Type", fontsize=12, fontweight='bold')
        self.ax.set_xlim(window_start_rel, window_end_rel + 0.0001)
        self.ax.set_xlabel("Time (s) relative", fontsize=12, fontweight='bold')
        self.ax.grid(True, axis='x', alpha=0.15, linestyle='--', linewidth=0.5)

        # Draw segments (non-overlapping by design since they're tied to burst_id)
        segments_shown = [s for s in list(self.segments)
                          if s["end_rel"] >= window_start_rel and s["start_rel"] <= window_end_rel]
        segments_shown.sort(key=lambda x: x["start_rel"])

        if segments_shown:
            for seg in segments_shown:
                coarse = seg["event_type"]
                y = EVENT_Y.get(coarse, -1)
                col = EVENT_COLOR.get(coarse, "#444444")

                s = max(seg["start_rel"], window_start_rel)
                e = min(seg["end_rel"], window_end_rel)
                d = e - s

                self.ax.barh([y], [d], left=[s], height=0.65, align='center',
                             color=col, alpha=0.25, edgecolor=col, linewidth=1.2, zorder=1)

        # Draw mid-markers as black vertical lines
        mid_markers_shown = [m for m in list(self.mid_markers)
                             if window_start_rel <= m["relative"] <= window_end_rel]
        for marker in mid_markers_shown:
            coarse = marker["event_type"]
            y = EVENT_Y.get(coarse, -1)
            x = marker["relative"]

            # Draw a vertical black line at the marker position
            self.ax.plot([x, x], [y - 0.35, y + 0.35],
                         color='black', linewidth=2, alpha=0.8, zorder=2)

        xs = []
        ys = []
        cs = []
        sizes = []
        alphas = []

        for e in list(self.events):
            rel = e["relative"]
            if rel < window_start_rel:
                continue
            coarse = e["coarse"]
            xs.append(rel)
            ys.append(EVENT_Y.get(coarse, -1))
            cs.append(EVENT_COLOR.get(coarse, "#999999"))
            age = window_end_rel - rel
            alpha = max(0.3, 1.0 - (age / self.window_s) * 0.7)
            alphas.append(alpha)
            sizes.append(MARKER_SIZE)

        if xs:
            for xi, yi, ci, si, ai in zip(xs, ys, cs, sizes, alphas):
                self.ax.scatter([xi], [yi], s=si, color=ci, alpha=ai, edgecolors='white',
                                linewidth=0.5, zorder=3)

        self.ax.axvline(window_end_rel, color='white', linewidth=1.2, alpha=0.7, linestyle=':', zorder=4)

        event_counts = {"click": 0, "move": 0, "scroll": 0, "key": 0}
        for e in self.events:
            c = e.get("coarse")
            if c in event_counts:
                event_counts[c] += 1

        # Count unique burst_ids currently being tracked
        active_bursts = len(self.pending_by_burst_id)

        info = f"Events: {len(self.events)} | Segments: {len(self.segments)} | Active: {active_bursts} | "
        info += f"[C]{event_counts['click']} [M]{event_counts['move']} [K]{event_counts['key']} [S]{event_counts['scroll']}"

        # Set window title only if using interactive backend
        import matplotlib.pyplot as plt
        if plt.get_backend() != 'Agg':
            try:
                self.fig.canvas.manager.set_window_title("Real-time Input Visualizer")
            except Exception:
                pass

        self.ax.text(0.995, 0.02, info, ha='right', va='bottom', transform=self.fig.transFigure,
                     fontsize=11, alpha=0.8, family='monospace')

    def run(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        self.ani = animation.FuncAnimation(
            self.fig, self._draw, interval=self.interval_ms,
            blit=False, cache_frame_data=False
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # On non-interactive backend, don't call show()
        if plt.get_backend() != 'Agg':
            plt.show()
