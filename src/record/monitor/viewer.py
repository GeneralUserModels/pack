import time
import json
import ast
import sys
import threading
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional
import warnings

from record.monitor.reader import TailReader

MAX_EVENTS = 5000
MAX_BURSTS = 300
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
        import matplotlib.animation as animation
        
        self.events_reader = TailReader(events_path, from_start=True)
        self.aggr_reader = TailReader(aggr_path, from_start=True)

        self.events: Deque[Dict] = deque(maxlen=MAX_EVENTS)
        self.bursts: Deque[Dict] = deque(maxlen=MAX_BURSTS)

        self.pending_starts: Dict[str, List[Dict]] = defaultdict(list)

        self.start_time: Optional[float] = None
        self.last_shown_time: Dict[str, float] = defaultdict(lambda: -float('inf'))

        self.window_s = window_s
        self.interval_ms = int(1000.0 / refresh_hz)
        self.throttle_s = EVENT_THROTTLE_MS / 1000.0

        self._last_burst_count = 0
        self._last_event_count = 0
        self._cached_bursts = {}

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
        """Aggregations are expected to have fields: timestamp, event_type, is_start (bool)"""
        for line in lines:
            ag = self._parse_aggregation_line(line)
            if not ag:
                continue
            ts = float(ag.get("timestamp", time.time()))
            if self.start_time is None:
                self.start_time = ts
            etype_raw = ag.get("event_type", "unknown")
            etype = self._coarse_from_type(etype_raw)
            is_start = bool(ag.get("is_start", False))
            if is_start:
                self.pending_starts[etype].append({"timestamp": ts, "raw": ag})
            else:
                if self.pending_starts.get(etype):
                    start = self.pending_starts[etype].pop(0)
                    burst = {
                        "event_type": etype,
                        "start_ts": start["timestamp"],
                        "end_ts": ts,
                        "start_rel": start["timestamp"] - self.start_time,
                        "end_rel": ts - self.start_time,
                        "duration": ts - start["timestamp"],
                        "raw_start": start["raw"],
                        "raw_end": ag
                    }
                    self.bursts.append(burst)
                else:
                    burst = {
                        "event_type": etype,
                        "start_ts": ts,
                        "end_ts": ts,
                        "start_rel": ts - self.start_time,
                        "end_rel": ts - self.start_time,
                        "duration": 0.0,
                        "raw_start": None,
                        "raw_end": ag
                    }
                    self.bursts.append(burst)

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
        if self.bursts:
            now_ts = max(now_ts, max(b["end_ts"] for b in self.bursts))

        if self.start_time is None:
            self.start_time = now_ts

        window_start_rel = (now_ts - self.start_time) - self.window_s
        window_end_rel = (now_ts - self.start_time)

        self._last_event_count = len(self.events)
        self._last_burst_count = len(self.bursts)

        self.ax.clear()
        self.ax.set_yticks(list(EVENT_Y.values()))
        self.ax.set_yticklabels(list(EVENT_Y.keys()), fontsize=11, fontweight='bold')
        self.ax.set_ylim(-0.5, max(EVENT_Y.values()) + 0.5)
        self.ax.set_ylabel("Event Type", fontsize=12, fontweight='bold')
        self.ax.set_xlim(window_start_rel, window_end_rel + 0.0001)
        self.ax.set_xlabel("Time (s) relative", fontsize=12, fontweight='bold')
        self.ax.grid(True, axis='x', alpha=0.15, linestyle='--', linewidth=0.5)

        bursts_shown = [b for b in list(self.bursts) if b["end_rel"] >= window_start_rel and b["start_rel"] <= window_end_rel]
        bursts_shown.sort(key=lambda x: x["start_rel"])

        if bursts_shown:
            for b in bursts_shown:
                coarse = b["event_type"]
                y = EVENT_Y.get(coarse, -1)
                col = EVENT_COLOR.get(coarse, "#444444")

                s = max(b["start_rel"], window_start_rel)
                e = min(b["end_rel"], window_end_rel)
                d = e - s

                self.ax.barh([y], [d], left=[s], height=0.65, align='center',
                             color=col, alpha=0.25, edgecolor=col, linewidth=1.2, zorder=1)

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

        info = f"Events: {len(self.events)} | Bursts: {len(self.bursts)} | "
        info += f"[C]{event_counts['click']} [M]{event_counts['move']} [K]{event_counts['key']} [S]{event_counts['scroll']}"

        # Set window title only if using interactive backend
        import matplotlib.pyplot as plt
        if plt.get_backend() != 'Agg':
            try:
                self.fig.canvas.manager.set_window_title("Real-time Input Visualizer")
            except:
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
