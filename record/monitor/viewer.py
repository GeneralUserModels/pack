import time
import json
import ast
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from record.monitor.reader import TailReader

MAX_EVENTS = 10000
MAX_BURSTS = 500
EVENT_Y = {'key': 3, 'click': 2, 'move': 1, 'scroll': 0}
EVENT_COLOR = {'key': 'tab:purple', 'click': 'tab:red', 'move': 'tab:green', 'scroll': 'tab:orange'}
MARKER_SIZE = 24
EVENT_THROTTLE_MS = 75


class RealtimeVisualizer:
    def __init__(self, events_path: str, aggr_path: str, refresh_hz: int = 16, window_s: float = 30.0):
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

        plt.rcParams["toolbar"] = "toolbar2"
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.suptitle("Realtime Input Visualizer (events + aggregations)")
        self.ax.set_yticks(list(EVENT_Y.values()))
        self.ax.set_yticklabels(list(EVENT_Y.keys()))
        self.ax.set_ylim(-0.5, max(EVENT_Y.values()) + 0.5)
        self.ax.set_ylabel("Event Type")
        self.ax.set_xlabel("Time (s) relative")

        self.scatter = None

        self.ax.grid(True, axis='x', alpha=0.25)

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

            # Throttle: only show one event per type every EVENT_THROTTLE_MS
            if ts - self.last_shown_time[coarse] >= self.throttle_s or ev.get("event_type") != "mouse_move":
                self.last_shown_time[coarse] = ts
                item = {
                    "timestamp": ts,
                    "relative": rel,
                    "coarse": coarse,
                    "raw": ev
                }
                self.events.append(item)

    def _process_new_aggrs(self, lines: List[str]):
        """
        Aggregations are expected to have fields:
          timestamp, event_type, is_start (bool)
        We pair start/end into bursts
        """
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

        self.ax.cla()
        self.ax.set_yticks(list(EVENT_Y.values()))
        self.ax.set_yticklabels(list(EVENT_Y.keys()))
        self.ax.set_ylim(-0.5, max(EVENT_Y.values()) + 0.5)
        self.ax.set_ylabel("Event Type")
        self.ax.grid(True, axis='x', alpha=0.25)
        self.ax.set_xlim(window_start_rel, window_end_rel + 0.0001)
        self.ax.set_xlabel("Time (s) relative")

        bursts_shown = [b for b in list(self.bursts) if b["end_rel"] >= window_start_rel and b["start_rel"] <= window_end_rel]
        bursts_shown.sort(key=lambda x: x["start_rel"])

        if bursts_shown:
            starts = np.array([max(b["start_rel"], window_start_rel) for b in bursts_shown])
            ends = np.array([min(b["end_rel"], window_end_rel) for b in bursts_shown])
            durations = ends - starts
            for i, (s, d, b) in enumerate(zip(starts, durations, bursts_shown)):
                coarse = b["event_type"]
                y = EVENT_Y.get(coarse, -1)
                if y < -0.4:
                    y = -0.75
                col = EVENT_COLOR.get(coarse, "tab:gray")
                self.ax.barh([y], [d], left=[s], height=0.6 * 0.8, align='center',
                             color=col, alpha=0.22, edgecolor="black", linewidth=0.4, zorder=1)
                label_x = s + d + 0.01
                if label_x < window_end_rel:
                    self.ax.text(label_x, y, coarse, va='center', fontsize=7, alpha=0.7, zorder=2)

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
            cs.append(EVENT_COLOR.get(coarse, "black"))
            age = window_end_rel - rel
            alpha = max(0.12, 1.0 - (age / self.window_s))
            alphas.append(alpha)
            sizes.append(MARKER_SIZE)

        for xi, yi, ci, si, ai in zip(xs, ys, cs, sizes, alphas):
            self.ax.scatter([xi], [yi], s=si, color=ci, alpha=ai, edgecolors='none', zorder=3)

        self.ax.axvline(window_end_rel, color='k', linewidth=0.6, alpha=0.6, zorder=4)

        info = f"Window: {self.window_s}s | events: {len(self.events)} | bursts stored: {len(self.bursts)}"
        self.fig.canvas.manager.set_window_title("Realtime Input Visualizer")
        self.ax.text(0.995, 0.01, info, ha='right', va='bottom', transform=self.fig.transFigure, fontsize=8, alpha=0.7)

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self._draw, interval=self.interval_ms, blit=False, cache_frame_data=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
