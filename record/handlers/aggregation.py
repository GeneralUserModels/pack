import threading
import queue
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from record.models.image import BufferImage


class AggregationHandler:
    """
    Simplified Aggregation worker: cluster events by group (mouse/key/etc),
    ignore mouse_move/mouse_scroll, and save screenshots 50ms before cluster
    start and 50ms after cluster end.

    Important: down/up pairs (mouse_down + mouse_up) and press/release
    (key_press + key_release) are grouped into the same cluster.
    """

    # --- tuning parameters ---
    MERGE_ACTION_GAP = 2  # seconds: gap threshold to cluster nearby events
    CLUSTER_PAD = 0.05       # seconds: pad before/after cluster to pick screenshots
    DEDUP_ROUND = 2          # decimals for dedupe rounding (0.01s granularity)
    MIN_CLUSTER_DURATION = 0.05  # seconds: minimum cluster duration (except clicks)

    # event filtering / grouping
    IGNORED_EVENT_TYPES = ()  # ("mouse_move", "mouse_scroll")
    # canonical event names you expect in events (keeps compatibility); grouping below maps them to groups
    EVENT_TYPES = ("mouse_down", "mouse_up", "key_press", "key_release") + ("mouse_move", "mouse_scroll")

    # map individual event types to a grouping key used to form clusters.
    # This ensures mouse_down + mouse_up end up in the same cluster (group 'mouse'),
    # and key_press + key_release end up in the same cluster (group 'keyboard').
    GROUP_FOR_EVENT = {
        "mouse_down": "click",
        "mouse_up": "click",
        "mouse_move": "move",     # ignored; kept here for completeness
        "mouse_scroll": "scroll",   # ignored; kept here for completeness
        "key_press": "keyboard",
        "key_release": "keyboard",
    }

    # Which event-types count as explicit clicks (exempt from min-duration rule)
    CLICK_EVENT_TYPES = ("mouse_down", "mouse_up", "mouse_click")

    def __init__(
        self,
        image_queue,
        ssim_queue,
        input_event_queue,
        save_worker,
        session_dir: Optional[Path] = None,
        debug: bool = False,
        queue_size: int = 12,
    ):
        self.image_queue = image_queue
        # ssim_queue is accepted for compatibility but not used in this simplified version
        self.ssim_queue = ssim_queue
        self.input_event_queue = input_event_queue
        self.save_worker = save_worker
        self.session_dir = Path(session_dir) if session_dir is not None else save_worker.session_dir
        self.debug = debug
        self.queue_size = queue_size

        self._proc_queue: "queue.Queue[BufferImage]" = queue.Queue(maxsize=1024)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._saved_pairs = set()  # dedupe by (group, start_ts_rounded, end_ts_rounded)
        self._agg_log = self.session_dir / "aggregated_actions.jsonl"

    # ----------------- lifecycle -----------------
    def start(self):
        if self._running:
            return
        self._running = True

        def cb(_ssim_item):
            try:
                self._proc_queue.put_nowait(True)
            except queue.Full:
                if self.debug:
                    print("AggregationHandler: processing queue full, dropping notification")

        # store callback but don't rely on SSIM content
        self._ssim_callback = cb
        try:
            self.ssim_queue.add_callback(self._ssim_callback)
        except Exception:
            # if ssim_queue doesn't support callbacks, we silently continue
            pass

        self._thread = threading.Thread(target=self._worker_loop, name="AggregationHandlerThread", daemon=True)
        self._thread.start()
        if self.debug:
            print("AggregationHandler (simplified) started")

    def stop(self, wait_timeout: float = 2.0):
        if not self._running:
            return
        self._running = False
        try:
            if hasattr(self, "_ssim_callback"):
                self.ssim_queue.remove_callback(self._ssim_callback)
        except Exception:
            pass
        try:
            self._proc_queue.put_nowait(None)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=wait_timeout)
        if self.debug:
            print("AggregationHandler (simplified) stopped")

    def _worker_loop(self):
        while self._running:
            try:
                item = self._proc_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                # Run the simplified pipeline
                self._process_pipeline()
            except Exception as e:
                print(f"AggregationHandler (simplified): error processing: {e}")

    def _log(self, *args):
        if self.debug:
            print("[AggregationHandler simplified]", *args)

    # ----------------- helpers: screenshots & events -----------------
    def _get_all_screenshots(self) -> List[BufferImage]:
        imgs = self.image_queue.get_all()
        return sorted(imgs, key=lambda x: x.timestamp)

    def _expand_and_sort_events(self, screenshots: List[BufferImage]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        raw_events = self.input_event_queue.get_all()
        for ev in raw_events:
            if hasattr(ev, "to_dict"):
                d = ev.to_dict()
            elif isinstance(ev, dict):
                d = dict(ev)
            else:
                d = {"timestamp": getattr(ev, "timestamp", getattr(ev, "time", None)), "repr": str(ev)}
            t = d.get("timestamp", d.get("time"))
            try:
                t = float(t) if t is not None else None
            except Exception:
                t = None
            if t is not None:
                d["time"] = t
                events.append(d)

        for s in screenshots:
            evs = getattr(s, "events", []) or []
            for ev in evs:
                rel = float(ev.get("relative_time", 0.0)) if ev.get("relative_time") is not None else 0.0
                abs_t = s.timestamp + rel
                normalized = {
                    "time": abs_t,
                    "event_type": ev.get("event_type"),
                    "details": ev.get("details", {}),
                    "cursor_pos": ev.get("cursor_pos"),
                    "monitor_id": ev.get("monitor_id", getattr(s, "monitor_index", None)),
                }
                events.append(normalized)

        events = [e for e in events if "time" in e and e["time"] is not None]
        events.sort(key=lambda x: x["time"])
        return events

    def _cluster_event_times(self, times: List[Dict[str, Any]], max_gap: float):
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

    def _find_closest_screenshot(self, target_time: float, screenshots: List[BufferImage]) -> Tuple[Optional[BufferImage], str]:
        """Return the screenshot closest to target_time (any screenshot, SSIM not required)."""
        if not screenshots:
            return None, "no_screenshots"
        closest = min(screenshots, key=lambda s: abs(s.timestamp - target_time))
        delta = abs(closest.timestamp - target_time)
        return closest, f"closest_delta={delta:.3f}"

    # ----------------- simplified pipeline -----------------
    def _process_pipeline(self):
        screenshots_all = self._get_all_screenshots()
        if not screenshots_all:
            return

        events = self._expand_and_sort_events(screenshots_all)
        if not events:
            self._log("No events to aggregate")
            return

        # Partition events by event group (mouse / keyboard / unknown), ignoring configured types
        group_map: Dict[str, List[Dict[str, Any]]] = {}
        for e in events:
            raw_et = e.get("event_type") or "unknown"
            if raw_et in self.IGNORED_EVENT_TYPES:
                continue
            group = self.GROUP_FOR_EVENT.get(raw_et, raw_et or "unknown")
            # normalize: treat unknown as "unknown"
            group = group or "unknown"
            group_map.setdefault(group, []).append(e)

        if not group_map:
            self._log("No event groups to process after filtering")
            return

        saved = 0
        for group, ev_list in group_map.items():
            clusters = self._cluster_event_times(ev_list, self.MERGE_ACTION_GAP)
            if not clusters:
                continue

            for start, end, cluster_events in clusters:
                # enforce minimum cluster duration (except for clicks)
                duration = end - start
                # a click is any original event that is in CLICK_EVENT_TYPES
                has_click = any((e.get("event_type") in self.CLICK_EVENT_TYPES) for e in cluster_events)
                if duration < self.MIN_CLUSTER_DURATION and not has_click:
                    if self.debug:
                        self._log(f"Skipping cluster too short: group={group} dur={duration:.3f}s events={len(cluster_events)}")
                    continue

                # dedupe by group and rounded cluster times
                cluster_key = (group, round(start, self.DEDUP_ROUND), round(end, self.DEDUP_ROUND))
                if cluster_key in self._saved_pairs:
                    self._log("Skipping already-saved cluster", cluster_key)
                    continue

                # desired snapshot times with pad
                desired_start = start - self.CLUSTER_PAD
                desired_end = end + self.CLUSTER_PAD

                start_s, start_reason = self._find_closest_screenshot(desired_start, screenshots_all)
                end_s, end_reason = self._find_closest_screenshot(desired_end, screenshots_all)

                if start_s is None and end_s is None:
                    if self.debug:
                        self._log("No screenshots available for cluster", group, start, end)
                    continue

                # If only one side found, allow it (we'll save whatever we have)
                s_start_ts = start_s.timestamp if start_s is not None else None
                s_end_ts = end_s.timestamp if end_s is not None else None

                # attempt to save screenshots
                try:
                    start_path = self.save_worker.save_screenshot(start_s, force_save=True) if start_s is not None else None
                    end_path = self.save_worker.save_screenshot(end_s, force_save=True) if end_s is not None else None
                except Exception as e:
                    print(f"AggregationHandler (simplified): error saving images: {e}")
                    continue

                print(f"Start Reason: {start_reason}, End Reason: {end_reason}")
                action_record = {
                    "created_at": time.time(),
                    "action_type": "cluster",
                    "cluster_type": group,
                    "cluster_start": start,
                    "cluster_end": end,
                    "cluster_event_count": len(cluster_events),
                    "events": cluster_events,
                    "start_screenshot_time": s_start_ts,
                    "end_screenshot_time": s_end_ts,
                    "start_path": start_path,
                    "end_path": end_path,
                    "start_selection": start_reason,
                    "end_selection": end_reason,
                }

                try:
                    with open(self._agg_log, "a") as f:
                        json.dump(action_record, f, default=str)
                        f.write("\n")
                    self._saved_pairs.add(cluster_key)
                    saved += 1
                    self._log("Saved cluster", action_record["cluster_start"], action_record["cluster_end"], "events:", action_record["cluster_event_count"])
                except Exception as e:
                    print(f"AggregationHandler (simplified): failed to write aggregated action: {e}")

        if self.debug:
            total_clusters = sum(len(self._cluster_event_times(ev_list, self.MERGE_ACTION_GAP)) for ev_list in group_map.values())
            self._log(f"Simplified pipeline processed {total_clusters} clusters, saved {saved} aggregated clusters.")
