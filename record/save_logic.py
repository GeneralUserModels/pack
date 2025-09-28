import threading
import time
from pathlib import Path
from typing import Deque, List, Dict, Any, Tuple, Optional
import multiprocessing
import shutil
import os

RESCAN_INTERVAL = 0.05        # main loop poll interval (s)
WAIT_FOR_FULL_PATH = 1.0     # how long to wait for the full-res buffer file to appear (s)
COPY_PROCESS_JOIN_TIMEOUT = None  # don't join child copy processes; leave them running (non-blocking)

# clustering / burst gaps (from your analysis)
CLICK_PAIR_MAX_GAP = 1.0
BURST_MAX_GAP_MOVE = 0.5
BURST_MAX_GAP_SCROLL = 0.7
BURST_MAX_GAP_KEYBOARD = 0.3

SSIM_AVERAGE_WINDOW = 3.0    # seconds to average SSIM for baseline
MIN_SSIM_BEFORE = 3          # require at least this many SSIM samples in the pre-window
MIN_SSIM_AFTER = 1           # require at least this many SSIM samples in the post-window

# SSIM difference thresholds (per action-type)
SSIM_DIFF_CLICK_AFTER = 0.05
SSIM_DIFF_MOVE_AFTER = 0.0
SSIM_DIFF_SCROLL_AFTER = -0.10
SSIM_DIFF_KEYBOARD_AFTER = 0.02

TIME_MAX_PAD_DEFAULT = 3.0
TIME_MIN_PAD = 0.05
# ------------------------------------------------

# Top-level helper for picklability (multiprocessing spawn on macOS)


def _async_copy_file(src: str, dst: str):
    """Copy file from src -> dst in a separate process. Performs an atomic move via tempfile+rename."""
    try:
        src_p = Path(src)
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        # Use a tmp filename in same directory to make rename atomic
        tmp = dst_p.with_suffix(dst_p.suffix + ".tmp")
        # copy2 preserves metadata and is better than copyfile
        shutil.copy2(src_p, tmp)
        os.replace(str(tmp), str(dst_p))  # atomic on POSIX
    except Exception as e:
        # We intentionally don't raise; just print to help debugging
        print(f"[async_copy_file] error copying {src} -> {dst}: {e}")


class SaveDeciderWorker:
    """
    Advanced save-decider that requires full-resolution images only.
    If copying takes long, the copy happens in a separate process.
    """

    def __init__(self, screenshot_manager, event_queue, screenshot_dir: Path, stop_event: threading.Event):
        self.screenshot_manager = screenshot_manager
        self.event_queue = event_queue
        self.screenshot_dir = Path(screenshot_dir)
        self.stop_event = stop_event

        self._thread = None
        self._processed_action_keys = set()
        self._saved_event_ids = set()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------- snapshots ----------
    def _snapshot_ssim_list(self) -> List[Dict[str, Any]]:
        try:
            ssim_deque: Deque = self.screenshot_manager.ssim_buffer
            entries = list(ssim_deque)
        except Exception:
            entries = []
        normalized = []
        for e in entries:
            try:
                ts = float(e.get("unix_timestamp", e.get("timestamp", 0.0)))
                ssim_val = float(e.get("ssim_similarity", e.get("ssim", 1.0)))
                normalized.append({
                    "unix_timestamp": ts,
                    "ssim": ssim_val,
                    "slot_index": e.get("slot_index"),
                    "monitor_id": e.get("monitor_id"),
                    "raw": e
                })
            except Exception:
                continue
        normalized.sort(key=lambda x: x["unix_timestamp"])
        return normalized

    def _snapshot_event_list(self) -> List[Dict[str, Any]]:
        try:
            tuples = self.event_queue.get_recent_events(0.0, None)
        except Exception:
            tuples = []
        evs = []
        for ts, raw in tuples:
            try:
                evs.append({
                    "time": float(ts),
                    "event_type": raw.event_type,
                    "details": raw.details if isinstance(raw.details, dict) else {},
                    "cursor_pos": raw.cursor_pos if raw.cursor_pos is not None else [],
                    "monitor_id": getattr(raw, "monitor_id", None),
                    "raw": raw
                })
            except Exception:
                continue
        evs.sort(key=lambda x: x["time"])
        return evs

    def _snapshot_screenshots_meta(self, ssim_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        snaps = []
        for s in ssim_list:
            snaps.append({
                "timestamp": s["unix_timestamp"],
                "ssim": s["ssim"],
                "monitor_id": s.get("monitor_id"),
                "slot_index": s.get("slot_index"),
                "raw": s.get("raw")
            })
        return snaps

    # ---------- ssim utilities ----------
    def _calculate_ssim_average(self, target_time: float, screenshots: List[Dict[str, Any]], window: float = SSIM_AVERAGE_WINDOW) -> float:
        relevant = [s for s in screenshots if target_time - window <= s["timestamp"] < target_time]
        if relevant:
            return sum(s["ssim"] for s in relevant) / len(relevant)
        before = [s for s in screenshots if s["timestamp"] < target_time]
        if before:
            return before[-1]["ssim"]
        return 1.0

    def _count_ssim_in_range(self, start: float, end: float, screenshots: List[Dict[str, Any]]) -> int:
        return sum(1 for s in screenshots if start <= s["timestamp"] <= end)

    def _find_screenshot_before(self, target_time: float, screenshots: List[Dict[str, Any]],
                                ssim_diff_threshold: float = 0.0,
                                time_max_pad: float = TIME_MAX_PAD_DEFAULT,
                                time_min_pad: float = TIME_MIN_PAD) -> Optional[Dict[str, Any]]:
        if not screenshots:
            return None
        ssim_avg = self._calculate_ssim_average(target_time, screenshots)
        candidates = [s for s in screenshots if s["timestamp"] < target_time - time_min_pad]
        if not candidates:
            return None
        for i in range(len(candidates) - 1, -1, -1):
            s = candidates[i]
            pre_start = s["timestamp"] - SSIM_AVERAGE_WINDOW
            pre_count = self._count_ssim_in_range(pre_start, s["timestamp"], screenshots)
            if pre_count < MIN_SSIM_BEFORE:
                continue
            ssim_diff = s["ssim"] - ssim_avg
            if ssim_diff >= ssim_diff_threshold:
                return s
            if target_time - s["timestamp"] <= time_max_pad:
                return s
        return candidates[-1] if candidates else None

    def _find_screenshot_after(self, target_time: float, screenshots: List[Dict[str, Any]],
                               ssim_diff_threshold: float = 0.0,
                               time_max_pad: float = TIME_MAX_PAD_DEFAULT,
                               time_min_pad: float = TIME_MIN_PAD) -> Optional[Dict[str, Any]]:
        if not screenshots:
            return None
        ssim_avg = self._calculate_ssim_average(target_time, screenshots)
        candidates = [s for s in screenshots if s["timestamp"] > target_time + time_min_pad]
        if not candidates:
            return None
        for s in candidates:
            post_start = s["timestamp"]
            post_end = s["timestamp"] + SSIM_AVERAGE_WINDOW
            post_count = self._count_ssim_in_range(post_start, post_end, screenshots)
            if post_count < MIN_SSIM_AFTER:
                continue
            ssim_diff = s["ssim"] - ssim_avg
            if ssim_diff >= ssim_diff_threshold:
                return s
            if s["timestamp"] - target_time <= time_max_pad:
                return s
        return candidates[0] if candidates else None

    # ---------- event clustering ----------
    def _cluster_event_times(self, events: List[Dict[str, Any]], max_gap: float) -> List[Tuple[float, float, List[Dict[str, Any]]]]:
        if not events:
            return []
        evs = sorted(events, key=lambda e: e["time"])
        clusters = []
        cur_start = evs[0]["time"]
        cur_end = evs[0]["time"]
        cur_list = [evs[0]]
        for e in evs[1:]:
            if e["time"] - cur_end <= max_gap:
                cur_end = e["time"]
                cur_list.append(e)
            else:
                clusters.append((cur_start, cur_end, cur_list))
                cur_start = e["time"]
                cur_end = e["time"]
                cur_list = [e]
        clusters.append((cur_start, cur_end, cur_list))
        return clusters

    def _find_click_pairs(self, click_downs: List[Dict[str, Any]], click_ups: List[Dict[str, Any]]) -> List[Tuple[float, float, Dict[str, Any], Dict[str, Any]]]:
        pairs = []
        for d in click_downs:
            ups = [u for u in click_ups if u["time"] >= d["time"] and u.get("monitor_id") == d.get("monitor_id") and u["time"] - d["time"] <= CLICK_PAIR_MAX_GAP]
            if ups:
                u = ups[0]
                pairs.append((d["time"], u["time"], d, u))
        return pairs

    def _generate_actions(self, screenshots: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        actions = []
        move_events = [e for e in events if e["event_type"] == "mouse_move"]
        scroll_events = [e for e in events if e["event_type"] == "mouse_scroll"]
        kbd_events = [e for e in events if e["event_type"] in ("keyboard_press", "keyboard_release")]
        click_downs = [e for e in events if e["event_type"] == "mouse_down"]
        click_ups = [e for e in events if e["event_type"] == "mouse_up"]

        click_pairs = self._find_click_pairs(click_downs, click_ups)
        for start_time, end_time, d, u in click_pairs:
            start_s = self._find_screenshot_before(start_time, screenshots, ssim_diff_threshold=0.0, time_max_pad=3)
            end_s = self._find_screenshot_after(end_time, screenshots, ssim_diff_threshold=SSIM_DIFF_CLICK_AFTER, time_max_pad=3)
            actions.append({
                "type": "click",
                "start": start_time,
                "end": end_time,
                "start_event_raw": d["raw"],
                "end_event_raw": u["raw"],
                "start_screenshot": start_s,
                "end_screenshot": end_s
            })

        move_clusters = self._cluster_event_times(move_events, BURST_MAX_GAP_MOVE)
        for start_time, end_time, ev_list in move_clusters:
            start_s = self._find_screenshot_before(start_time, screenshots, ssim_diff_threshold=0.0, time_max_pad=0)
            end_s = self._find_screenshot_after(end_time, screenshots, ssim_diff_threshold=SSIM_DIFF_MOVE_AFTER, time_max_pad=0)
            actions.append({
                "type": "move",
                "start": start_time,
                "end": end_time,
                "events": ev_list,
                "start_screenshot": start_s,
                "end_screenshot": end_s,
                "start_event_raw": ev_list[0]["raw"] if ev_list else None
            })

        scroll_clusters = self._cluster_event_times(scroll_events, BURST_MAX_GAP_SCROLL)
        for start_time, end_time, ev_list in scroll_clusters:
            start_s = self._find_screenshot_before(start_time, screenshots, ssim_diff_threshold=0.0, time_max_pad=2)
            end_s = self._find_screenshot_after(end_time, screenshots, ssim_diff_threshold=SSIM_DIFF_SCROLL_AFTER, time_max_pad=2)
            actions.append({
                "type": "scroll",
                "start": start_time,
                "end": end_time,
                "events": ev_list,
                "start_screenshot": start_s,
                "end_screenshot": end_s,
                "start_event_raw": ev_list[0]["raw"] if ev_list else None
            })

        kbd_clusters = self._cluster_event_times(kbd_events, BURST_MAX_GAP_KEYBOARD)
        for start_time, end_time, ev_list in kbd_clusters:
            start_s = self._find_screenshot_before(start_time, screenshots, ssim_diff_threshold=0.0, time_max_pad=3)
            end_s = self._find_screenshot_after(end_time, screenshots, ssim_diff_threshold=SSIM_DIFF_KEYBOARD_AFTER, time_max_pad=3)
            actions.append({
                "type": "keyboard",
                "start": start_time,
                "end": end_time,
                "events": ev_list,
                "start_screenshot": start_s,
                "end_screenshot": end_s,
                "start_event_raw": ev_list[0]["raw"] if ev_list else None
            })

        actions.sort(key=lambda a: a["start"])
        return actions

    # ---------- full-res-only saving ----------
    def _wait_for_fullres_file(self, ts: float, timeout: float = WAIT_FOR_FULL_PATH) -> Optional[str]:
        """
        Try immediate lookup, then poll for up to `timeout` seconds for a saved buffer file that matches ts.
        Uses screenshot_manager.find_buffer_fullres(ts) if available.
        """
        # immediate attempt
        try:
            fp = self.screenshot_manager.find_buffer_fullres(ts)
            if fp and Path(fp).exists():
                return fp
        except Exception:
            pass

        # poll
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                fp = self.screenshot_manager.find_buffer_fullres(ts)
                if fp and Path(fp).exists():
                    return fp
            except Exception:
                pass
            time.sleep(0.02)
        return None

    def _dispatch_copy_process(self, src: str, dst: str):
        """Start a separate process to copy src -> dst and return immediately."""
        try:
            p = multiprocessing.Process(target=_async_copy_file, args=(str(src), str(dst)), daemon=True)
            p.start()
            # we deliberately do not join; child will run independently
            return p
        except Exception as e:
            print(f"[AdvancedSaveDecider] failed to spawn copy process: {e}")
            return None

    def _save_screenshot_by_ssim_entry(self, ssim_entry: Dict[str, Any], associated_raw_event):
        """
        Strict full-res-only save:
         - If screenshot_manager.save_all_buffer is True:
             wait briefly for a buffer file to appear (stable) and then spawn a non-blocking copy process to screenshot_dir.
         - If save_all_buffer is False:
             request an on-demand export from screenshot_manager (which should write directly to out_path).
         - Do NOT fallback to thumbnails.
        Returns path string on success, or None on failure.
        """
        if not ssim_entry:
            return None

        ts = float(ssim_entry.get("unix_timestamp", ssim_entry.get("timestamp", time.time())))
        target = self.screenshot_dir / f"saved_{ts:.6f}.jpg"

        try:
            # Case A: continuous buffer is enabled -> find an existing buffer file and copy it
            if getattr(self.screenshot_manager, "save_all_buffer", False):
                # Wait for a stable full-res buffer file (this method should ensure stability)
                full_path = self._wait_for_fullres_file(ts, timeout=WAIT_FOR_FULL_PATH)
                if not full_path:
                    print(f"[AdvancedSaveDecider] full-res buffer file for ts={ts:.6f} not found within {WAIT_FOR_FULL_PATH}s; skipping.")
                    return None

                # If source equals target (rare), just mark saved and return
                src_path = Path(full_path)
                if src_path.resolve() == target.resolve():
                    try:
                        if associated_raw_event is not None:
                            rid = getattr(associated_raw_event, "id", None)
                            if rid is not None:
                                self._saved_event_ids.add(rid)
                    except Exception:
                        pass
                    return str(target)

                # Dispatch non-blocking copy process
                proc = self._dispatch_copy_process(str(src_path), str(target))
                if proc is None:
                    print(f"[AdvancedSaveDecider] failed to spawn copy process for {src_path} -> {target}")
                    return None

                # Mark associated event as saved (preemptive)
                try:
                    if associated_raw_event is not None:
                        rid = getattr(associated_raw_event, "id", None)
                        if rid is not None:
                            self._saved_event_ids.add(rid)
                except Exception:
                    pass

                return str(target)

            # Case B: continuous buffer disabled -> request on-demand export from capture process
            else:
                # The request_fullres_save helper should ask the capture process to write `target`
                # and return True on success (within timeout).
                try:
                    ok = False
                    # request_fullres_save may block (poll) internally up to the given timeout
                    if hasattr(self.screenshot_manager, "request_fullres_save"):
                        ok = self.screenshot_manager.request_fullres_save(float(ts), str(target), timeout=WAIT_FOR_FULL_PATH)
                    else:
                        # Fallback: if helper not present, we cannot export on-demand
                        print("[AdvancedSaveDecider] screenshot_manager.request_fullres_save is not implemented; cannot export on-demand.")
                        return None

                    if not ok:
                        print(f"[AdvancedSaveDecider] on-demand export for ts={ts:.6f} timed out or failed")
                        return None

                    # On success, mark associated event as saved and return path
                    try:
                        if associated_raw_event is not None:
                            rid = getattr(associated_raw_event, "id", None)
                            if rid is not None:
                                self._saved_event_ids.add(rid)
                    except Exception:
                        pass

                    return str(target)
                except Exception as e:
                    print(f"[AdvancedSaveDecider] error requesting on-demand export for ts={ts:.6f}: {e}")
                    return None

        except Exception as e:
            print(f"[AdvancedSaveDecider] unexpected error while saving ts={ts:.6f}: {e}")
            return None

    # ---------- main loop ----------
    def _run(self):
        while not self.stop_event.is_set():
            try:
                ssim_list = self._snapshot_ssim_list()
                screenshots_meta = self._snapshot_screenshots_meta(ssim_list)
                events = self._snapshot_event_list()

                if not ssim_list or not events:
                    time.sleep(RESCAN_INTERVAL)
                    continue

                actions = self._generate_actions(screenshots_meta, events)

                for act in actions:
                    key = (act["type"], float(act["start"]), float(act["end"]))
                    if key in self._processed_action_keys:
                        continue

                    start_s = act.get("start_screenshot")
                    end_s = act.get("end_screenshot")

                    ok_start = True
                    if start_s:
                        pre_count = self._count_ssim_in_range(start_s["timestamp"] - SSIM_AVERAGE_WINDOW, start_s["timestamp"], screenshots_meta)
                        if pre_count < MIN_SSIM_BEFORE:
                            ok_start = False

                    ok_end = True
                    if end_s:
                        post_count = self._count_ssim_in_range(end_s["timestamp"], end_s["timestamp"] + SSIM_AVERAGE_WINDOW, screenshots_meta)
                        if post_count < MIN_SSIM_AFTER:
                            ok_end = False

                    if not ok_start and not ok_end:
                        # wait for more SSIMs to accumulate
                        continue

                    associated_raw = act.get("start_event_raw") or act.get("end_event_raw")

                    if start_s and getattr(associated_raw, "id", None) not in self._saved_event_ids and ok_start:
                        p = self._save_screenshot_by_ssim_entry(start_s["raw"], associated_raw)
                        if p:
                            print(f"[AdvancedSaveDecider] dispatched start screenshot copy for {act['type']} -> {p}")

                    if end_s and getattr(associated_raw, "id", None) not in self._saved_event_ids and ok_end:
                        p = self._save_screenshot_by_ssim_entry(end_s["raw"], associated_raw)
                        if p:
                            print(f"[AdvancedSaveDecider] dispatched end screenshot copy for {act['type']} -> {p}")

                    self._processed_action_keys.add(key)

                time.sleep(RESCAN_INTERVAL)
            except Exception as e:
                print(f"[AdvancedSaveDecider] error: {e}")
                time.sleep(0.1)
