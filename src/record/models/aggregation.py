from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class AggregationConfig:
    gap_threshold: float  # Max time gap between consecutive events (seconds)
    total_threshold: float  # Max time span from first to last event (seconds)


@dataclass
class AggregationRequest:
    """Represents a request for a screenshot at a specific time."""
    timestamp: float
    end_timestamp: Optional[float]
    reason: str  # e.g., "key_start", "mouse_move_end"
    event_type: str  # e.g., "key", "move"
    request_state: str  # burst start, mid or end
    screenshot: Optional[Any] = None  # Screenshot object from ImageQueue
    screenshot_path: Optional[str] = None  # Path to saved screenshot
    screenshot_timestamp: Optional[float] = None  # Timestamp of the screenshot
    end_screenshot_timestamp: Optional[float] = None  # Timestamp of end screenshot
    monitor: Optional[dict] = None  # Monitor info at the time of screenshot
    burst_id: Optional[int] = None  # ID of the burst this request belongs to
    scale_factor: float = 1.0  # Scaling factor of the screenshot


@dataclass
class ProcessedAggregation:
    """Represents an aggregation with matched screenshot and events."""
    request: AggregationRequest
    events: List[dict]

    @property
    def screenshot(self) -> Optional[Any]:
        """Get screenshot from request."""
        return self.request.screenshot

    @property
    def screenshot_path(self) -> Optional[str]:
        """Get screenshot path from request."""
        return self.request.screenshot_path

    def to_dict(self):
        request = self.request.__dict__.copy()
        request.pop('screenshot', None)
        return {
            "request": request,
            "events": self.events
        }

    def to_prompt(self, timestamp: str) -> str:
        """Convert aggregation data to prompt string."""
        events = self._compress_events(self.events)

        actions = []
        keys_pressed = []

        default_monitor = (self.events[0].get("monitor", {}) if self.events else {})

        start_cursor_pos = None
        end_cursor_pos = None

        for event in events:
            cursor_pos = event.get("cursor_position")
            if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) == 2 and \
               isinstance(cursor_pos[0], (list, tuple)) and isinstance(cursor_pos[1], (list, tuple)):
                if start_cursor_pos is None:
                    start_cursor_pos = cursor_pos[0]
                end_cursor_pos = cursor_pos[1]
            else:
                if cursor_pos:
                    if start_cursor_pos is None:
                        start_cursor_pos = cursor_pos
                    end_cursor_pos = cursor_pos

        for event in events:
            event_type = event.get("event_type")
            monitor = event.get("monitor", default_monitor)

            if event_type == "key_press":
                details = event.get("details", {})
                key = details.get("key", "unknown")
                key = key.replace("Key.", "") if key and isinstance(key, str) and key.startswith("Key.") else key

                if key:
                    focused = details.get("focused_element", {})

                    key_str = key
                    if focused:
                        role = focused.get("AXRole", "")
                        title = focused.get("AXTitle", "")
                        if title and role:
                            key_str = f"{key} (in {role}: '{title}')"
                        elif title:
                            key_str = f"{key} (in '{title}')"
                        elif role:
                            key_str = f"{key} (in {role})"

                    keys_pressed.append(key_str)

            elif event_type == "mouse_click":
                if keys_pressed:
                    actions.append(
                        f"Key pressed: {keys_pressed[0]}"
                        if len(keys_pressed) == 1
                        else f"Keys pressed: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                details = event.get("details", {})
                button = details.get("button", "unknown")
                button = button.replace("Button.", "") if button and isinstance(button, str) and button.startswith("Button.") else button
                cursor_pos = event.get("cursor_position", "unknown position")
                double_click = details.get("double_click", False)

                accessibility = details.get("accessibility", {})

                mouse_str = f"Mouse clicked {button} at {self._convert_pos_to_gemini_relative(cursor_pos, monitor)}"
                if double_click:
                    mouse_str += " (double click)"

                if accessibility:
                    role = accessibility.get("AXRole", "")
                    title = accessibility.get("AXTitle", "")
                    desc = accessibility.get("AXDescription", "")

                    if title and role:
                        mouse_str += f" on {role}: '{title}'"
                    elif title:
                        mouse_str += f" on '{title}'"
                    elif desc and role:
                        mouse_str += f" on {role}: '{desc}'"
                    elif role:
                        mouse_str += f" on {role}"

                actions.append(mouse_str)

            elif event_type == "mouse_scroll":
                if keys_pressed:
                    actions.append(
                        f"Key pressed: {keys_pressed[0]}"
                        if len(keys_pressed) == 1
                        else f"Keys pressed: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                details = event.get("details", {})
                direction = event.get("_direction") or self._convert_scroll_direction(details)
                accessibility = details.get("accessibility", {})

                scroll_str = f"Scrolled {direction}"

                if accessibility:
                    role = accessibility.get("AXRole", "")
                    title = accessibility.get("AXTitle", "")

                    if title and role:
                        scroll_str += f" in {role}: '{title}'"
                    elif title:
                        scroll_str += f" in '{title}'"
                    elif role:
                        scroll_str += f" in {role}"

                actions.append(scroll_str)

            elif event_type == "mouse_move":
                if keys_pressed:
                    actions.append(
                        f"Key pressed: {keys_pressed[0]}"
                        if len(keys_pressed) == 1
                        else f"Keys pressed: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                cursor_pos = event.get("cursor_position", "unknown position")

                if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) == 2 and \
                   isinstance(cursor_pos[0], (list, tuple)) and isinstance(cursor_pos[1], (list, tuple)):
                    start_pos, last_pos = cursor_pos[0], cursor_pos[1]
                    actions.append(
                        f"Mouse moved from {self._convert_pos_to_gemini_relative(start_pos, monitor)} to {self._convert_pos_to_gemini_relative(last_pos, monitor)}"
                    )
                else:
                    actions.append(f"Mouse moved to {self._convert_pos_to_gemini_relative(cursor_pos, monitor)}")

        if keys_pressed:
            actions.append(
                f"Key pressed: {keys_pressed[0]}"
                if len(keys_pressed) == 1
                else f"Keys pressed: {'|'.join(keys_pressed)}"
            )

        action_list = None
        if actions:
            action_list = '\t' + '\n\t'.join([f"{action}" for action in actions])

        prompt = f"""
Event at {timestamp}:
Cursor positions: {self._convert_pos_to_gemini_relative(start_cursor_pos, default_monitor)} to {self._convert_pos_to_gemini_relative(end_cursor_pos, default_monitor) if end_cursor_pos else 'N/A'}
Actions:
{''.join(action_list) if action_list else 'No actions recorded.'}
        """
        return prompt

    def _compress_events(self, events: List[dict]) -> List[dict]:
        """
        Compress consecutive mouse_move and mouse_scroll events:
         - mouse_move: collapse consecutive moves into a single event with cursor_position = (first_pos, last_pos)
         - mouse_scroll: collapse consecutive scrolls by direction changes, keeping one event per direction change
        Other events are left unchanged and order is preserved.
        """
        compressed = []
        i = 0
        n = len(events)

        while i < n:
            e = events[i]
            et = e.get("event_type")

            if et == "mouse_move":
                start_pos = e.get("cursor_position")
                last_pos = start_pos
                j = i + 1
                while j < n and events[j].get("event_type") == "mouse_move":
                    candidate_pos = events[j].get("cursor_position", last_pos)
                    if candidate_pos is not None:
                        last_pos = candidate_pos
                    j += 1
                monitor = e.get("monitor", {})
                compressed.append({
                    "event_type": "mouse_move",
                    "cursor_position": (start_pos, last_pos),
                    "monitor": monitor
                })
                i = j

            elif et == "mouse_scroll":
                j = i
                last_dir = None
                monitor = e.get("monitor", {})
                while j < n and events[j].get("event_type") == "mouse_scroll":
                    details = events[j].get("details", {})
                    dir_ = self._convert_scroll_direction(details)
                    if dir_ != last_dir:
                        compressed.append({
                            "event_type": "mouse_scroll",
                            "details": details,
                            "_direction": dir_,
                            "monitor": events[j].get("monitor", monitor)
                        })
                        last_dir = dir_
                    j += 1
                i = j

            else:
                compressed.append(e)
                i += 1

        return compressed

    def _convert_pos_to_gemini_relative(self, pos, monitor):
        """Convert absolute screen coordinates to relative coordinates (0-1000 scale)."""
        if not monitor or not pos:
            return pos

        width = monitor.get("width", 1)
        height = monitor.get("height", 1)

        if isinstance(pos, (tuple, list)) and len(pos) == 2 and isinstance(pos[0], (int, float)):
            x, y = pos
            x -= monitor.get("left", 0)
            y -= monitor.get("top", 0)
            relative_x = round((x / width) * 1_000)
            relative_y = round((y / height) * 1_000)
            return (relative_x, relative_y)

        if isinstance(pos, (tuple, list)) and len(pos) == 2 and \
           isinstance(pos[0], (tuple, list)) and isinstance(pos[1], (tuple, list)):
            start_pos, end_pos = pos[0], pos[1]
            sx, sy = start_pos
            ex, ey = end_pos
            sx -= monitor.get("left", 0)
            sy -= monitor.get("top", 0)
            ex -= monitor.get("left", 0)
            ey -= monitor.get("top", 0)
            srx = round((sx / width) * 1_000)
            sry = round((sy / height) * 1_000)
            erx = round((ex / width) * 1_000)
            ery = round((ey / height) * 1_000)
            return ((srx, sry), (erx, ery))

        return pos

    @staticmethod
    def _convert_scroll_direction(scroll_data):
        """Convert scroll data to human-readable direction."""
        if isinstance(scroll_data, dict):
            dx = scroll_data.get("dx", 0)
            dy = scroll_data.get("dy", 0)
        elif isinstance(scroll_data, (list, tuple)) and len(scroll_data) >= 2:
            dx, dy = scroll_data[0], scroll_data[1]
        else:
            return "no scroll"

        directions = []
        if dy > 0:
            directions.append("up")
        elif dy < 0:
            directions.append("down")
        if dx > 0:
            directions.append("right")
        elif dx < 0:
            directions.append("left")

        direction_str = " ".join(directions) if directions else "no scroll"
        return direction_str
