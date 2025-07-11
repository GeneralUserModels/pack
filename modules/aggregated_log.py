from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from datetime import datetime


from modules.raw_log import RawLogEvents


@dataclass
class AggregatedLog:
    start_timestamp: str
    end_timestamp: str
    monitor: Dict[str, int]
    start_screenshot_path: Optional[str] = None
    end_screenshot_path: Optional[str] = None
    start_cursor_pos: Optional[Tuple[int, int]] = None
    end_cursor_pos: Optional[Tuple[int, int]] = None
    click_positions: Optional[List[Tuple[int, int]]] = None
    scroll_directions: Optional[List[Tuple[int, int]]] = None
    keys_pressed: Optional[List[List[str]]] = None
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        # return asdict(self)
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "monitor": self.monitor,
            "start_screenshot_path": self.start_screenshot_path,
            "end_screenshot_path": self.end_screenshot_path,
            "start_cursor_pos": self.start_cursor_pos,
            "end_cursor_pos": self.end_cursor_pos,
            "events": self.events,
        }

    @classmethod
    def from_raw_log_events(cls, raw_log_events: RawLogEvents):
        raw_log_events.sort()
        cursor_pos = [ev.cursor_pos for ev in raw_log_events.events if ev.cursor_pos]
        return AggregatedLog(
            start_timestamp=raw_log_events[0].timestamp,
            end_timestamp=raw_log_events[-1].timestamp,
            monitor=raw_log_events[0].monitor,
            start_screenshot_path=raw_log_events[0].screenshot_path,
            end_screenshot_path=raw_log_events[-1].screenshot_path,
            start_cursor_pos=cursor_pos[0] if cursor_pos else None,
            end_cursor_pos=cursor_pos[-1] if cursor_pos else None,
            click_positions=[ev.cursor_pos for ev in raw_log_events.events if ev.event_type == 'mouse_click' and ev.cursor_pos],
            scroll_directions=[ev.details.get('direction') for ev in raw_log_events.events if ev.event_type == 'mouse_scroll' and 'direction' in ev.details],
            keys_pressed=[ev.details.get('key') for ev in raw_log_events.events if ev.event_type == "keyboard_press" and 'key' in ev.details],
            events=raw_log_events.format_to_event_list()
        )

    @classmethod
    def from_dict(cls, data: dict):
        return AggregatedLog(
            start_timestamp=data.get("start_timestamp"),
            end_timestamp=data.get("end_timestamp"),
            monitor=data.get("monitor", {}),
            start_screenshot_path=data.get("start_screenshot_path"),
            end_screenshot_path=data.get("end_screenshot_path"),
            start_cursor_pos=tuple(data.get("start_cursor_pos", (None, None))),
            end_cursor_pos=tuple(data.get("end_cursor_pos", (None, None))),
            click_positions=[tuple(pos) for pos in data.get("click_positions", [])],
            scroll_directions=data.get("scroll_directions", []),
            keys_pressed=data.get("keys_pressed", []),
            events=data.get("events", [])
        )

    def _convert_pos_to_gemini_relative(self, pos):
        if not self.monitor:
            return pos
        width = self.monitor.get('width', 1)
        height = self.monitor.get('height', 1)
        if isinstance(pos, tuple) and len(pos) == 2:
            x, y = pos
            x -= self.monitor.get('left', 0)
            y -= self.monitor.get('top', 0)
            relative_x = round(x / width, 2) * 1_000
            relative_y = round(y / height, 2) * 1_000
            return (relative_x, relative_y)

    @staticmethod
    def _convert_scroll_direction(scroll_data):
        """Convert scroll data to human-readable direction"""
        if isinstance(scroll_data, dict):
            dx = scroll_data.get('dx', 0)
            dy = scroll_data.get('dy', 0)
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

        if directions:
            direction_str = " ".join(directions)
        else:
            direction_str = "no scroll"
        return direction_str

    def to_prompt(self, frame, fps: int = 1):
        actions = []
        keys_pressed = []

        for event in self.events:
            event_type = event.get("event_type")

            if event_type == 'keyboard_press':
                key = event.get('details', {}).get('key', 'Key.unknown')
                key = key.replace("Key.", "") if key.startswith("Key.") else key
                keys_pressed.append(key)

            elif event_type == 'mouse_down':
                if keys_pressed:
                    actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()
                button = event.get('details', {}).get('button', 'Button.unknown')
                button = button.replace('Button.', '') if button.startswith('Button.') else button
                cursor_pos = event.get('cursor_pos', 'unknown position')
                actions.append(f"Mouse clicked {button} at {self._convert_pos_to_gemini_relative(cursor_pos)}.")

            elif event_type == 'mouse_scroll':
                if keys_pressed:
                    actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()
                scroll_data = event.get('details', {}).get('scroll', {})
                direction = self._convert_scroll_direction(scroll_data)
                actions.append(f"Scrolled {direction}")

            elif event_type == 'mouse_move':
                if keys_pressed:
                    actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()
                cursor_pos = event.get('cursor_pos', 'unknown position')
                actions.append(f"Mouse moved to {self._convert_pos_to_gemini_relative(cursor_pos)}.")

        if keys_pressed:
            actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")

        start_time_seconds = frame / fps
        start_time_mm_ss = datetime.utcfromtimestamp(start_time_seconds).strftime('%M:%S')
        end_time_seconds = (frame + 2) / fps
        end_time_mm_ss = datetime.utcfromtimestamp(end_time_seconds).strftime('%M:%S')

        action_list = '\n'.join([f"  {action}" for action in actions])

        prompt = f"""
        {start_time_mm_ss} - {end_time_mm_ss} Events. Starting:
        Cursor position: {self._convert_pos_to_gemini_relative(self.start_cursor_pos)} to {self._convert_pos_to_gemini_relative(self.end_cursor_pos) if self.end_cursor_pos else 'N/A'}
        Actions:
        {action_list}
        """
        return prompt

    def __repr__(self) -> str:
        keys_str = ''.join([str(k) for k in (self.keys_pressed or [])])
        return f"Log(start_timestamp={self.start_timestamp}, end_timestamp={self.end_timestamp}, keys_pressed={keys_str})"
