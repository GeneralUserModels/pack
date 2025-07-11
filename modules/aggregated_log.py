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

    def _mm_ss_difference(self, start: str, end: str) -> str:
        """Calculate time difference and format as MM:SS or HH:MM:SS for longer durations"""
        def parse_ts(ts: str) -> datetime:
            if '_' in ts:
                date_part, time_part = ts.split('_', 1)

                if '.' in time_part:
                    time_part = time_part.split('.')[0]

                time_components = time_part.split('-')
                if len(time_components) >= 3:
                    time_part = f"{time_components[0]}:{time_components[1]}:{time_components[2]}"

                datetime_str = f"{date_part}T{time_part}"
                return datetime.fromisoformat(datetime_str)
            else:
                return datetime.fromisoformat(ts)

        try:
            start_dt = parse_ts(start)
            end_dt = parse_ts(end)
        except ValueError as e:
            return f"ERROR: Invalid timestamp format - {str(e)}"

        if end_dt < start_dt:
            delta = start_dt - end_dt
            negative = True
        else:
            delta = end_dt - start_dt
            negative = False

        total_seconds = int(delta.total_seconds())

        if total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return f"-{formatted}" if negative else formatted

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

    def to_prompt(self, start_timestamp):
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
                actions.append(f"Mouse clicked {button} at {cursor_pos}.")

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
                actions.append(f"Mouse moved to {cursor_pos}.")

        # Add any remaining keys that weren't followed by a mouse event
        if keys_pressed:
            actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")

        start_time_diff = self._mm_ss_difference(start_timestamp, self.start_timestamp)
        end_time_diff = self._mm_ss_difference(start_timestamp, self.end_timestamp)

        action_list = '\n'.join([f"  {action}" for action in actions])

        prompt = f"""
{start_time_diff} - {end_time_diff} Events. Starting:
Cursor position: {self.start_cursor_pos} to {self.end_cursor_pos if self.end_cursor_pos else 'N/A'}
Actions:
{action_list}
"""
        return prompt

    def __repr__(self) -> str:
        keys_str = ''.join([str(k) for k in (self.keys_pressed or [])])
        return f"Log(start_timestamp={self.start_timestamp}, end_timestamp={self.end_timestamp}, keys_pressed={keys_str})"
