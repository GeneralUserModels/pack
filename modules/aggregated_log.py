from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


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
    def from_raw_log_events(self, raw_log_events: RawLogEvents):
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

    def __repr__(self) -> str:
        return f"Log(start_timestamp={self.start_timestamp}, end_timestamp={self.end_timestamp}, keys_pressed={''.join(self.keys_pressed)})"
