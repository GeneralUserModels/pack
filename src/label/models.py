from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from PIL import Image


@dataclass
class ImagePath:
    path: Path
    fallback_dir: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def resolve(self) -> Path:
        if self.path.exists():
            return self.path

        if self.fallback_dir:
            relative = self.fallback_dir / "screenshots" / self.path.name
            if relative.exists():
                return relative

        raise FileNotFoundError(f"Image not found: {self.path}")

    def load(self) -> Image.Image:
        return Image.open(self.resolve()).convert('RGB')


@dataclass
class VideoPath:
    path: Path

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def exists(self) -> bool:
        return self.path.exists()

    def resolve(self) -> Path:
        if not self.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        return self.path


@dataclass
class EventDetails:
    data: Dict[str, Any]

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    @property
    def button(self) -> str:
        return self.data.get('button', 'unknown').replace('Button.', '')

    @property
    def key(self) -> str:
        return self.data.get('key', 'unknown').replace('Key.', '')

    @property
    def is_double_click(self) -> bool:
        return self.data.get('double_click', False)

    def to_dict(self) -> Dict[str, Any]:
        return self.data


@dataclass
class Event:
    event_type: str
    timestamp: float
    cursor_position: List[int]
    details: EventDetails
    monitor: Optional[Dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> Event:
        return cls(
            event_type=data['event_type'],
            timestamp=data['timestamp'],
            cursor_position=data.get('cursor_position', []),
            details=EventDetails(data.get('details', {})),
            monitor=data.get('monitor')
        )

    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'cursor_position': self.cursor_position,
            'details': self.details.to_dict(),
            'monitor': self.monitor
        }

    @property
    def is_mouse_event(self) -> bool:
        return self.event_type in ['mouse_click', 'mouse_press', 'mouse_release', 'mouse_down']

    @property
    def is_key_event(self) -> bool:
        return self.event_type == 'key_press'

    @property
    def is_scroll(self) -> bool:
        return self.event_type == 'mouse_scroll'

    @property
    def is_move(self) -> bool:
        return self.event_type == 'mouse_move'


@dataclass
class Aggregation:
    timestamp: float
    end_timestamp: Optional[float]
    reason: str
    event_type: str
    request_state: bool
    screenshot_path: Optional[str]
    events: List[Event]
    monitor: Optional[Dict[str, int]] = None
    burst_id: Optional[str] = None,
    scale_factor: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict) -> Aggregation:
        events = [Event.from_dict(e) for e in data.get('events', [])]
        return cls(
            timestamp=data['screenshot_timestamp'] if 'screenshot_timestamp' in data else data.get('timestamp'),
            end_timestamp=data['end_screenshot_timestamp'] if 'end_screenshot_timestamp' in data else data.get('end_timestamp'),
            reason=data['reason'],
            event_type=data['event_type'],
            request_state=data['request_state'],
            screenshot_path=data.get('screenshot_path'),
            events=events,
            monitor=data.get('monitor'),
            burst_id=data.get('burst_id'),
            scale_factor=data.get('scale_factor', 1.0)
        )

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'end_timestamp': self.end_timestamp,
            'reason': self.reason,
            'event_type': self.event_type,
            'request_state': self.request_state,
            'screenshot_path': self.screenshot_path,
            'events': [e.to_dict() for e in self.events],
            'monitor': self.monitor,
            'burst_id': self.burst_id,
            'scale_factor': self.scale_factor
        }

    def _click_to_relative(self, pos, monitor):
        if not monitor:
            print("No monitor info available for position conversion.")
            return pos
        width = monitor.get('width', 1)
        height = monitor.get('height', 1)
        if isinstance(pos, tuple) or isinstance(pos, list) and len(pos) == 2:
            x, y = pos
            x -= monitor.get('left', 0)
            y -= monitor.get('top', 0)
            relative_x = round((x / width) * 1_000)
            relative_y = round((y / height) * 1_000)
            return (relative_x, relative_y)

    def _reorder_move_scroll_blocks(self, events: List[Event]) -> List[Event]:
        """
        For each contiguous block that contains only 'mouse_move' and 'mouse_scroll' events,
        return the block with all 'mouse_scroll' events first (original relative order preserved),
        then all 'mouse_move' events (original relative order preserved).
        Other events are left in place.
        """
        if not events:
            return []

        out: List[Event] = []
        i = 0
        n = len(events)
        while i < n:
            ev = events[i]
            if ev.event_type in ("mouse_move", "mouse_scroll"):
                # gather contiguous block of only move/scroll
                j = i
                block = []
                while j < n and events[j].event_type in ("mouse_move", "mouse_scroll"):
                    block.append(events[j])
                    j += 1
                # re-order: all scrolls (preserve order) then all moves (preserve order)
                scrolls = [b for b in block if b.event_type == "mouse_scroll"]
                moves = [b for b in block if b.event_type == "mouse_move"]
                out.extend(scrolls)
                out.extend(moves)
                i = j
            else:
                out.append(ev)
                i += 1
        return out

    def to_prompt(self, time, deduplicate=True, min_count=3):
        """
        Generate a prompt with optional event deduplication.

        Args:
            time: Timestamp string for the event
            deduplicate: Whether to deduplicate consecutive events of same type
            min_count: Minimum consecutive events needed to summarize (when deduplicating)
        """
        actions = []
        keys_pressed = []
        events_for_summary = self._reorder_move_scroll_blocks(self.events)

        # Group consecutive events by type
        i = 0
        while i < len(events_for_summary):
            event = self.events[i]
            event_type = event.event_type

            # Count consecutive events of same type
            count = 1
            while i + count < len(self.events) and self.events[i + count].event_type == event_type:
                count += 1

            # Get the group of events
            event_group = self.events[i:i + count]

            if event_type == 'key_press':
                for e in event_group:
                    key = e.details.get('key', 'Key.unknown')
                    key = key.replace("Key.", "") if key and key.startswith("Key.") else key
                    if key:
                        keys_pressed.append(key)

            # Handle mouse events
            elif event_type == 'mouse_down':
                if keys_pressed:
                    actions.append(f"Key(s) pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()

                # Process each click individually (clicks are usually meaningful)
                for e in event_group:
                    button = e.details.get('button', 'Button.unknown')
                    button = button.replace('Button.', '') if button and button.startswith('Button.') else button
                    cursor_pos = e.cursor_position
                    double_click = e.details.get('double_click', False)
                    mouse_str = f"Mouse clicked {button} at {self._click_to_relative(cursor_pos, e.monitor)}"
                    mouse_str += " (double click)" if double_click else ""
                    actions.append(mouse_str)

            elif event_type == 'mouse_scroll':
                if keys_pressed:
                    actions.append(f"Key pressed: {keys_pressed[0]}" if len(keys_pressed) == 1 else f"Keys pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()

                if count >= min_count:
                    scroll_data = event_group[0].details.data
                    direction = self._convert_scroll_direction(scroll_data)
                    actions.append(f"Scrolled {direction} ({count} times)")
                else:
                    for e in event_group:
                        scroll_data = e.details.data
                        direction = self._convert_scroll_direction(scroll_data)
                        actions.append(f"Scrolled {direction}")

            elif event_type == 'mouse_move':
                if keys_pressed:
                    actions.append(f"Key pressed: {keys_pressed[0]}" if len(keys_pressed) == 1 else f"Keys pressed: {'|'.join(keys_pressed)}")
                    keys_pressed.clear()

                # Summarize if many consecutive moves
                if count >= min_count:
                    start_pos = self._click_to_relative(event_group[0].cursor_position, event_group[0].monitor)
                    end_pos = self._click_to_relative(event_group[-1].cursor_position, event_group[-1].monitor)
                    actions.append(f"Mouse moved from {start_pos} to {end_pos} ({count} movements)")
                else:
                    for e in event_group:
                        cursor_pos = e.cursor_position
                        actions.append(f"Mouse moved to {self._click_to_relative(cursor_pos, e.monitor)}")

            i += count

        # Flush any remaining keys
        if keys_pressed:
            actions.append(f"Key pressed: {keys_pressed[0]}" if len(keys_pressed) == 1 else f"Keys pressed: {'|'.join(keys_pressed)}")

        # Format output
        action_list = None
        if actions:
            action_list = '\t' + '\n\t'.join([f"{action}" for action in actions])

        if len(self.events) > 1:
            click_start = self._click_to_relative(self.events[0].cursor_position, self.events[0].monitor)
            click_end = self._click_to_relative(self.events[-1].cursor_position, self.events[-1].monitor)
            click_prompt = f"Cursor positions: {click_start} to {click_end}"
        elif len(self.events) == 1:
            click_start = self._click_to_relative(self.events[0].cursor_position, self.events[0].monitor)
            click_prompt = f"Cursor position: {click_start}"
        else:
            click_prompt = ""

        prompt = f"""
{time} Event:
{click_prompt}
Actions:
{''.join(action_list) if action_list else 'No actions recorded.'}
        """
        return prompt

    def get_image_path(self, session_dir: Path) -> Optional[ImagePath]:
        if not self.screenshot_path:
            return None
        return ImagePath(Path(self.screenshot_path), session_dir)

    @staticmethod
    def _convert_scroll_direction(scroll_data):
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

    def __add__(self, other: Aggregation) -> Aggregation:
        first_agg = self if self.timestamp <= other.timestamp else other
        last_agg = other if first_agg is self else self
        combined_events = self.events + other.events
        combined_events.sort(key=lambda e: e.timestamp)
        return Aggregation(
            timestamp=first_agg.timestamp,
            end_timestamp=last_agg.end_timestamp,
            reason=last_agg.reason,
            event_type=last_agg.event_type,
            request_state=first_agg.request_state,
            screenshot_path=first_agg.screenshot_path,
            events=combined_events,
            monitor=last_agg.monitor,
            burst_id=last_agg.burst_id,
        )


@dataclass
class Caption:
    start_seconds: float
    end_seconds: float
    text: str
    chunk_index: int = 0

    @property
    def start_formatted(self) -> str:
        return f"{int(self.start_seconds // 60):02d}:{int(self.start_seconds % 60):02d}"

    @property
    def end_formatted(self) -> str:
        return f"{int(self.end_seconds // 60):02d}:{int(self.end_seconds % 60):02d}"

    @classmethod
    def from_dict(cls, data: Dict) -> Caption:
        return cls(
            start_seconds=data['start_seconds'],
            end_seconds=data['end_seconds'],
            text=data['caption'],
            chunk_index=data.get('chunk_index', 0)
        )

    def to_dict(self) -> Dict:
        return {
            'start': self.start_formatted,
            'end': self.end_formatted,
            'start_seconds': self.start_seconds,
            'end_seconds': self.end_seconds,
            'caption': self.text,
            'chunk_index': self.chunk_index
        }


@dataclass
class MatchedCaption:
    caption: Caption
    aggregations: List[Aggregation]
    start_index: int
    end_index: int
    screenshot_scale_factor: float = 1.0

    @property
    def image_path(self) -> Optional[str]:
        if self.aggregations:
            return self.aggregations[0].screenshot_path
        return None

    @property
    def all_events(self) -> List[Event]:
        events = []
        for agg in self.aggregations:
            events.extend(agg.events)
        return events

    def to_dict(self) -> Dict:
        return {
            'start_time': self.aggregations[0].timestamp if self.aggregations else 0,
            'end_time': self.aggregations[-1].timestamp if self.aggregations else 0,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'img': self.image_path,
            'caption': self.caption.text,
            'raw_events': [e.to_dict() for e in self.all_events],
            'num_aggregations': len(self.aggregations),
            'start_formatted': self.caption.start_formatted,
            'end_formatted': self.caption.end_formatted,
            'scale_factor': self.screenshot_scale_factor
        }


@dataclass
class ChunkTask:
    session_id: str
    chunk_index: int
    video_path: VideoPath
    prompt: str
    aggregations: List[Aggregation]
    chunk_start_time: float
    chunk_duration: int


@dataclass
class SessionConfig:
    session_folder: Path
    chunk_duration: int = 60
    video_path: Optional[VideoPath] = None
    agg_path: Optional[Path] = None

    @property
    def session_id(self) -> str:
        return self.session_folder.name

    @property
    def chunks_dir(self) -> Path:
        return self.session_folder / "chunks"

    @property
    def captions_dir(self) -> Path:
        return self.session_folder / "captions"

    @property
    def aggregations_dir(self) -> Path:
        return self.session_folder / "aggregations"

    @property
    def screenshots_dir(self) -> Path:
        return self.session_folder / "screenshots"

    @property
    def master_video_path(self) -> Path:
        return self.chunks_dir / "master.mp4"

    @property
    def captions_jsonl(self) -> Path:
        return self.session_folder / "captions.jsonl"

    @property
    def matched_captions_jsonl(self) -> Path:
        return self.session_folder / "data.jsonl"

    def ensure_dirs(self):
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.captions_dir.mkdir(parents=True, exist_ok=True)
        self.aggregations_dir.mkdir(parents=True, exist_ok=True)

    def load_aggregations(self) -> List[Aggregation]:
        if not self.agg_path or not self.agg_path.exists():
            return []

        aggs = []
        with open(self.agg_path, 'r') as f:
            for line in f:
                if line.strip():
                    aggs.append(Aggregation.from_dict(json.loads(line)))
        # add up all aggregations with identical screenshot_paths
        final_aggs = []
        for agg in aggs:
            if final_aggs and final_aggs[-1].screenshot_path == agg.screenshot_path:
                combined_agg = final_aggs[-1] + agg
                final_aggs[-1] = combined_agg
            else:
                final_aggs.append(agg)
        return final_aggs

    def save_captions(self, captions: List[Caption]):
        with open(self.captions_jsonl, 'w', encoding='utf-8') as f:
            for cap in captions:
                f.write(json.dumps(cap.to_dict(), ensure_ascii=False) + '\n')

    def save_matched_captions(self, matched: List[MatchedCaption]):
        with open(self.matched_captions_jsonl, 'w', encoding='utf-8') as f:
            for m in matched:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + '\n')
