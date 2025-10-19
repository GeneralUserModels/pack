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
    is_start: bool
    screenshot_path: Optional[str]
    events: List[Event]
    monitor: Optional[Dict[str, int]] = None
    burst_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> Aggregation:
        events = [Event.from_dict(e) for e in data.get('events', [])]
        return cls(
            timestamp=data['timestamp'],
            end_timestamp=data.get('end_timestamp'),
            reason=data['reason'],
            event_type=data['event_type'],
            is_start=data['is_start'],
            screenshot_path=data.get('screenshot_path'),
            events=events,
            monitor=data.get('monitor'),
            burst_id=data.get('burst_id')
        )

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'end_timestamp': self.end_timestamp,
            'reason': self.reason,
            'event_type': self.event_type,
            'is_start': self.is_start,
            'screenshot_path': self.screenshot_path,
            'events': [e.to_dict() for e in self.events],
            'monitor': self.monitor,
            'burst_id': self.burst_id
        }

    def to_prompt(self, time_marker: str) -> str:
        return f"[{time_marker}] {self.reason}\n"

    def get_image_path(self, session_dir: Path) -> Optional[ImagePath]:
        if not self.screenshot_path:
            return None
        return ImagePath(Path(self.screenshot_path), session_dir)


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
        return aggs

    def save_captions(self, captions: List[Caption]):
        with open(self.captions_jsonl, 'w', encoding='utf-8') as f:
            for cap in captions:
                f.write(json.dumps(cap.to_dict(), ensure_ascii=False) + '\n')

    def save_matched_captions(self, matched: List[MatchedCaption]):
        with open(self.matched_captions_jsonl, 'w', encoding='utf-8') as f:
            for m in matched:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + '\n')
