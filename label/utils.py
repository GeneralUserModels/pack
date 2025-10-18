from pathlib import Path
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import re


def list_screenshots(session_folder: Path) -> List[Path]:
    """List all screenshot files in the session folder."""
    screenshots = []
    screenshots_dir = session_folder / "screenshots"

    if not screenshots_dir.exists():
        return []

    for p in sorted(screenshots_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            screenshots.append(p)

    return screenshots


def extract_timestamp_from_filename(p: Path) -> Optional[float]:
    """Extract timestamp from filename or use file modification time."""
    timestamp_re = re.compile(r"(\d+\.\d+)")
    m = timestamp_re.search(p.name)

    if m:
        try:
            return float(m.group(1))
        except Exception as e:
            print(f"Error parsing timestamp from filename {p.name}: {e}")
            pass

    try:
        return p.stat().st_mtime
    except Exception as e:
        print(f"Error getting modification time for file {p}: {e}")
        return None


@dataclass
class SessionTask:
    """Single chunk processing task."""
    session_id: str
    chunk_index: int
    video_path: Path
    prompt: str
    aggregations: List[Any]
    output_dir: Path
    chunk_start_time: float
    chunk_duration: int


@dataclass
class CaptionEntry:
    """Caption with timestamp."""
    timestamp_seconds: float
    timestamp_formatted: str
    caption: str
    chunk_index: int
    metadata: Optional[Dict] = None
