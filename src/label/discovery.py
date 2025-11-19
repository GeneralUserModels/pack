from pathlib import Path
from typing import List, Tuple

from label.models import SessionConfig, VideoPath


def discover_sessions(
    sessions_root: Path,
    chunk_duration: int = 60,
    skip_existing: bool = False
) -> List[SessionConfig]:

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root not found: {sessions_root}")

    configs = []

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        screenshots_dir = session_dir / "screenshots"
        agg_path = session_dir / "aggregations.jsonl"

        if not (screenshots_dir.exists() and agg_path.exists()):
            continue

        if skip_existing and (session_dir / "data.jsonl").exists():
            continue

        has_images = any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for p in screenshots_dir.iterdir()
        )

        if has_images:
            configs.append(SessionConfig(
                session_folder=session_dir,
                chunk_duration=chunk_duration,
                agg_path=agg_path
            ))

    return configs


def discover_screenshots_sessions(
    sessions_root: Path,
    chunk_duration: int = 60,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> List[SessionConfig]:

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root not found: {sessions_root}")

    configs = []

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        # Look for screenshots directory
        screenshots_dir = session_dir / "screenshots"
        if not screenshots_dir.exists():
            continue

        # Check if there are any image files
        image_files = [
            f for f in screenshots_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        ]

        if image_files:
            configs.append(SessionConfig(
                session_folder=session_dir,
                chunk_duration=chunk_duration,
                _screenshots_dir=screenshots_dir
            ))

    return configs


def create_single_config(
    session_dir: Path,
    chunk_duration: int,
    screenshots_only: bool,
    image_exts: Tuple[str, ...],
    prompt: str = ""
) -> SessionConfig:

    if screenshots_only:
        # Check if there's a screenshots subdirectory first
        screenshots_dir = session_dir / "screenshots"
        if screenshots_dir.exists() and screenshots_dir.is_dir():
            search_dir = screenshots_dir
        else:
            search_dir = session_dir
        
        image_files = [
            f for f in search_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        ]

        if not image_files:
            raise RuntimeError(f"No image files found in {search_dir}")

        return SessionConfig(
            session_folder=session_dir,
            chunk_duration=chunk_duration,
            _screenshots_dir=search_dir
        )
    else:
        return SessionConfig(
            session_folder=session_dir,
            chunk_duration=chunk_duration,
            agg_path=session_dir / "aggregations.jsonl"
        )
