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


def discover_video_sessions(
    sessions_root: Path,
    chunk_duration: int = 60,
    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
) -> List[SessionConfig]:

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root not found: {sessions_root}")

    configs = []

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        video_files = [
            f for f in session_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_exts
        ]

        video_subdir = session_dir / "video"
        if video_subdir.exists():
            video_files.extend([
                f for f in video_subdir.iterdir()
                if f.is_file() and f.suffix.lower() in video_exts
            ])

        if video_files:
            configs.append(SessionConfig(
                session_folder=session_dir,
                chunk_duration=chunk_duration,
                video_path=VideoPath(video_files[0])
            ))

    return configs


def create_single_config(
    session_dir: Path,
    chunk_duration: int,
    video_only: bool,
    video_exts: Tuple[str, ...],
    prompt: str = ""
) -> SessionConfig:

    if video_only:
        video_files = [
            f for f in session_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_exts
        ]

        video_subdir = session_dir / "video"
        if video_subdir.exists():
            video_files.extend([
                f for f in video_subdir.iterdir()
                if f.is_file() and f.suffix.lower() in video_exts
            ])

        if not video_files:
            raise RuntimeError(f"No video files found in {session_dir}")

        return SessionConfig(
            session_folder=session_dir,
            chunk_duration=chunk_duration,
            video_path=VideoPath(video_files[0])
        )
    else:
        return SessionConfig(
            session_folder=session_dir,
            chunk_duration=chunk_duration,
            agg_path=session_dir / "aggregations.jsonl"
        )
