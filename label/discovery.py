from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SessionConfig:
    """Configuration for a single session."""
    session_folder: Path
    agg_jsonl: Optional[Path]  # None for video-only mode
    out_chunks_dir: Path
    video_path: Optional[Path] = None  # For video-only mode


def discover_sessions(
    sessions_root: Path,
    agg_filename: str = "aggregations.jsonl",
    chunk_duration: int = 60,
    skip_existing: bool = False
) -> List[SessionConfig]:
    """Discover sessions with screenshots and aggregation logs."""
    configs = []

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root not found: {sessions_root}")

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        screenshots_dir = session_dir / "screenshots"
        agg_path = session_dir / agg_filename

        if not (screenshots_dir.exists() and agg_path.exists()):
            continue
        if skip_existing and (session_dir / "matched_captions.jsonl").exists():
            print(f"[Discovery] Skipping {session_dir.name}: already processed")
            continue

        has_images = any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for p in screenshots_dir.iterdir()
        )

        if has_images:
            configs.append(SessionConfig(
                session_folder=session_dir,
                agg_jsonl=agg_path,
                out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
            ))
            print(f"[Discovery] Found: {session_dir.name}")
        else:
            print(f"[Discovery] Skipping {session_dir.name}: no images")

    return configs


def discover_video_only_sessions(
    sessions_root: Path,
    chunk_duration: int = 60,
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
) -> List[SessionConfig]:
    """Discover sessions with only video files (no logs)."""
    configs = []

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root not found: {sessions_root}")

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        # Look for video in session dir or video/ subdir
        video_files = [
            f for f in session_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]

        video_subdir = session_dir / "video"
        if video_subdir.exists():
            video_files.extend([
                f for f in video_subdir.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ])

        if video_files:
            configs.append(SessionConfig(
                session_folder=session_dir,
                agg_jsonl=None,
                out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
                video_path=video_files[0],
            ))
            print(f"[Discovery] Found video: {session_dir.name} ({video_files[0].name})")

    return configs


def create_single_session_config(
    session_dir: Path,
    agg_filename: str,
    chunk_duration: int,
    video_only: bool,
    video_extensions: Tuple[str, ...]
) -> SessionConfig:
    """Create configuration for a single session."""
    if video_only:
        video_files = [
            f for f in session_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]

        video_subdir = session_dir / "video"
        if video_subdir.exists():
            video_files.extend([
                f for f in video_subdir.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ])

        if not video_files:
            raise RuntimeError(f"No video files found in {session_dir}")

        return SessionConfig(
            session_folder=session_dir,
            agg_jsonl=None,
            out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
            video_path=video_files[0],
        )
    else:
        return SessionConfig(
            session_folder=session_dir,
            agg_jsonl=session_dir / agg_filename,
            out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
        )
