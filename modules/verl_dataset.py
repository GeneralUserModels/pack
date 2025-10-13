import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import base64
from datetime import datetime

# =======================
# Image helpers
# =======================

def process_image_to_bytes(image_path: str) -> Dict[str, str]:
    """Convert image file to base64 bytes format. If path is missing, skip."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return None
    with open(p, 'rb') as f:
        image_bytes = f.read()
    # detect format from extension (fallback to png)
    ext = p.suffix.lower().lstrip('.') or 'png'
    return {
        "bytes": base64.b64encode(image_bytes).decode('utf-8'),
        "format": ext,
    }


def create_dataset_entry(question: str, images: List[Dict[str, str]], ground_truth: str) -> Dict[str, Any]:
    return {
        "prompt": [
            {"role": "user", "content": question}
        ],
        "images": [im for im in images if im],  # filter Nones
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "agent_name": "think_retrieve_revise_agent",
        "data_source": "your_dataset",
        "extra_info": {
            "question": question,
            "answer": ground_truth
        }
    }


# =======================
# Dataset loader (original structure preserved, with small fixes)
# =======================

class CompletionDataset:
    def __init__(self, session_names: List[str], percentile: int = 85, video_length: int = 60):
        self.sessions = session_names
        self.percentile = percentile
        self.video_length = video_length
        self.data = {}
        random.seed(42)

    def get_splits(self, split_sizes=None):
        if split_sizes is None:
            split_sizes = {"train": 0.8, "validation": 0.1, "test": 0.1}

        # Load and combine all sessions data
        all_data = []
        for session in self.sessions:
            session_data = self.load_captions_from_session(session)
            all_data.extend(session_data)

        # Sort by start_time for temporal splitting
        all_data.sort(key=lambda x: x.get("start_time", ""))

        # Calculate split indices based on temporal order
        total_samples = len(all_data)
        train_end = int(total_samples * split_sizes["train"])
        val_end = train_end + int(total_samples * split_sizes["validation"])

        splits_data = {
            "train": all_data[:train_end],
            "validation": all_data[train_end:val_end],
            "test": all_data[val_end:]
        }

        return splits_data

    def load_captions_from_session(self, session: str) -> List[Dict[str, Any]]:
        session_path = Path(__file__).parent.parent / "logs" / session
        captions: List[Dict[str, Any]] = []
        if not session_path.exists():
            raise FileNotFoundError(f"Session path {session_path} does not exist.")
        chunks_path = session_path / f"chunks_{self.percentile}_{self.video_length}"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks path {chunks_path} does not exist.")

        aggregated_path = session_path / f"aggregated_logs_{self.percentile}.json"
        if not aggregated_path.exists():
            raise FileNotFoundError(f"Aggregated logs path {aggregated_path} does not exist.")

        with open(aggregated_path, 'r', encoding='utf-8') as f:
            aggregated_logs = json.load(f)

        flattened_aggregated_logs = []
        for log in aggregated_logs:
            flattened = [
                {
                    "time": log.get("start_timestamp"),
                    "img": log.get("start_screenshot_path"),
                },
                {
                    "time": log.get("end_timestamp"),
                    "img": log.get("end_screenshot_path"),
                }
            ]
            flattened_aggregated_logs.extend(flattened)

        for chunk_path in sorted(chunks_path.glob("*_result.json")):
            captions.extend(self._load_captions_from_chunk(chunk_path, flattened_aggregated_logs))
        return captions

    def _load_captions_from_chunk(self, file_path: Path, aggregated_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as file:
            chunk_json = json.load(file)

        results = chunk_json.get("result", [])

        enriched = []
        for entry in results:
            # entry is expected to have 'caption', 'start', 'end' (start/end are mm:ss strings)
            caption_text = entry.get("caption")
            start_rel = entry.get("start")
            end_rel = entry.get("end")
            if caption_text is None or start_rel is None or end_rel is None:
                # skip malformed entries
                continue

            start_offset = self._parse_time_to_seconds(start_rel)
            end_offset = self._parse_time_to_seconds(end_rel)

            # ensure within bounds
            end_offset = min(end_offset, max(0, len(aggregated_logs) - 1))
            start_offset = min(start_offset, max(0, len(aggregated_logs) - 1))

            try:
                enriched_entry = {
                    "text": caption_text,
                    "start_time": aggregated_logs[start_offset].get("time"),
                    "end_time": aggregated_logs[end_offset].get("time"),
                    "start_img": aggregated_logs[start_offset].get("img"),
                    "end_img": aggregated_logs[end_offset].get("img"),
                }
                enriched.append(enriched_entry)
            except Exception as e:
                print(f"Error: {e}")

        return enriched

    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> int:
        parts = time_str.split(":")
        try:
            parts = [int(p) for p in parts]
        except ValueError:
            return 0
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0

    # =======================
    # Pretty timestamp helpers (for question/ground_truth formatting)
    # =======================

    def _ts(self, s: str) -> datetime:
        # expected format: "%Y-%m-%d_%H-%M-%S-%f"
        return datetime.strptime(s, "%Y-%m-%d_%H-%M-%S-%f")

    def _format_timestamp_nice(self, dt: datetime) -> str:
        """Format timestamp as "Monday, July 4th - 8:15 AM" (minute precision)."""
        snapped_dt = dt.replace(second=0, microsecond=0)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = day_names[snapped_dt.weekday()]

        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        month_name = month_names[snapped_dt.month - 1]

        day = snapped_dt.day
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

        hour = snapped_dt.hour
        if hour == 0:
            hour = 12
            ampm = "AM"
        elif hour < 12:
            ampm = "AM"
        elif hour == 12:
            ampm = "PM"
        else:
            hour -= 12
            ampm = "PM"

        minute_str = f"{snapped_dt.minute:02d}"
        return f"{day_name}, {month_name} {day}{suffix} - {hour}:{minute_str} {ampm}"

    def _fmt_event_line(self, item: Dict[str, Any]) -> str:
        """Return a single line like: [ Sunday, August 3rd - 8:14 PM ] Some action text."""
        # Allow items without timestamps to pass through gracefully
        ts = item.get('start_time')
        if ts:
            try:
                dt = self._ts(ts)
                pretty = self._format_timestamp_nice(dt)
                return f"[ {pretty} ] {item['text']}"
            except Exception:
                pass
        # fallback: raw
        return f"[ {item.get('start_time','')} ] {item['text']}"

    # =======================
    # Sliding-window -> entries
    # =======================

    def sliding_windows(self, split_items: List[Dict[str, Any]], past_len: int, future_len: int, stride: int) -> List[Tuple[int, int]]:
        windows = []
        min_required = past_len + future_len
        n = len(split_items)
        for i in range(0, max(0, n - min_required + 1), stride):
            windows.append((i, i + past_len, i + past_len + future_len))  # [i:past_end) -> past, [past_end:future_end) -> future
        return windows

    def build_question_and_images(self, past_events: List[Dict[str, Any]], image_mode: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Assemble the question string (with optional <image> tags) and a parallel images list
        whose order matches the positions of <image> in the text.
        image_mode: 'none' | 'start' | 'end' | 'sandwich'
        """
        image_mode = (image_mode or 'end').lower()
        assert image_mode in {"none", "start", "end", "sandwich"}

        lines: List[str] = []
        images: List[Dict[str, str]] = []

        for ev in past_events:
            line = self._fmt_event_line(ev)
            # insert images per mode, ensuring <image> occurrences match images list order
            if image_mode == 'start':
                img = process_image_to_bytes(ev.get('start_img'))
                if img:
                    lines.append("<image>")
                    images.append(img)
                lines.append(line)
            elif image_mode == 'end':
                lines.append(line)
                img = process_image_to_bytes(ev.get('end_img'))
                if img:
                    lines.append("<image>")
                    images.append(img)
            elif image_mode == 'sandwich':
                img_s = process_image_to_bytes(ev.get('start_img'))
                if img_s:
                    lines.append("<image>")
                    images.append(img_s)
                lines.append(line)
                img_e = process_image_to_bytes(ev.get('end_img'))
                if img_e:
                    lines.append("<image>")
                    images.append(img_e)
            else:  # none
                lines.append(line)

        question = "\n\n".join(lines)
        return question, images

    def build_ground_truth(self, future_events: List[Dict[str, Any]]) -> str:
        lines = [self._fmt_event_line(ev) for ev in future_events]
        return "\n\n".join(lines)


# =======================
# Orchestrator: create parquet datasets per split
# =======================

def build_and_save_parquets(
    dataset: CompletionDataset,
    output_dir: Path,
    past_len: int = 3,
    future_len: int = 2,
    stride: int = 1,
    image_mode: str = "end",
    split_sizes=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = dataset.get_splits(split_sizes=split_sizes)

    # helpful debug: show counts per split and min required for a window
    min_required = past_len + future_len
    for split_name, items in splits.items():
        n_items = len(items)
        print(f"[split] {split_name}: {n_items} items (min_required={min_required}, stride={stride})")

        windows = dataset.sliding_windows(items, past_len=past_len, future_len=future_len, stride=stride)
        print(f"[split] {split_name}: {len(windows)} windows")

        entries: List[Dict[str, Any]] = []

        for (i, past_end, future_end) in windows:
            past_events = items[i:past_end]
            future_events = items[past_end:future_end]

            question, images = dataset.build_question_and_images(past_events, image_mode=image_mode)
            ground_truth = dataset.build_ground_truth(future_events)

            entry = create_dataset_entry(question=question, images=images, ground_truth=ground_truth)
            entries.append(entry)
        # Save as parquet with native Python objects
        if entries:
            df = pd.DataFrame(entries)
            out_path = output_dir / f"{split_name}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[write] {split_name}: wrote {len(df)} rows to {out_path}")
        else:
            print(f"[warn] No entries for split '{split_name}'. Skipping file.")


# =======================
# CLI
# =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sliding-window parquet datasets from session logs")
    parser.add_argument("--percentile", type=int, default=85, help="Percentile value for aggregated logs (default: 85)")
    parser.add_argument("--video-length", type=int, default=60, help="Video length in seconds (default: 60)")
    parser.add_argument("--past-len", type=int, default=8, help="Number of past events (K) for the question")
    parser.add_argument("--future-len", type=int, default=8, help="Number of future events (K) for ground truth")
    parser.add_argument("--stride", type=int, default=4, help="Stride for the sliding window")
    parser.add_argument("--image-mode", type=str, default="end", choices=["none", "start", "end", "sandwich"], help="Include images before/after/both/none per event in the question, using <image> tags and parallel images list")
    parser.add_argument("--out", type=str, default=None, help="Output directory for parquet files (defaults to ../datasets)")

    args = parser.parse_args()

    path = Path(__file__).parent.parent / "logs"
    sessions = []
    for session in path.iterdir():
        if session.is_dir() and session.name.startswith("session_") and (session / f"chunks_{args.percentile}_{args.video_length}").exists():
            sessions.append(session.name)

    print(f"Found {len(sessions)} sessions: {sessions}")
    print("Using temporal splitting (combining all sessions and sorting by timestamp)")

    dataset = CompletionDataset(session_names=sessions, percentile=args.percentile, video_length=args.video_length)

    output_dir = Path(args.out) if args.out else (Path(__file__).parent.parent / "datasets")

    build_and_save_parquets(
        dataset=dataset,
        output_dir=output_dir,
        past_len=args.past_len,
        future_len=args.future_len,
        stride=args.stride,
        image_mode=args.image_mode,
    )

    print(f"Parquet files written to: {output_dir}")
