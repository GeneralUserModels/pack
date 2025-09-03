import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage


class CompletionDataset:
    def __init__(self, session_names: List[str], percentile: int = 87, video_length: int = 60, log_interval_seconds: float = 4.0):
        self.sessions = session_names
        self.percentile = percentile
        self.video_length = video_length
        self.log_interval_seconds = float(log_interval_seconds)
        self.data = DatasetDict()
        random.seed(42)

    def to_dataset(self) -> DatasetDict:

        sessions_data = []
        for session in self.sessions:
            sessions_data.append(self.load_captions_from_session(session))

        flattened = [entry for session_list in sessions_data for entry in session_list]

        features = Features({
            **{k: Value("string") for k in flattened[0] if k not in ("start_img", "end_img")},
            "start_img": HFImage(),
            "end_img": HFImage()
        })

        self.data = Dataset.from_list(flattened, features=features)
        return self.data

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

        for chunk_path in sorted(chunks_path.glob("*_result.json")):
            captions.extend(self._load_captions_from_chunk(chunk_path, aggregated_logs))
        return captions

    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> float:
        """
        Accepts formats like "MM:SS", "HH:MM:SS", "SS", or "SS.sss".
        Returns seconds as float. Returns 0.0 on parse errors.
        """
        if time_str is None:
            return 0.0
        if isinstance(time_str, (int, float)):
            return float(time_str)
        parts = time_str.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return 0.0
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600.0 + minutes * 60.0 + seconds
        elif len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60.0 + seconds
        elif len(parts) == 1:
            return parts[0]
        else:
            return 0.0

    def _load_captions_from_chunk(self, file_path: Path, aggregated_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as file:
            chunk_json = json.load(file)

        results = chunk_json.get("result", [])
        chunk_start_time_str = chunk_json.get("start_time", "00:00")
        chunk_start_seconds = self._parse_time_to_seconds(chunk_start_time_str)

        enriched = []
        for entry in results:
            caption_text = entry.get("caption")
            start_rel = entry.get("start")
            end_rel = entry.get("end")
            if caption_text is None or start_rel is None or end_rel is None:
                continue

            start_offset = chunk_start_seconds + self._parse_time_to_seconds(start_rel)
            end_offset = chunk_start_seconds + self._parse_time_to_seconds(end_rel)

            if not aggregated_logs:
                print(f"Warning: no aggregated logs available for {file_path}. Skipping timestamps.")
                enriched_entry = {
                    "text": caption_text,
                    "start_time": None,
                    "end_time": None,
                    "start_img": None,
                    "end_img": None,
                }
                enriched.append(enriched_entry)
                continue

            interval = self.log_interval_seconds

            start_idx = max(0, int(math.floor(start_offset / interval)))
            end_idx = min(len(aggregated_logs) - 1, int(math.ceil(end_offset / interval)))

            if start_idx <= end_idx:
                aggregated_entry_logs = aggregated_logs[start_idx:end_idx + 1]
                enriched_entry = {
                    "text": caption_text,
                    "start_time": aggregated_entry_logs[0].get("start_timestamp"),
                    "end_time": aggregated_entry_logs[-1].get("end_timestamp"),
                    "start_img": aggregated_entry_logs[0].get("start_screenshot_path"),
                    "end_img": aggregated_entry_logs[-1].get("end_screenshot_path"),
                }
            else:
                print(f"Warning: start_idx {start_idx} > end_idx {end_idx} for {file_path.parent.parent.name}.")
                enriched_entry = {
                    "text": caption_text,
                    "start_time": None,
                    "end_time": None,
                    "start_img": None,
                    "end_img": None,
                }

            enriched.append(enriched_entry)

        return enriched

    def save(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)
        if not self.data:
            raise ValueError("No data to save. Please run to_dataset() first.")
        self.data.save_to_disk(path)


if __name__ == "__main__":
    PERCENTILE = 85
    VIDEO_LENGTH = 60
    path = Path(__file__).parent.parent / "logs"
    sessions = []
    for session in path.iterdir():
        if session.is_dir() and session.name.startswith("session_") and (session / f"chunks_{PERCENTILE}_{VIDEO_LENGTH}").exists():
            sessions.append(session.name)
    dataset = CompletionDataset(session_names=sessions, percentile=PERCENTILE, video_length=VIDEO_LENGTH)
    dataset.to_dataset()
    dataset.save(Path(__file__).parent.parent / "datasets" / "image_dataset_85")
