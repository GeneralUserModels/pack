import json
import random
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
import pdb

class CompletionDataset:
    def __init__(self, session_names: List[str], percentile: int = 85, video_length: int = 60):
        self.sessions = session_names
        self.percentile = percentile
        self.video_length = video_length
        self.data = DatasetDict()
        random.seed(42)

    def to_dataset(self, split_sizes=None) -> DatasetDict:
        if split_sizes is None:
            split_sizes = {"train": 0.8, "test": 0.2}

        sessions_data = []
        for session in self.sessions:
            sessions_data.append(self.load_captions_from_session(session))

        random.shuffle(sessions_data)
        split_counts = self._session_counts_from_split_sizes(split_sizes)
        start = 0

        for split in ("train", "test"):
            count = split_counts.get(split, 0)
            split_data = sessions_data[start:start + count]
            flattened = [entry for session_list in split_data for entry in session_list]

            features = Features({
                **{k: Value("string") for k in flattened[0] if k not in ("start_img", "end_img")},
                "start_img": HFImage(),
                "end_img": HFImage()
            })

            self.data[split] = Dataset.from_list(flattened, features=features)
            start += count

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

            # just in case the end_offset is out of bounds
            end_offset = min(end_offset, len(aggregated_logs) - 1)

            try:
                enriched_entry = {
                    "text": caption_text,
                    "start_time": aggregated_logs[start_offset].get("time"),
                    "end_time": aggregated_logs[end_offset].get("time"),
                    "start_img": aggregated_logs[start_offset].get("img"),
                    "end_img": aggregated_logs[end_offset].get("img"),
                }

            except Exception as e:
                print(f"Error: {e}")

            enriched.append(enriched_entry)

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

    def _session_counts_from_split_sizes(self, split_sizes: dict) -> dict:
        total = len(self.sessions)
        if total < len(split_sizes):
            raise ValueError(f"Not enough sessions ({total}) to cover all splits ({len(split_sizes)}).")

        raw_counts = {k: total * v for k, v in split_sizes.items()}
        floored = {k: int(math.floor(c)) for k, c in raw_counts.items()}

        for k in floored:
            if floored[k] == 0:
                floored[k] = 1

        while sum(floored.values()) > total:
            max_k = max(floored, key=floored.get)
            floored[max_k] -= 1

        remainder = total - sum(floored.values())
        residuals = sorted(
            ((k, raw_counts[k] - floored[k]) for k in split_sizes),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(remainder):
            floored[residuals[i % len(residuals)][0]] += 1

        return floored

    def save(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)
        if not self.data:
            raise ValueError("No data to save. Please run to_dataset() first.")
        self.data.save_to_disk(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completion dataset from session logs")
    parser.add_argument("--percentile", type=int, default=85, 
                       help="Percentile value for aggregated logs (default: 85)")
    parser.add_argument("--video-length", type=int, default=60,
                       help="Video length in seconds (default: 60)")
    
    args = parser.parse_args()
    
    path = Path(__file__).parent.parent / "logs"
    sessions = []
    for session in path.iterdir():
        if session.is_dir() and session.name.startswith("session_") and (session / f"chunks_{args.percentile}_{args.video_length}").exists():
            sessions.append(session.name)
    dataset = CompletionDataset(session_names=sessions, percentile=args.percentile, video_length=args.video_length)
    dataset.to_dataset()
    dataset.save(Path(__file__).parent.parent / "datasets" / "completion_dataset")
