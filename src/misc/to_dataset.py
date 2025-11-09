import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from datasets import Dataset, Features, Value, Image as HFImage


def parse_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    Parse timestamp from filename in various formats.

    Supports:
    - Unix timestamp: 1762508790.129177_reason_move_start_stale.jpg
    - Date format: img_motogfinalfix20171012150835.jpg

    Returns Unix timestamp as float.
    """
    # Try Unix timestamp format first (more precise)
    unix_match = re.search(r'(\d+\.\d+)', filename)
    if unix_match:
        return float(unix_match.group(1))

    # Try date format: YYYYMMDDHHMMSS
    date_match = re.search(r'(\d{14})', filename)
    if date_match:
        date_str = date_match.group(1)
        dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
        return dt.timestamp()

    return None


def mmss_to_seconds(mmss: str) -> int:
    """Convert MM:SS format to total seconds."""
    parts = mmss.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def unix_to_formatted_timestamp(unix_time: float) -> str:
    """
    Convert Unix timestamp to format: 2025-07-30_10-12-54-036554
    """
    dt = datetime.fromtimestamp(unix_time)
    microseconds = int((unix_time % 1) * 1_000_000)
    return dt.strftime('%Y-%m-%d_%H-%M-%S') + f'-{microseconds:06d}'


def load_and_sort_screenshots(img_dir: Path) -> List[Tuple[Path, float]]:
    """
    Load all screenshots from directory and sort by timestamp.

    Returns list of (filepath, timestamp) tuples sorted by timestamp.
    """
    screenshots = []

    for img_file in img_dir.glob('*.jpg'):
        timestamp = parse_timestamp_from_filename(img_file.name)
        if timestamp is not None:
            screenshots.append((img_file, timestamp))

    for img_file in img_dir.glob('*.png'):
        timestamp = parse_timestamp_from_filename(img_file.name)
        if timestamp is not None:
            screenshots.append((img_file, timestamp))

    # Sort by timestamp
    screenshots.sort(key=lambda x: x[1])

    return screenshots


def get_screenshot_by_mmss_index(screenshots: List[Tuple[Path, float]],
                                 mmss: str) -> Optional[Tuple[Path, float]]:
    """
    Get screenshot by MM:SS index.
    00:00 returns screenshots[0], 00:01 returns screenshots[1], etc.
    """
    seconds = mmss_to_seconds(mmss)

    if 0 <= seconds < len(screenshots):
        return screenshots[seconds]

    return None


def process_format1(jsonl_path: Path) -> List[Dict]:
    """Process format 1: data.jsonl with direct img paths."""
    records = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            record = {
                'text': data['caption'],
                'start_time': unix_to_formatted_timestamp(data['start_time']),
                'end_time': unix_to_formatted_timestamp(data['end_time']),
                'img': data['img']
            }
            records.append(record)

    return records


def process_format2(jsonl_path: Path, img_dir: Path) -> List[Dict]:
    """Process format 2: captions.jsonl with separate screenshot directory."""
    records = []

    # Load and sort screenshots
    screenshots = load_and_sort_screenshots(img_dir)

    if not screenshots:
        raise ValueError(f"No valid screenshots found in {img_dir}")

    print(f"Loaded {len(screenshots)} screenshots from {img_dir}")

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            # Get start screenshot
            start_screenshot = get_screenshot_by_mmss_index(screenshots, data['start'])

            # Get end screenshot (next second)
            end_mmss_seconds = mmss_to_seconds(data['end']) + 1
            end_mmss = f"{end_mmss_seconds // 60:02d}:{end_mmss_seconds % 60:02d}"
            end_screenshot = get_screenshot_by_mmss_index(screenshots, end_mmss)

            # If end screenshot doesn't exist, use the last one
            if end_screenshot is None:
                end_screenshot = screenshots[-1]

            if start_screenshot is None:
                print(f"Warning: Could not find screenshot for {data['start']}, skipping...")
                continue

            record = {
                'text': data['caption'],
                'start_time': unix_to_formatted_timestamp(start_screenshot[1]),
                'end_time': unix_to_formatted_timestamp(end_screenshot[1]),
                'img': str(start_screenshot[0])
            }
            records.append(record)

    return records


def create_hf_dataset(records: List[Dict]) -> Dataset:
    """Create HuggingFace Dataset from records."""
    # Define features
    features = Features({
        'text': Value('string'),
        'start_time': Value('string'),
        'end_time': Value('string'),
        'img': HFImage()
    })

    # Create dataset
    dataset = Dataset.from_dict(
        {
            'text': [r['text'] for r in records],
            'start_time': [r['start_time'] for r in records],
            'end_time': [r['end_time'] for r in records],
            'img': [r['img'] for r in records]
        },
        features=features
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSONL data to HuggingFace Dataset'
    )
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='Path to JSONL file'
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        help='Directory containing screenshots (required for format 2)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory to save the dataset'
    )
    parser.add_argument(
        '--format',
        type=int,
        choices=[1, 2],
        help='Input format (1 or 2). If not specified, auto-detect from first line.'
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    output_dir = Path(args.output_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    # Auto-detect format if not specified
    input_format = args.format
    if input_format is None:
        with open(jsonl_path, 'r') as f:
            first_line = json.loads(f.readline().strip())
            # Format 1 has 'img' field with full path, format 2 has 'start' field
            if 'img' in first_line and 'raw_events' in first_line:
                input_format = 1
                print("Auto-detected format 1")
            elif 'start' in first_line and 'chunk_index' in first_line:
                input_format = 2
                print("Auto-detected format 2")
            else:
                raise ValueError("Could not auto-detect format. Please specify --format")

    # Process based on format
    if input_format == 1:
        print("Processing format 1...")
        records = process_format1(jsonl_path)
    else:  # format 2
        if args.img_dir is None:
            raise ValueError("--img-dir is required for format 2")
        img_dir = Path(args.img_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        print("Processing format 2...")
        records = process_format2(jsonl_path, img_dir)

    print(f"Processed {len(records)} records")

    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = create_hf_dataset(records)

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    print(f"Dataset saved successfully with {len(dataset)} examples!")
    print(f"\nDataset info:")
    print(dataset)


if __name__ == '__main__':
    main()
