import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

from label.processor import load_hash_cache, dedupe_images_by_hash


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


def formatted_timestamp_to_unix(formatted: str) -> float:
    """
    Convert formatted timestamp back to Unix timestamp.
    Format: 2025-07-30_10-12-54-036554
    """
    # Split off microseconds
    main_part, microseconds_str = formatted.rsplit('-', 1)
    dt = datetime.strptime(main_part, '%Y-%m-%d_%H-%M-%S')
    microseconds = int(microseconds_str) / 1_000_000
    return dt.timestamp() + microseconds


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


def process_format2(
    jsonl_path: Path,
    img_dir: Path,
    hash_map: Optional[Dict[str, int]] = None,
    dedupe_threshold: int = 1
) -> List[Dict]:
    """Process format 2: captions.jsonl with separate screenshot directory.
    
    Args:
        jsonl_path: Path to the captions.jsonl file
        img_dir: Directory containing screenshots
        hash_map: Optional hash cache for deduplication (to match processor behavior)
        dedupe_threshold: Hamming distance threshold for deduplication
    """
    records = []

    # Load and sort screenshots
    screenshots = load_and_sort_screenshots(img_dir)

    if not screenshots:
        raise ValueError(f"No valid screenshots found in {img_dir}")
    
    # Apply same deduplication as processor if hash_map provided
    if hash_map:
        paths = [s[0] for s in screenshots]
        deduped_paths = dedupe_images_by_hash(paths, hash_map, dedupe_threshold)
        # Rebuild screenshots list with only kept paths
        deduped_set = set(deduped_paths)
        screenshots = [(p, t) for p, t in screenshots if p in deduped_set]

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


def split_dataset(dataset: Dataset, split_ratios: List[float]) -> DatasetDict:
    """
    Split dataset into train/test/validation sets.

    Args:
        dataset: The full dataset to split
        split_ratios: List of [train_ratio, test_ratio, val_ratio]

    Returns:
        DatasetDict with 'train', 'test', and 'validation' splits
    """
    train_ratio, test_ratio, val_ratio = split_ratios

    # Validate ratios sum to 1.0
    total = sum(split_ratios)
    if not (0.99 <= total <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    # Validation gets the remainder to handle rounding
    val_size = total_size - train_size - test_size

    print(f"\nSplit sizes: train={train_size}, test={test_size}, validation={val_size}")

    # Create splits
    train_test_split = dataset.train_test_split(
        test_size=test_size + val_size,
        shuffle=False,
    )

    test_val_split = train_test_split['test'].train_test_split(
        test_size=val_size,
        shuffle=False,
    )

    return DatasetDict({
        'train': train_test_split['train'],
        'test': test_val_split['train'],
        'validation': test_val_split['test']
    })


def split_dataset_by_time(dataset: Dataset, time_durations: List[int]) -> DatasetDict:
    """
    Split dataset into train/test/validation sets based on time durations.
    
    Data is split sequentially (no shuffling) based on timestamps:
    - First duration: train set
    - Second duration: test set  
    - Third duration: validation set
    
    Args:
        dataset: The full dataset to split (must have 'start_time' field)
        time_durations: List of [train_seconds, test_seconds, val_seconds]
    
    Returns:
        DatasetDict with 'train', 'test', and 'validation' splits
    """
    train_duration, test_duration, val_duration = time_durations
    
    # Get timestamps from dataset
    timestamps = [formatted_timestamp_to_unix(t) for t in dataset['start_time']]
    
    if not timestamps:
        raise ValueError("Dataset is empty")
    
    # Find the start time (first record)
    start_time = timestamps[0]
    
    # Calculate cutoff times
    train_cutoff = start_time + train_duration
    test_cutoff = train_cutoff + test_duration
    val_cutoff = test_cutoff + val_duration
    
    # Find split indices
    train_end_idx = 0
    test_end_idx = 0
    
    for i, ts in enumerate(timestamps):
        if ts < train_cutoff:
            train_end_idx = i + 1
        if ts < test_cutoff:
            test_end_idx = i + 1
    
    # Handle edge cases
    total_size = len(dataset)
    train_end_idx = min(train_end_idx, total_size)
    test_end_idx = min(test_end_idx, total_size)
    
    # Create splits using select
    train_indices = list(range(0, train_end_idx))
    test_indices = list(range(train_end_idx, test_end_idx))
    val_indices = list(range(test_end_idx, total_size))
    
    # Format times for display
    def format_duration(seconds: int) -> str:
        if seconds >= 7 * 24 * 3600 and seconds % (7 * 24 * 3600) == 0:
            return f"{seconds // (7 * 24 * 3600)}wk"
        elif seconds >= 24 * 3600 and seconds % (24 * 3600) == 0:
            return f"{seconds // (24 * 3600)}d"
        else:
            return f"{seconds // 3600}hr"
    
    print(f"\nTime-based split:")
    print(f"  Data spans: {datetime.fromtimestamp(timestamps[0])} to {datetime.fromtimestamp(timestamps[-1])}")
    print(f"  Train: first {format_duration(train_duration)} ({len(train_indices)} samples)")
    print(f"  Test: next {format_duration(test_duration)} ({len(test_indices)} samples)")
    print(f"  Validation: next {format_duration(val_duration)} ({len(val_indices)} samples)")
    
    if len(val_indices) == 0:
        print(f"  [Warning] Validation set is empty - data may not span the full duration")
    if len(test_indices) == 0:
        print(f"  [Warning] Test set is empty - data may not span the full duration")
    
    return DatasetDict({
        'train': dataset.select(train_indices) if train_indices else dataset.select([]),
        'test': dataset.select(test_indices) if test_indices else dataset.select([]),
        'validation': dataset.select(val_indices) if val_indices else dataset.select([])
    })


def parse_time_duration(duration_str: str) -> int:
    """
    Parse time duration string into seconds.
    
    Supports:
    - Weeks: 2wk, 1wk
    - Days: 14d, 7d
    - Hours: 24hr, 48hr
    
    Returns duration in seconds.
    """
    duration_str = duration_str.strip().lower()
    
    # Try weeks
    match = re.match(r'^(\d+)wk$', duration_str)
    if match:
        return int(match.group(1)) * 7 * 24 * 60 * 60
    
    # Try days
    match = re.match(r'^(\d+)d$', duration_str)
    if match:
        return int(match.group(1)) * 24 * 60 * 60
    
    # Try hours
    match = re.match(r'^(\d+)hr$', duration_str)
    if match:
        return int(match.group(1)) * 60 * 60
    
    raise ValueError(f"Invalid time duration format: {duration_str}. Use Xwk, Xd, or Xhr")


def is_time_based_split(split_str: str) -> bool:
    """Check if split string is time-based (e.g., '2wk,1wk,1wk') vs ratio-based (e.g., '0.5,0.25,0.25')."""
    parts = split_str.split(',')
    if len(parts) != 3:
        return False
    # Check if any part contains time unit suffixes
    return any(re.match(r'^\s*\d+(wk|d|hr)\s*$', p, re.IGNORECASE) for p in parts)


def parse_time_durations(duration_str: str) -> List[int]:
    """Parse split duration string like '2wk,1wk,1wk' into list of seconds."""
    try:
        durations = [parse_time_duration(x) for x in duration_str.split(',')]
        if len(durations) != 3:
            raise ValueError("Must provide exactly 3 durations")
        return durations
    except Exception as e:
        raise ValueError(f"Invalid time durations format: {e}")


def parse_split_ratios(ratio_str: str) -> List[float]:
    """Parse split ratio string like '0.5,0.25,0.25' into list of floats."""
    try:
        ratios = [float(x.strip()) for x in ratio_str.split(',')]
        if len(ratios) != 3:
            raise ValueError("Must provide exactly 3 ratios")
        if any(r < 0 or r > 1 for r in ratios):
            raise ValueError("All ratios must be between 0 and 1")
        return ratios
    except Exception as e:
        raise ValueError(f"Invalid split ratios format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSONL data to HuggingFace Dataset with train/test/validation splits'
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
    parser.add_argument(
        '--split-ratios',
        type=str,
        default='0.5,0.25,0.25',
        help='Split ratios for train,test,validation. Supports two formats:\n'
             '  Ratio-based: 0.5,0.25,0.25 (default)\n'
             '  Time-based: 2wk,1wk,1wk or 14d,7d,7d or 48hr,24hr,24hr\n'
             'Time-based splits are sequential based on timestamps in the data.'
    )
    parser.add_argument(
        '--hash-cache',
        type=str,
        default=None,
        help='Path to hash cache JSON file for deduplication (format 2 only). '
             'Use the same cache that was used during processing to ensure correct alignment.'
    )
    parser.add_argument(
        '--dedupe-threshold',
        type=int,
        default=1,
        help='Hamming distance threshold for deduplication (default: 1). '
             'Must match the threshold used during processing.'
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    output_dir = Path(args.output_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    # Detect split type and parse accordingly
    use_time_based_split = is_time_based_split(args.split_ratios)
    
    if use_time_based_split:
        time_durations = parse_time_durations(args.split_ratios)
        print(f"Using time-based splits - train: {args.split_ratios.split(',')[0].strip()}, "
              f"test: {args.split_ratios.split(',')[1].strip()}, "
              f"validation: {args.split_ratios.split(',')[2].strip()}")
    else:
        split_ratios = parse_split_ratios(args.split_ratios)
        print(f"Using split ratios - train: {split_ratios[0]}, test: {split_ratios[1]}, validation: {split_ratios[2]}")

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

    # Load hash cache if provided
    hash_map = None
    if args.hash_cache:
        hash_map = load_hash_cache(args.hash_cache)
        if hash_map is None:
            print("[Warning] Could not load hash cache, proceeding without deduplication")

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
        records = process_format2(jsonl_path, img_dir, hash_map, args.dedupe_threshold)

    print(f"Processed {len(records)} records")

    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    full_dataset = create_hf_dataset(records)

    # Split dataset
    print("Splitting dataset...")
    if use_time_based_split:
        dataset = split_dataset_by_time(full_dataset, time_durations)
    else:
        dataset = split_dataset(full_dataset, split_ratios)

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    print("\nDataset saved successfully!")
    print("\nDataset info:")
    print(dataset)
    print(f"\nTrain examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Validation examples: {len(dataset['validation'])}")


if __name__ == '__main__':
    main()
