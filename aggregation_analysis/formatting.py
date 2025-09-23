import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries"""
    data = []
    base_dir = Path(__file__).parent / "session_6"
    file_path = base_dir / filepath
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist. Returning empty list.")
        return data
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def parse_timestamp_to_unix(timestamp_val) -> Optional[float]:
    """
    Convert timestamp string/number to unix timestamp (float).
    Handles:
      - already-numeric values (int/float or numeric string)
      - formatted strings like "2025-09-18_23-16-24-001956"
    Returns None if parsing fails.
    """
    if timestamp_val is None:
        return None

    # If it's already numeric, return float
    if isinstance(timestamp_val, (int, float)):
        try:
            return float(timestamp_val)
        except Exception:
            return None

    # If it's a numeric string, convert
    if isinstance(timestamp_val, str):
        stripped = timestamp_val.strip()
        # quick reject if obviously placeholder-like
        if stripped == "":
            return None

        # Try numeric conversion first (some JSONL might already store a stringified float)
        try:
            return float(stripped)
        except Exception:
            pass

        # Try known datetime formats
        for fmt in ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d_%H-%M-%S"):
            try:
                dt = datetime.strptime(stripped, fmt)
                return dt.timestamp()
            except Exception:
                continue

    # Unknown format
    return None


def map_events_by_intervals(
    events_df: pd.DataFrame,
    similarities_df: pd.DataFrame,
    include_eq_current: bool = False,
    include_after_last: bool = True
) -> List[List[Dict[str, Any]]]:
    """
    Map events to similarity intervals:
      for each similarity row i with timestamp t_i, find events with
      (t_i < event_ts < t_{i+1})  (default)

    Returns list-of-lists where mapped[i] corresponds to similarities_df.iloc[i].
    """
    # If similarities_df is empty return empty list
    if similarities_df is None or len(similarities_df) == 0:
        return []

    # Ensure columns exist
    if 'unix_timestamp' not in similarities_df.columns:
        raise ValueError("similarities_df must contain 'unix_timestamp' column")

    # If events_df empty return empty lists for all similarities
    if events_df is None or len(events_df) == 0:
        return [[] for _ in range(len(similarities_df))]

    # Ensure sorted ascending by unix_timestamp
    similarities_df = similarities_df.sort_values('unix_timestamp').reset_index(drop=True)
    events_df = events_df.sort_values('unix_timestamp').reset_index(drop=True)

    # compute next timestamps
    similarities_df['next_unix_timestamp'] = similarities_df['unix_timestamp'].shift(-1)

    mapped: List[List[Dict[str, Any]]] = []
    for idx, sim in similarities_df.iterrows():
        t_i = float(sim['unix_timestamp'])
        next_t = sim['next_unix_timestamp']

        if pd.isna(next_t):
            # Last similarity
            if include_after_last:
                if include_eq_current:
                    sel = events_df['unix_timestamp'] >= t_i
                else:
                    sel = events_df['unix_timestamp'] > t_i
            else:
                sel = pd.Series(False, index=events_df.index)
        else:
            if include_eq_current:
                left_sel = events_df['unix_timestamp'] >= t_i
            else:
                left_sel = events_df['unix_timestamp'] > t_i
            right_sel = events_df['unix_timestamp'] < float(next_t)
            sel = left_sel & right_sel

        # select and keep order
        selected = events_df[sel].copy().sort_values('unix_timestamp')

        # convert to the event dict format
        events_list: List[Dict[str, Any]] = []
        for _, event in selected.iterrows():
            # skip poll if desired (remove this check to include poll events)
            if event.get("event_type") == "poll":
                continue

            event_dict = {
                'event_type': event.get('event_type'),
                'relative_time': event['unix_timestamp'] - t_i,
                'details': event.get('details', {}) if isinstance(event.get('details', {}), dict) else {},
                'cursor_pos': event.get('cursor_pos', []) if event.get('cursor_pos') is not None else [],
                'monitor_id': (event.get('monitor') or {}).get('monitor_id') if isinstance(event.get('monitor'), dict) else None
            }
            events_list.append(event_dict)

        mapped.append(events_list)

    return mapped


def create_regression_dataset(
    events_file: str = 'events.jsonl',
    similarities_file: str = 'img_similarities.jsonl',
    labels_file: str = 'manual_labels.jsonl',
    output_file: str = 'regression_dataset.json',
    time_window: float = 5.0  # kept for API compatibility but unused when mapping by intervals
):
    """
    Create regression dataset mapping image similarities to save labels.

    This version maps events to non-overlapping intervals between successive similarity timestamps:
      events with t_i < event_ts < t_{i+1} are assigned to similarity i.
    For the last similarity, events after it are included if include_after_last=True.
    """

    print("Loading data...")

    # Load all data
    events_data = load_jsonl(events_file)
    similarities_data = load_jsonl(similarities_file)
    labels_data = load_jsonl(labels_file)

    # Prepare DataFrames
    events_df = pd.DataFrame(events_data) if events_data else pd.DataFrame()
    similarities_df = pd.DataFrame(similarities_data) if similarities_data else pd.DataFrame()
    labels_df = pd.DataFrame(labels_data) if labels_data else pd.DataFrame()

    # Parse/ensure unix_timestamp for events
    if not events_df.empty:
        if 'unix_timestamp' in events_df.columns:
            events_df['unix_timestamp'] = events_df['unix_timestamp'].apply(parse_timestamp_to_unix)
        elif 'timestamp' in events_df.columns:
            events_df['unix_timestamp'] = events_df['timestamp'].apply(parse_timestamp_to_unix)
        else:
            # No timestamp column — set empty unix_timestamp so dropna will remove rows
            events_df['unix_timestamp'] = None

        events_df = events_df.dropna(subset=['unix_timestamp'])
        events_df['unix_timestamp'] = events_df['unix_timestamp'].astype(float)

    # Parse/ensure unix_timestamp for similarities
    if not similarities_df.empty:
        if 'unix_timestamp' in similarities_df.columns:
            similarities_df['unix_timestamp'] = similarities_df['unix_timestamp'].apply(parse_timestamp_to_unix)
        elif 'timestamp' in similarities_df.columns:
            similarities_df['unix_timestamp'] = similarities_df['timestamp'].apply(parse_timestamp_to_unix)
        else:
            similarities_df['unix_timestamp'] = None

        similarities_df = similarities_df.dropna(subset=['unix_timestamp'])
        similarities_df['unix_timestamp'] = similarities_df['unix_timestamp'].astype(float)

    # Build label lookup dict: unix_timestamp (float) -> bool(should_save)
    save_labels_dict: Dict[float, bool] = {}
    if not labels_df.empty:
        # ensure unix_timestamp column exists
        if 'unix_timestamp' not in labels_df.columns:
            if 'timestamp' in labels_df.columns:
                labels_df['unix_timestamp'] = labels_df['timestamp'].apply(parse_timestamp_to_unix)
            elif 'filename' in labels_df.columns and 'timestamp' in labels_df.columns:
                labels_df['unix_timestamp'] = labels_df['timestamp'].apply(parse_timestamp_to_unix)
            else:
                labels_df['unix_timestamp'] = None

        labels_df = labels_df.dropna(subset=['unix_timestamp'])
        for _, row in labels_df.iterrows():
            try:
                key = float(row['unix_timestamp'])
                save_labels_dict[key] = bool(row.get('should_save', False))
            except Exception:
                continue

    print(
        f"Loaded {len(events_df) if not events_df.empty else 0} events, "
        f"{len(similarities_df) if not similarities_df.empty else 0} similarities, "
        f"{len(labels_df) if not labels_df.empty else 0} labels"
    )

    regression_data: List[Dict[str, Any]] = []

    # If there are no similarities, nothing to map
    if similarities_df.empty:
        print("No similarities found — exiting without creating dataset.")
        return regression_data

    # Sort dataframes
    similarities_df = similarities_df.sort_values('unix_timestamp').reset_index(drop=True)
    if not events_df.empty:
        events_df = events_df.sort_values('unix_timestamp').reset_index(drop=True)

    # Map events into non-overlapping intervals between similarity timestamps
    mapped_events = map_events_by_intervals(events_df, similarities_df, include_eq_current=False, include_after_last=True)

    print("Creating regression dataset (mapping events between successive similarity timestamps)...")

    # Iterate similarities and the mapped events together
    for idx, similarity in similarities_df.reset_index(drop=True).iterrows():
        target_timestamp = float(similarity['unix_timestamp'])
        events_for_similarity = mapped_events[idx] if idx < len(mapped_events) else []

        # Find the corresponding label (closest label timestamp within a tolerance)
        should_save = False
        min_time_diff = float('inf')
        for label_ts, label_val in save_labels_dict.items():
            time_diff = abs(label_ts - target_timestamp)
            if time_diff < min_time_diff and time_diff <= 1.0:  # 1 second tolerance (adjustable)
                min_time_diff = time_diff
                should_save = label_val

        data_point = {
            "input": {
                "timestamp": target_timestamp,
                "ssim_similarity": similarity.get('ssim_similarity', 0.0),
                "monitor_id": similarity.get('monitor_id'),
                "image_width": int(similarity.get('image_width', 0) or 0),
                "image_height": int(similarity.get('image_height', 0) or 0),
                "events": events_for_similarity,
                "num_events": len(events_for_similarity)
            },
            "output": should_save
        }
        regression_data.append(data_point)

        # Debugging print to confirm mapping (comment out in production)
        print(f"Similarity idx={idx}, ts={target_timestamp}: mapped {len(events_for_similarity)} events")

    # Save the formatted data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(regression_data, f, indent=2)

    print(f"Regression dataset saved to {output_file}")

    # Print statistics
    total_samples = len(regression_data)
    positive_samples = sum(1 for item in regression_data if item['output'])
    negative_samples = total_samples - positive_samples

    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (should_save=True): {positive_samples}")
    print(f"Negative samples (should_save=False): {negative_samples}")
    if total_samples > 0:
        print(f"Class balance: {positive_samples / total_samples:.2%} positive")

    # Event statistics
    all_event_types = set()
    total_events = 0
    for item in regression_data:
        for event in item['input']['events']:
            all_event_types.add(event['event_type'])
        total_events += len(item['input']['events'])

    if total_samples > 0:
        print(f"Average events per sample: {total_events / total_samples:.2f}")
    else:
        print("Average events per sample: N/A (no samples)")
    print(f"Event types found: {sorted(list(all_event_types))}")

    # Show a sample
    if regression_data:
        print(f"\nSample data point:")
        sample = regression_data[0]
        print(f"Timestamp: {sample['input']['timestamp']}")
        print(f"SSIM similarity: {sample['input']['ssim_similarity']}")
        print(f"Number of events: {sample['input']['num_events']}")
        print(f"Should save: {sample['output']}")
        if sample['input']['events']:
            print(f"First event: {sample['input']['events'][0]}")

    return regression_data


def create_features_dataset(regression_data, output_file='features_dataset.json'):
    """
    Create a more ML-friendly version with engineered features
    """
    features_data = []

    for item in regression_data:
        input_data = item['input']
        events = input_data['events']

        # Engineer features from events
        event_features = {
            'num_events': len(events),
            'event_types': {},
            'time_span': 0,
            'has_keyboard': False,
            'has_mouse': False,
            'has_click': False,
            'cursor_movement': 0,
            'avg_time_to_event': 0
        }

        # Count event types
        for event_type in ['poll', 'keyboard_press', 'keyboard_release', 'mouse_move', 'mouse_click', 'mouse_down', 'mouse_up']:
            event_features['event_types'][f'count_{event_type}'] = 0

        if events:
            # Calculate event statistics
            event_times = [event['relative_time'] for event in events]
            event_features['time_span'] = max(event_times) - min(event_times) if len(event_times) > 1 else 0
            event_features['avg_time_to_event'] = sum(abs(t) for t in event_times) / len(event_times) if event_times else 0

            # Count event types and detect patterns
            cursor_positions = []
            for event in events:
                event_type = event['event_type']
                if f'count_{event_type}' in event_features['event_types']:
                    event_features['event_types'][f'count_{event_type}'] += 1

                # Pattern detection
                if 'keyboard' in event_type:
                    event_features['has_keyboard'] = True
                if 'mouse' in event_type:
                    event_features['has_mouse'] = True
                if 'click' in event_type or 'down' in event_type or 'up' in event_type:
                    event_features['has_click'] = Tru
