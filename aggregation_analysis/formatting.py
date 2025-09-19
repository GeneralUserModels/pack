import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def load_jsonl(filepath):
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(Path(__file__).parent / "session" / filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def parse_timestamp_to_unix(timestamp_str):
    """Convert timestamp string to unix timestamp"""
    try:
        # Handle the format: "2025-09-18_23-16-24-001956"
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")
        return dt.timestamp()
    except ValueError:
        return None


def find_events_in_time_window(events_df, target_timestamp, window_seconds=5.0):
    """Find events within a time window around the target timestamp"""
    # Find events within the time window
    time_diff = abs(events_df['unix_timestamp'] - target_timestamp)
    events_in_window = events_df[time_diff <= window_seconds].copy()

    # Sort by timestamp to maintain chronological order
    events_in_window = events_in_window.sort_values('unix_timestamp')

    # Convert to list of event dictionaries with relative time
    events_list = []
    for _, event in events_in_window.iterrows():
        if event["event_type"] == "poll":
            continue
        event_dict = {
            'event_type': event['event_type'],
            'relative_time': event['unix_timestamp'] - target_timestamp,  # seconds before/after
            'details': event.get('details', {}),
            'cursor_pos': event.get('cursor_pos', []),
            'monitor_id': event.get('monitor', {}).get('monitor_id', None)
        }
        events_list.append(event_dict)

    return events_list


def create_regression_dataset(events_file='events.jsonl',
                              similarities_file='img_similarities.jsonl',
                              labels_file='manual_labels.jsonl',
                              output_file='regression_dataset.json',
                              time_window=5.0):
    """
    Create regression dataset mapping image similarities to save labels

    Args:
        events_file: Path to events JSONL file
        similarities_file: Path to image similarities JSONL file
        labels_file: Path to manual labels JSONL file
        output_file: Path to output JSON file
        time_window: Time window in seconds to look for events around each similarity timestamp
    """

    print("Loading data...")

    # Load all data
    events_data = load_jsonl(events_file)
    similarities_data = load_jsonl(similarities_file)
    labels_data = load_jsonl(labels_file)

    # Process events data
    events_df = pd.DataFrame(events_data)
    events_df['unix_timestamp'] = events_df['timestamp'].apply(parse_timestamp_to_unix)
    events_df = events_df.dropna(subset=['unix_timestamp'])

    # Process similarities data
    similarities_df = pd.DataFrame(similarities_data)

    # Process labels data - create a lookup dictionary
    labels_df = pd.DataFrame(labels_data)

    # Create a mapping from unix_timestamp to should_save
    # We'll use a tolerance for timestamp matching since they might not be exact
    save_labels_dict = {}
    for _, label in labels_df.iterrows():
        save_labels_dict[label['unix_timestamp']] = label['should_save']

    print(f"Loaded {len(events_df)} events, {len(similarities_df)} similarities, {len(labels_df)} labels")

    # Create regression dataset
    regression_data = []

    print("Creating regression dataset...")

    for _, similarity in similarities_df.iterrows():
        target_timestamp = similarity['unix_timestamp']

        # Find events around this timestamp
        events_in_window = find_events_in_time_window(events_df, target_timestamp, time_window)

        # Find the corresponding label (with tolerance for timestamp matching)
        should_save = False
        min_time_diff = float('inf')

        # Find the closest label timestamp within a reasonable tolerance (e.g., 1 second)
        for label_timestamp, label_value in save_labels_dict.items():
            time_diff = abs(label_timestamp - target_timestamp)
            if time_diff < min_time_diff and time_diff <= 1.0:  # 1 second tolerance
                min_time_diff = time_diff
                should_save = label_value

        # Create the input-output pair
        data_point = {
            "input": {
                "timestamp": target_timestamp,
                "ssim_similarity": similarity['ssim_similarity'],
                "monitor_id": similarity['monitor_id'],
                "image_width": similarity.get('image_width', 0),
                "image_height": similarity.get('image_height', 0),
                "events": events_in_window,
                "num_events": len(events_in_window)
            },
            "output": should_save
        }

        regression_data.append(data_point)

    # Save the formatted data
    with open(output_file, 'w') as f:
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
    print(f"Class balance: {positive_samples / total_samples:.2%} positive")

    # Event statistics
    all_event_types = set()
    total_events = 0
    for item in regression_data:
        for event in item['input']['events']:
            all_event_types.add(event['event_type'])
        total_events += len(item['input']['events'])

    print(f"Average events per sample: {total_events / total_samples:.2f}")
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
        for event_type in ['poll', 'keyboard_press', 'keyboard_release', 'mouse_move', 'mouse_click']:
            event_features['event_types'][f'count_{event_type}'] = 0

        if events:
            # Calculate event statistics
            event_times = [event['relative_time'] for event in events]
            event_features['time_span'] = max(event_times) - min(event_times)
            event_features['avg_time_to_event'] = sum(abs(t) for t in event_times) / len(event_times)

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
                if 'click' in event_type:
                    event_features['has_click'] = True

                # Cursor movement calculation
                if event['cursor_pos']:
                    cursor_positions.append(event['cursor_pos'])

            # Calculate cursor movement distance
            if len(cursor_positions) > 1:
                total_distance = 0
                for i in range(1, len(cursor_positions)):
                    dx = cursor_positions[i][0] - cursor_positions[i - 1][0]
                    dy = cursor_positions[i][1] - cursor_positions[i - 1][1]
                    total_distance += (dx**2 + dy**2)**0.5
                event_features['cursor_movement'] = total_distance

        # Combine all features
        feature_point = {
            "features": {
                **{k: v for k, v in input_data.items() if k != 'events'},
                **event_features
            },
            "output": item['output']
        }

        features_data.append(feature_point)

    # Save features dataset
    with open(output_file, 'w') as f:
        json.dump(features_data, f, indent=2)

    print(f"Features dataset saved to {output_file}")
    return features_data


if __name__ == "__main__":
    # Create the main regression dataset
    regression_data = create_regression_dataset(
        time_window=0.05, output_file='./aggregation_analysis/regression_dataset.json'
    )

    # Create a features-engineered version
    features_data = create_features_dataset(regression_data, output_file='./aggregation_analysis/features_dataset.json')

    print("\nData formatting complete!")
    print("Files created:")
    print("- regression_dataset.json: Raw format with events mapped to timestamps")
    print("- features_dataset.json: Engineered features for ML models")
