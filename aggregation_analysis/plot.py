import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np


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
        # Handle other formats if needed
        return None


def plot_jsonl_data():
    # Load all JSONL files
    print("Loading JSONL files...")

    # Load events data
    events_data = load_jsonl('events.jsonl')
    events_df = pd.DataFrame(events_data)

    # Load image similarities data
    similarities_data = load_jsonl('img_similarities.jsonl')
    similarities_df = pd.DataFrame(similarities_data)

    # Load manual labels data
    labels_data = load_jsonl('manual_labels.jsonl')
    labels_df = pd.DataFrame(labels_data)

    print(f"Loaded {len(events_df)} events, {len(similarities_df)} similarities, {len(labels_df)} labels")

    # Process events data
    events_df['unix_timestamp'] = events_df['timestamp'].apply(parse_timestamp_to_unix)
    events_df = events_df.dropna(subset=['unix_timestamp'])
    events_df['datetime'] = pd.to_datetime(events_df['unix_timestamp'], unit='s')

    # Process similarities data (already has unix_timestamp)
    similarities_df['datetime'] = pd.to_datetime(similarities_df['unix_timestamp'], unit='s')

    # Process labels data (already has unix_timestamp)
    labels_df['datetime'] = pd.to_datetime(labels_df['unix_timestamp'], unit='s')
    save_labels = labels_df[labels_df['should_save'] == True]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Events scatter plot
    event_types = events_df['event_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(event_types)))

    for i, event_type in enumerate(event_types):
        event_subset = events_df[events_df['event_type'] == event_type]
        ax1.scatter(event_subset['datetime'], [i] * len(event_subset),
                    label=event_type, alpha=0.7, s=30, color=colors[i])

    ax1.set_ylabel('Event Types')
    ax1.set_title('Events Timeline')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(range(len(event_types)))
    ax1.set_yticklabels(event_types)

    # Plot 2: Image similarities line plot
    similarities_df_sorted = similarities_df.sort_values('unix_timestamp')
    ax2.plot(similarities_df_sorted['datetime'], similarities_df_sorted['ssim_similarity'],
             'b-', linewidth=1, alpha=0.7, label='SSIM Similarity')
    ax2.set_ylabel('SSIM Similarity')
    ax2.set_xlabel('Time')
    ax2.set_title('Image Similarity Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add vertical lines for manual labels where should_save == True
    for _, label in save_labels.iterrows():
        ax1.axvline(x=label['datetime'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axvline(x=label['datetime'], color='red', linestyle='--', alpha=0.7, linewidth=1)

    # Add legend for vertical lines
    if len(save_labels) > 0:
        ax1.axvline(x=save_labels.iloc[0]['datetime'], color='red', linestyle='--',
                    alpha=0.7, linewidth=1, label='Manual Save Labels')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax2.axvline(x=save_labels.iloc[0]['datetime'], color='red', linestyle='--',
                    alpha=0.7, linewidth=1, label='Manual Save Labels')
        ax2.legend()

    # Format x-axis
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=30))

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('jsonl_analysis_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'jsonl_analysis_plot.png'")

    # Show the plot
    plt.show()

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Time range: {events_df['datetime'].min()} to {events_df['datetime'].max()}")
    print(f"Event types: {list(event_types)}")
    print(f"Number of save labels: {len(save_labels)}")
    print(f"Average SSIM similarity: {similarities_df['ssim_similarity'].mean():.3f}")
    print(f"Min/Max SSIM similarity: {similarities_df['ssim_similarity'].min():.3f} / {similarities_df['ssim_similarity'].max():.3f}")


if __name__ == "__main__":
    plot_jsonl_data()
