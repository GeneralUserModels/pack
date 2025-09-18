import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import datetime
import re
from collections import defaultdict
import argparse


def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")
    except ValueError:
        # Try without microseconds
        try:
            return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return None


def load_buffer_images(buffer_dir):
    """Load all buffer images and extract timestamps"""
    buffer_path = Path(buffer_dir)
    if not buffer_path.exists():
        print(f"Buffer directory not found: {buffer_dir}")
        return {}

    images = {}
    pattern = re.compile(r'buffer_(\d+)_(\d+\.\d+)\.jpg')

    for img_file in sorted(buffer_path.glob("buffer_*.jpg")):
        match = pattern.match(img_file.name)
        if match:
            monitor_id = int(match.group(1))
            timestamp = float(match.group(2))

            try:
                img = Image.open(img_file)
                # Convert to grayscale for SSIM calculation
                img_gray = img.convert('L')

                if monitor_id not in images:
                    images[monitor_id] = []
                images[monitor_id].append((timestamp, np.array(img_gray), img_file.name))
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")

    # Sort by timestamp for each monitor
    for monitor_id in images:
        images[monitor_id].sort(key=lambda x: x[0])

    return images


def load_logs(log_file):
    """Load and parse the JSONL log file"""
    logs = []
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return []

    with open(log_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                timestamp = parse_timestamp(log_entry['timestamp'])
                if timestamp:
                    log_entry['datetime'] = timestamp
                    # Convert datetime to unix timestamp for easier comparison
                    log_entry['unix_timestamp'] = timestamp.timestamp()
                    logs.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing log line: {e}")

    return sorted(logs, key=lambda x: x['unix_timestamp'])


def calculate_ssim_similarities(images):
    """Calculate SSIM similarities between consecutive images for each monitor"""
    similarities = {}

    for monitor_id, img_list in images.items():
        if len(img_list) < 2:
            continue

        similarities[monitor_id] = []

        for i in range(1, len(img_list)):
            prev_timestamp, prev_img, prev_name = img_list[i - 1]
            curr_timestamp, curr_img, curr_name = img_list[i]

            try:
                # Calculate SSIM
                similarity = ssim(prev_img, curr_img)
                similarities[monitor_id].append({
                    'timestamp': curr_timestamp,
                    'similarity': similarity,
                    'prev_image': prev_name,
                    'curr_image': curr_name
                })
            except Exception as e:
                print(f"Error calculating SSIM between {prev_name} and {curr_name}: {e}")

    return similarities


def save_percentile_images(similarities, buffer_dir, percentile_threshold, output_dir):
    """Save images that fall within the specified percentile to a separate folder"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    buffer_path = Path(buffer_dir)
    saved_count = 0

    for monitor_id, sim_data in similarities.items():
        if not sim_data:
            continue

        for sim_info in sim_data:
            if sim_info['similarity'] <= percentile_threshold:
                # Copy both the current and previous image
                curr_img_path = buffer_path / sim_info['curr_image']
                prev_img_path = buffer_path / sim_info['prev_image']

                if curr_img_path.exists():
                    dest_curr = output_path / f"monitor_{monitor_id}_{sim_info['curr_image']}"
                    Image.open(curr_img_path).save(dest_curr)
                    saved_count += 1

                if prev_img_path.exists():
                    dest_prev = output_path / f"monitor_{monitor_id}_{sim_info['prev_image']}"
                    if not dest_prev.exists():  # Avoid duplicates
                        Image.open(prev_img_path).save(dest_prev)
                        saved_count += 1

    print(f"Saved {saved_count} images to {output_path}")
    return saved_count


def create_visualization(similarities, logs, percentile_value=None, output_path=None):
    """Create comprehensive visualization of SSIM and log events"""

    # Calculate percentile threshold if provided
    percentile_threshold = None
    if percentile_value is not None:
        all_similarities = []
        for sim_data in similarities.values():
            all_similarities.extend([s['similarity'] for s in sim_data])

        if all_similarities:
            percentile_threshold = np.percentile(all_similarities, percentile_value)
            print(f"\n{percentile_value}th percentile SSIM value: {percentile_threshold:.4f}")

    # Define colors for different event types
    event_colors = {
        'keyboard_press': '#FF6B6B',
        'keyboard_release': '#4ECDC4',
        'mouse_down': '#45B7D1',
        'mouse_up': '#96CEB4',
        'mouse_move': '#FFEAA7',
        'mouse_scroll': '#DDA0DD',
        'poll': '#98D8C8',
    }

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
    fig.suptitle('Screenshot SSIM Analysis and Input Events', fontsize=16, fontweight='bold')

    # Plot 1: SSIM similarities over time
    for monitor_id, sim_data in similarities.items():
        if not sim_data:
            continue

        timestamps = [s['timestamp'] for s in sim_data]
        similarities_values = [s['similarity'] for s in sim_data]

        # Convert timestamps to datetime objects for plotting
        datetime_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

        # Plot regular points
        ax1.plot(datetime_timestamps, similarities_values,
                 marker='o', markersize=2, linewidth=1, alpha=0.7,
                 label=f'Monitor {monitor_id}')

        # Highlight percentile values if threshold is set
        if percentile_threshold is not None:
            percentile_mask = np.array(similarities_values) <= percentile_threshold
            if np.any(percentile_mask):
                percentile_timestamps = np.array(datetime_timestamps)[percentile_mask]
                percentile_similarities = np.array(similarities_values)[percentile_mask]

                ax1.scatter(percentile_timestamps, percentile_similarities,
                            c='red', s=20, alpha=0.8, zorder=5,
                            label=f'≤{percentile_value}th percentile' if monitor_id == list(similarities.keys())[0] else "")

    ax1.set_ylabel('SSIM Similarity', fontsize=12)
    ax1.set_title('Screenshot Similarity Over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Add horizontal line for percentile threshold
    if percentile_threshold is not None:
        ax1.axhline(y=percentile_threshold, color='red', linestyle='--', alpha=0.7,
                    label=f'{percentile_value}th percentile ({percentile_threshold:.4f})')
        ax1.legend()  # Refresh legend

    # Format x-axis for time
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Log events over time
    event_type_counts = defaultdict(list)
    event_timestamps = defaultdict(list)
    event_details = defaultdict(list)

    for log in logs:
        if 'datetime' in log:
            event_type = log.get('event_type', 'unknown')
            event_type_counts[event_type].append(1)
            event_timestamps[event_type].append(log['datetime'])

            # Extract meaningful details
            details = log.get('details', {})
            detail_str = ""
            if 'key' in details:
                detail_str = details['key']
            elif 'button' in details:
                detail_str = f"Button: {details['button']}"
            elif 'scroll' in details:
                detail_str = f"Scroll: {details['scroll']}"

            event_details[event_type].append(detail_str)

    # Plot events as scatter points
    y_positions = {}
    y_pos = 0

    for event_type in event_type_counts:
        color = event_colors.get(event_type, '#95A5A6')
        timestamps = event_timestamps[event_type]
        details = event_details[event_type]

        y_positions[event_type] = y_pos
        y_values = [y_pos] * len(timestamps)

        scatter = ax2.scatter(timestamps, y_values,
                              c=color, alpha=0.7, s=30, label=event_type)

        # Add text annotations for key details (sample every 5th event to avoid clutter)
        for i, (ts, detail) in enumerate(zip(timestamps, details)):
            if i % 5 == 0 and detail:  # Show every 5th event detail
                ax2.annotate(detail, (ts, y_pos),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8, rotation=45)

        y_pos += 1

    ax2.set_ylabel('Event Types', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_title('Input Events Timeline', fontsize=14)
    ax2.set_yticks(list(y_positions.values()))
    ax2.set_yticklabels(list(y_positions.keys()))
    ax2.grid(True, alpha=0.3)

    # Format x-axis for time
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Align time axes
    if similarities:
        # Get time range from similarities data
        all_timestamps = []
        for sim_data in similarities.values():
            all_timestamps.extend([s['timestamp'] for s in sim_data])

        if all_timestamps:
            start_time = datetime.datetime.fromtimestamp(min(all_timestamps))
            end_time = datetime.datetime.fromtimestamp(max(all_timestamps))

            ax1.set_xlim(start_time, end_time)
            ax2.set_xlim(start_time, end_time)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()

    # Print summary statistics
    print("\n=== ANALYSIS SUMMARY ===")
    total_percentile_images = 0

    for monitor_id, sim_data in similarities.items():
        if sim_data:
            similarities_values = [s['similarity'] for s in sim_data]
            print(f"\nMonitor {monitor_id}:")
            print(f"  Total image pairs: {len(similarities_values)}")
            print(f"  Average SSIM: {np.mean(similarities_values):.4f}")
            print(f"  Min SSIM: {np.min(similarities_values):.4f}")
            print(f"  Max SSIM: {np.max(similarities_values):.4f}")
            print(f"  Std SSIM: {np.std(similarities_values):.4f}")

            # Count percentile images
            if percentile_threshold is not None:
                percentile_count = sum(1 for s in sim_data if s['similarity'] <= percentile_threshold)
                total_percentile_images += percentile_count
                print(f"  Images in {percentile_value}th percentile: {percentile_count}")

            # Find lowest similarity changes (most significant changes)
            low_similarity_threshold = 0.8
            significant_changes = [s for s in sim_data if s['similarity'] < low_similarity_threshold]
            if significant_changes:
                print(f"  Significant changes (SSIM < {low_similarity_threshold}): {len(significant_changes)}")

    if percentile_threshold is not None:
        print(f"\nTotal images in {percentile_value}th percentile: {total_percentile_images}")

    print(f"\nTotal log events: {len(logs)}")
    for event_type, count in event_type_counts.items():
        print(f"  {event_type}: {len(count)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze screenshot SSIM and input events')
    parser.add_argument('session_dir', help='Session directory containing buffer_screenshots and events.jsonl')
    parser.add_argument('--output', '-o', help='Output path for the plot image')
    parser.add_argument('--percentile', '-p', type=float,
                        help='Percentile threshold for highlighting significant changes (e.g., 10 for bottom 10%%)')
    parser.add_argument('--save-percentile-images', '-s',
                        help='Directory to save images falling within the percentile')

    args = parser.parse_args()

    session_path = Path(args.session_dir)
    buffer_dir = session_path / "buffer_screenshots"
    log_file = session_path / "events.jsonl"

    print("Loading buffer images...")
    images = load_buffer_images(buffer_dir)
    print(f"Loaded images for {len(images)} monitors")

    for monitor_id, img_list in images.items():
        print(f"  Monitor {monitor_id}: {len(img_list)} images")

    print("\nLoading logs...")
    logs = load_logs(log_file)
    print(f"Loaded {len(logs)} log entries")

    print("\nCalculating SSIM similarities...")
    similarities = calculate_ssim_similarities(images)

    print("Creating visualization...")
    output_path = args.output or session_path / "ssim_analysis.png"
    create_visualization(similarities, logs, args.percentile, output_path)

    # Save percentile images if requested
    if args.percentile is not None and args.save_percentile_images:
        percentile_dir = Path(args.save_percentile_images)
        if not percentile_dir.is_absolute():
            percentile_dir = session_path / args.save_percentile_images

        print(f"\nSaving {args.percentile}th percentile images...")
        save_percentile_images(similarities, buffer_dir,
                               np.percentile([s['similarity'] for sim_data in similarities.values() for s in sim_data], args.percentile),
                               percentile_dir)


if __name__ == "__main__":
    main()
