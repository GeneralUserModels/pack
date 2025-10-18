from pathlib import Path
from typing import List, Dict, Any, Optional
import json


def match_captions_with_events(
    captions_path: Path,
    aggregations_path: Path,
    output_path: Path,
    fps: int = 1
) -> List[Dict[str, Any]]:
    """
    Match captions with aggregated events based on timestamps.

    Args:
        captions_path: Path to captions.jsonl
        aggregations_path: Path to aggregations.jsonl
        output_path: Path to save matched_captions.jsonl
        fps: Frames per second used in video creation

    Returns:
        List of matched caption-event objects
    """
    # Load captions
    captions = []
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                captions.append(json.loads(line))

    # Load aggregations
    aggregations = []
    with open(aggregations_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                agg = json.loads(line)
                aggregations.append(agg)

    # Sort aggregations by timestamp
    aggregations.sort(key=lambda x: x.get('timestamp', 0))

    if not aggregations:
        print("[Matcher] Warning: No aggregations found")
        return []

    # Get first aggregation timestamp (video start time)
    first_timestamp = aggregations[0].get('timestamp', 0)

    print(f"[Matcher] Video start time: {first_timestamp}")
    print(f"[Matcher] Total aggregations: {len(aggregations)}")
    print(f"[Matcher] FPS: {fps}")

    # Match captions with events
    matched_data = []

    for caption in captions:
        # Convert MM:SS to seconds
        start_seconds = caption['start_seconds']
        end_seconds = caption['end_seconds']

        # Convert video time to aggregation indices
        # Each aggregation represents 1 frame, so index = seconds * fps
        start_index = int(start_seconds * fps)
        end_index = int(end_seconds * fps)

        # Clamp to valid range
        start_index = max(0, min(start_index, len(aggregations) - 1))
        end_index = max(start_index, min(end_index, len(aggregations) - 1))

        print(f"[Matcher] Caption '{caption['caption'][:50]}...' -> indices [{start_index}, {end_index}]")

        # Get aggregations in this range
        matched_aggs = aggregations[start_index:end_index + 1]

        if not matched_aggs:
            # No events matched, but still save the caption
            matched_entry = {
                "start_time": first_timestamp + start_seconds,
                "end_time": first_timestamp + end_seconds,
                "start_index": start_index,
                "end_index": end_index,
                "img": None,
                "caption": caption['caption'],
                "raw_events": [],
                "num_aggregations": 0,
                "start_formatted": caption['start'],
                "end_formatted": caption['end'],
            }
        else:
            # Get first and last aggregation for time and image
            first_agg = matched_aggs[0]
            last_agg = matched_aggs[-1]

            # Concatenate all events from matched aggregations
            all_events = []
            for agg in matched_aggs:
                events = agg.get('events', [])
                all_events.extend(events)

            matched_entry = {
                "start_time": first_agg.get('timestamp'),
                "end_time": last_agg.get('timestamp'),
                "start_index": start_index,
                "end_index": end_index,
                "img": first_agg.get('screenshot_path'),
                "caption": caption['caption'],
                "raw_events": all_events,
                "num_aggregations": len(matched_aggs),
                "start_formatted": caption['start'],
                "end_formatted": caption['end'],
            }

        matched_data.append(matched_entry)

    # Save matched data
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in matched_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[Matcher] Saved {len(matched_data)} matched entries to {output_path}")

    return matched_data


def create_matched_captions_for_session(session_dir: Path, fps: int = 1) -> Optional[Path]:
    """
    Create matched_captions.jsonl for a session directory.

    Args:
        session_dir: Path to session directory
        fps: Frames per second used in video creation

    Returns:
        Path to created matched_captions.jsonl or None if failed
    """
    captions_path = session_dir / "captions.jsonl"
    aggregations_path = session_dir / "aggregations.jsonl"
    output_path = session_dir / "matched_captions.jsonl"

    if not captions_path.exists():
        print(f"[Matcher] Warning: {captions_path} not found")
        return None

    if not aggregations_path.exists():
        print(f"[Matcher] Warning: {aggregations_path} not found")
        return None

    match_captions_with_events(captions_path, aggregations_path, output_path, fps)

    return output_path
