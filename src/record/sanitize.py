from pathlib import Path
import json


def sanitize_aggregations(input_file: Path):
    """
    Sanitize aggregations.jsonl by properly aligning screenshot timestamps
    and redistributing ALL events to their correct time windows.
    """
    # move filename to filename_raw
    output_file = input_file.parent / "aggregations.jsonl"

    # Read all records
    records = []
    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)

    print(f"Total records read: {len(records)}")

    # Filter out records without screenshot_timestamp or screenshot_path
    valid_records = []
    for record in records:
        if (record.get('screenshot_timestamp') is not None and
                record.get('screenshot_path') is not None):
            valid_records.append(record)
        else:
            print(f"Skipping record with missing screenshot info: reason={record.get('reason')}")

    print(f"Valid records: {len(valid_records)}")

    # Sort by screenshot_timestamp
    valid_records.sort(key=lambda x: x['screenshot_timestamp'])

    # Collect ALL events from ALL records (including invalid ones)
    all_events = []
    for record in records:
        if 'events' in record and record['events']:
            all_events.extend(record['events'])

    print(f"Total events collected from all records: {len(all_events)}")

    # Sort all events by timestamp
    all_events.sort(key=lambda x: x.get('timestamp', 0))

    if all_events:
        print(f"Event timestamp range: {all_events[0].get('timestamp')} to {all_events[-1].get('timestamp')}")

    # Create sanitized timestamp pairs
    timestamp_pairs = []
    for i, record in enumerate(valid_records):
        start_timestamp = record['screenshot_timestamp']

        # Determine end timestamp
        if i < len(valid_records) - 1:
            end_timestamp = valid_records[i + 1]['screenshot_timestamp']
        else:
            # For last record, use its end_screenshot_timestamp or last event timestamp
            end_timestamp = record.get('end_screenshot_timestamp')
            if end_timestamp is None and all_events:
                # Use timestamp beyond last event
                end_timestamp = all_events[-1].get('timestamp', start_timestamp) + 1

        # Check if changed
        original_end = record.get('end_screenshot_timestamp')
        if original_end != end_timestamp:
            print(f"Record {i}: end_screenshot_timestamp changed from {original_end} to {end_timestamp}")

        timestamp_pairs.append({
            'record': record,
            'start': start_timestamp,
            'end': end_timestamp,
            'index': i
        })

    print(f"\nTimestamp pairs created: {len(timestamp_pairs)}")
    if timestamp_pairs:
        print(f"Coverage range: {timestamp_pairs[0]['start']} to {timestamp_pairs[-1]['end']}")

    # Redistribute ALL events to their correct pairs
    sanitized_records = []
    events_assigned = 0
    events_outside_range = 0

    for pair in timestamp_pairs:
        matching_events = []

        for event in all_events:
            event_timestamp = event.get('timestamp')

            if event_timestamp is None:
                continue

            if pair['start'] < event_timestamp <= pair['end']:
                matching_events.append(event)

        events_assigned += len(matching_events)

        # Create sanitized record
        sanitized_record = pair['record'].copy()
        sanitized_record['end_screenshot_timestamp'] = pair['end']
        sanitized_record['events'] = matching_events
        sanitized_record['num_events'] = len(matching_events)

        original_count = len(pair['record'].get('events', []))
        if len(matching_events) != original_count:
            print(f"Record {pair['index']} (reason={pair['record'].get('reason')}): "
                  f"Events changed from {original_count} to {len(matching_events)}")

        sanitized_records.append(sanitized_record)

    # Check for events outside the coverage range
    for event in all_events:
        event_timestamp = event.get('timestamp')
        if event_timestamp is None:
            continue

        if (event_timestamp < timestamp_pairs[0]['start'] or
                event_timestamp >= timestamp_pairs[-1]['end']):
            events_outside_range += 1

    # Write sanitized records
    with open(output_file, 'w') as f:
        for record in sanitized_records:
            f.write(json.dumps(record) + '\n')

    print(f"\n{'=' * 60}")
    print(f"Sanitization complete!")
    print(f"{'=' * 60}")
    print(f"Input records: {len(records)}")
    print(f"Valid records: {len(valid_records)}")
    print(f"Output records: {len(sanitized_records)}")
    print(f"Total events collected: {len(all_events)}")
    print(f"Events assigned to windows: {events_assigned}")
    print(f"Events outside coverage range: {events_outside_range}")

    print(f"\nSanitized file saved to: {output_file}")
