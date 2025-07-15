import json
from pathlib import Path
from aggregate.logs import main as aggregate_logs
from aggregate.video import convert_to_video, render_images
from analyze.plot_raw_logs import plot_interactive


def list_sessions():
    logs_path = Path(__file__).parent / 'logs'
    if not logs_path.exists():
        print("Logs directory not found!")
        return []

    sessions = [d for d in logs_path.iterdir() if d.is_dir()]
    sessions.sort()

    if not sessions:
        print("No session directories found!")
        return []

    print("\nAvailable sessions:")
    for i, session in enumerate(sessions, 1):
        print(f"{i}. {session.name}")

    return sessions


def get_session_choice(sessions):
    while True:
        try:
            choice = input(f"\nSelect session (1-{len(sessions)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(sessions):
                return sessions[index]
            else:
                print(f"Please enter a number between 1 and {len(sessions)}")
        except ValueError:
            print("Please enter a valid number")


def show_actions_menu():
    print("\nAvailable actions:")
    print("1. Aggregate logs")
    print("2. Plot raw logs")
    print("3. Convert to video")
    print("4. Visualize images")
    print("5. Set percentile (default 95)")
    print("\nYou can select multiple actions (e.g., '134' for actions 1, 3, and 4)")


def get_actions_choice():
    while True:
        choice = input("\nSelect action(s): ").strip()
        if choice and all(c in '12345' for c in choice):
            return [int(c) for c in choice]
        else:
            print("Please enter valid action numbers (1-5)")


def get_percentile():
    while True:
        try:
            percentile = input("\nEnter percentile (default 95): ").strip()
            if not percentile:
                return 95
            value = int(percentile)
            if 1 <= value <= 100:
                return value
            else:
                print("Please enter a percentile between 1 and 100")
        except ValueError:
            print("Please enter a valid number")


def aggregate_logs_action(session_path, percentile):
    print(f"\nAggregating logs for {session_path.name} (percentile: {percentile})...")
    try:
        aggregated_logs = aggregate_logs(session_path, percentile)
        output_file = session_path / f'aggregated_logs_{percentile}.json'
        with open(output_file, 'w') as f:
            json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)
        print(f"✓ Aggregated logs saved to {output_file}")
        return aggregated_logs
    except Exception as e:
        print(f"✗ Error aggregating logs: {e}")
        return None


def plot_raw_logs_action(session_path):
    print(f"\nPlotting raw logs for {session_path.name}...")
    try:
        events_file = session_path / 'events.jsonl'
        if not events_file.exists():
            print(f"✗ Events file not found: {events_file}")
            return
        plot_interactive(events_file)
        print("✓ Interactive plot generated")
    except Exception as e:
        print(f"✗ Error plotting raw logs: {e}")


def convert_to_video_action(aggregated_logs, session_path):
    print(f"\nConverting to video for {session_path.name}...")
    try:
        if aggregated_logs is None:
            print("✗ No aggregated logs available. Please run 'Aggregate logs' first.")
            return

        seconds_per_frame = input("Enter seconds per frame (default 1): ").strip()
        seconds_per_frame = float(seconds_per_frame) if seconds_per_frame else 1.0

        convert_to_video(aggregated_logs, "event_logs_video", session_path,
                         should_annotate=True, seconds_per_frame=seconds_per_frame)
        print("✓ Video conversion completed")
    except Exception as e:
        print(f"✗ Error converting to video: {e}")


def visualize_images_action(aggregated_logs, session_path, percentile):
    print(f"\nVisualizing images for {session_path.name}...")
    try:
        if aggregated_logs is None:
            print("✗ No aggregated logs available. Please run 'Aggregate logs' first.")
            return

        visualization_path = session_path / f"agg_{percentile}_visualizations"
        visualization_path.mkdir(parents=True, exist_ok=True)
        render_images(aggregated_logs, visualization_path)
        print(f"✓ Images rendered to {visualization_path}")
    except Exception as e:
        print(f"✗ Error visualizing images: {e}")


def main():
    print("=== Interactive Log Processing Tool ===")

    sessions = list_sessions()
    if not sessions:
        return

    session_path = get_session_choice(sessions)
    print(f"\nSelected session: {session_path.name}")

    show_actions_menu()
    actions = get_actions_choice()

    percentile = 95
    if 5 in actions:
        percentile = get_percentile()

    aggregated_logs = None

    for action in actions:
        if action == 1:
            aggregated_logs = aggregate_logs_action(session_path, percentile)
        elif action == 2:
            plot_raw_logs_action(session_path)
        elif action == 3:
            convert_to_video_action(aggregated_logs, session_path)
        elif action == 4:
            visualize_images_action(aggregated_logs, session_path, percentile)

    print("\n=== Processing completed ===")


if __name__ == "__main__":
    main()
