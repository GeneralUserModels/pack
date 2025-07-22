import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from aggregate.logs import main as aggregate_logs
from aggregate.video import convert_to_video, render_images
from analyze.plot_raw_logs import plot_interactive

print_lock = threading.Lock()


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
    print("0. All sessions")
    for i, session in enumerate(sessions, 1):
        print(f"{i}. {session.name}")

    return sessions


def get_session_choice(sessions):
    while True:
        try:
            choice = input(f"\nSelect session (0 for all, 1-{len(sessions)}): ").strip()
            index = int(choice)
            if index == 0:
                return "all"
            elif 1 <= index <= len(sessions):
                return sessions[index - 1]
            else:
                print(f"Please enter a number between 0 and {len(sessions)}")
        except ValueError:
            print("Please enter a valid number")


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


def get_num_workers(num_sessions):
    while True:
        try:
            workers = input(f"\nEnter number of workers (1-{num_sessions}, default {min(4, num_sessions)}): ").strip()
            if not workers:
                return min(4, num_sessions)
            value = int(workers)
            if 1 <= value <= num_sessions:
                return value
            else:
                print(f"Please enter a number between 1 and {num_sessions}")
        except ValueError:
            print("Please enter a valid number")
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
    with print_lock:
        print(f"\nAggregating logs for {session_path.name} (percentile: {percentile})...")
    try:
        aggregated_logs = aggregate_logs(session_path, percentile)
        output_file = session_path / f'aggregated_logs_{percentile}.json'
        with open(output_file, 'w') as f:
            json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)
        with print_lock:
            print(f"✓ Aggregated logs saved to {output_file}")
        return aggregated_logs
    except Exception as e:
        with print_lock:
            print(f"✗ Error aggregating logs: {e}")
        return None


def plot_raw_logs_action(session_path, percentile):
    with print_lock:
        print(f"\nPlotting raw logs for {session_path.name}...")
    try:
        events_file = session_path / 'events.jsonl'
        if not events_file.exists():
            with print_lock:
                print(f"✗ Events file not found: {events_file}")
            return
        plot_interactive(events_file, percentile)
        with print_lock:
            print("✓ Interactive plot generated")
    except Exception as e:
        with print_lock:
            print(f"✗ Error plotting raw logs: {e}")


def convert_to_video_action(aggregated_logs, session_path, percentile):
    with print_lock:
        print(f"\nConverting to video for {session_path.name}...")
    try:
        if aggregated_logs is None:
            with print_lock:
                print("✗ No aggregated logs available. Please run 'Aggregate logs' first.")
            return

        convert_to_video(
            aggregated_logs,
            "event_logs_video",
            session_path,
            should_annotate=True,
            seconds_per_frame=1.0,
            percentile=percentile
        )
        with print_lock:
            print("✓ Video conversion completed")
    except Exception as e:
        with print_lock:
            print(f"✗ Error converting to video: {e}")


def visualize_images_action(aggregated_logs, session_path, percentile):
    with print_lock:
        print(f"\nVisualizing images for {session_path.name}...")
    try:
        if aggregated_logs is None:
            with print_lock:
                print("✗ No aggregated logs available. Please run 'Aggregate logs' first.")
            return

        visualization_path = session_path / f"agg_{percentile}_visualizations"
        visualization_path.mkdir(parents=True, exist_ok=True)
        render_images(aggregated_logs, visualization_path)
        with print_lock:
            print(f"✓ Images rendered to {visualization_path}")
    except Exception as e:
        with print_lock:
            print(f"✗ Error visualizing images: {e}")


def process_single_session(session_path, actions, percentile, session_num=None, total_sessions=None):
    """Process a single session with the given actions and percentile"""
    if session_num and total_sessions:
        header = f"[{session_num}/{total_sessions}] Processing session: {session_path.name}"
    else:
        header = f"Processing session: {session_path.name}"

    with print_lock:
        print(f"\n{'=' * len(header)}")
        print(header)
        print(f"{'=' * len(header)}")

    aggregated_logs = None

    for action in actions:
        if action == 1:
            aggregated_logs = aggregate_logs_action(session_path, percentile)
        elif action == 2:
            plot_raw_logs_action(session_path, percentile)
        elif action == 3:
            convert_to_video_action(aggregated_logs, session_path, percentile)
        elif action == 4:
            visualize_images_action(aggregated_logs, session_path, percentile)

    return session_path.name


def process_all_sessions_parallel(sessions, actions, percentile, num_workers):
    print(f"\nProcessing {len(sessions)} sessions with {num_workers} workers and percentile: {percentile}")

    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_session = {
            executor.submit(
                process_single_session,
                session,
                actions,
                percentile,
                i + 1,
                len(sessions)
            ): session
            for i, session in enumerate(sessions)
        }

        for future in as_completed(future_to_session):
            session = future_to_session[future]
            try:
                result = future.result()
                completed += 1
                with print_lock:
                    print(f"\n✓ Completed session {result} ({completed}/{len(sessions)})")
            except Exception as e:
                completed += 1
                with print_lock:
                    print(f"\n✗ Failed to process session {session.name}: {e} ({completed}/{len(sessions)})")


def main():
    print("=== Interactive Log Processing Tool ===")

    sessions = list_sessions()
    if not sessions:
        return

    session_choice = get_session_choice(sessions)

    if session_choice == "all":
        print(f"\nSelected: All sessions ({len(sessions)} sessions)")
        num_workers = get_num_workers(len(sessions))
    else:
        print(f"\nSelected session: {session_choice.name}")
        num_workers = None

    show_actions_menu()
    actions = get_actions_choice()

    percentile = 95
    if 5 in actions:
        percentile = get_percentile()
        actions = [a for a in actions if a != 5]

    if session_choice == "all":
        process_all_sessions_parallel(sessions, actions, percentile, num_workers)
    else:
        process_single_session(session_choice, actions, percentile)

    print("\n=== Processing completed ===")


if __name__ == "__main__":
    main()
