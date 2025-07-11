import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import tempfile
import argparse

PERCENTILE = 95

AGG_JSON = Path(__file__).parent.parent / "logs" / 'session_2025-07-11_04-03-47-306009' / f'aggregated_logs_{PERCENTILE}.json'
OUTPUT_DIR = Path(__file__).parent.parent / "logs" / 'session_2025-07-11_04-03-47-306009' / f'agg_{PERCENTILE}_visualizations'
BORDER_WIDTH = 10
CLICK_MARKER_RADIUS = 8

BUTTON_COLORS = {
    'Button.left': 'red',
    'left': 'red',
    'Button.right': 'blue',
    'right': 'blue',
    'Button.middle': 'green',
    'middle': 'green'
}

EVENT_COLORS = {
    'mouse_click': 'red',
    'mouse_press': 'orange',
    'mouse_release': 'yellow',
    'mouse_move': 'cyan',
    'scroll': 'purple',
    'keyboard_press': 'lightgreen',
    'keyboard_release': 'darkgreen',
    'poll': 'gray'
}


def screen_to_image_coords(screen_pos, monitor):
    """Convert screen coordinates to image coordinates"""
    x, y = screen_pos
    img_x = x - monitor['left']
    img_y = y - monitor['top']
    return img_x, img_y


def draw_border(img: Image.Image, color: str, width: int) -> Image.Image:
    """Draw a colored border around the image"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        rect = [i, i, w - i - 1, h - i - 1]
        draw.rectangle(rect, outline=color)
    return img


def draw_clicks(img: Image.Image, click_positions, monitor, marker_radius: int) -> Image.Image:
    """Draw click markers on the image"""
    draw = ImageDraw.Draw(img)
    for click in click_positions:
        button = click.get('button', 'Button.left')
        position = click.get('position', click)

        img_x, img_y = screen_to_image_coords(position, monitor)

        if img_x < 0 or img_y < 0 or img_x >= img.width or img_y >= img.height:
            continue

        color = BUTTON_COLORS.get(button, 'yellow')

        draw.ellipse(
            [(img_x - marker_radius, img_y - marker_radius),
             (img_x + marker_radius, img_y + marker_radius)],
            fill=color, outline='black', width=2
        )
    return img


def draw_cursor_arrow(img: Image.Image, start_pos, end_pos, monitor, color='orange', label=None) -> Image.Image:
    """Draw an arrow showing cursor movement"""
    draw = ImageDraw.Draw(img)

    start_x, start_y = screen_to_image_coords(start_pos, monitor)
    end_x, end_y = screen_to_image_coords(end_pos, monitor)

    if (start_x < 0 or start_y < 0 or start_x >= img.width or start_y >= img.height or
            end_x < 0 or end_y < 0 or end_x >= img.width or end_y >= img.height):
        return img

    if abs(start_x - end_x) < 2 and abs(start_y - end_y) < 2:
        return img

    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=3)

    arrow_length = 15
    arrow_angle = 25

    dx = end_x - start_x
    dy = end_y - start_y
    angle = np.arctan2(dy, dx)

    arrow_angle_rad = np.radians(arrow_angle)
    x1 = end_x - arrow_length * np.cos(angle - arrow_angle_rad)
    y1 = end_y - arrow_length * np.sin(angle - arrow_angle_rad)
    x2 = end_x - arrow_length * np.cos(angle + arrow_angle_rad)
    y2 = end_y - arrow_length * np.sin(angle + arrow_angle_rad)

    draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill=color, outline='darkorange')

    draw.ellipse([(start_x - 4, start_y - 4), (start_x + 4, start_y + 4)],
                 fill='lime', outline='darkgreen', width=2)

    return img


def extract_mouse_events(events):
    """Extract mouse click events from the events list"""
    mouse_events = []
    for event in events:
        event_type = event.get('event_type', '')
        details = event.get('details', {})
        cursor_pos = event.get('cursor_pos', [])

        if event_type in ['mouse_click', 'mouse_press', 'mouse_release']:
            button = details.get('button', 'Button.left')
            mouse_events.append({
                'button': button,
                'position': cursor_pos,
                'event_type': event_type
            })

    return mouse_events


def get_cursor_movements(events):
    """Extract cursor movement trajectories from events"""
    movements = []
    prev_pos = None

    for event in events:
        cursor_pos = event.get('cursor_pos', [])
        event_type = event.get('event_type', '')

        if cursor_pos and len(cursor_pos) >= 2:
            if prev_pos and prev_pos != cursor_pos:
                movements.append({
                    'start': prev_pos,
                    'end': cursor_pos,
                    'event_type': event_type
                })
            prev_pos = cursor_pos

    return movements


def format_event_info(log: dict):
    """Format event information for display"""
    info_lines = []

    keyboard_events = [e for e in log.get('events', []) if e.get('event_type', '').startswith('keyboard')]
    if keyboard_events:
        keys = [e.get('details', {}).get('key', 'unknown') for e in keyboard_events]
        info_lines.append(f"Keys: {', '.join(keys)}")
    else:
        info_lines.append("Keys: <none>")

    mouse_events = extract_mouse_events(log.get('events', []))
    if mouse_events:
        info_lines.append("Mouse Events:")
        for i, event in enumerate(mouse_events, 1):
            button = event.get('button', 'unknown')
            position = event.get('position', [])
            event_type = event.get('event_type', '')
            info_lines.append(f"  {i}. {event_type}: {button} at {position}")
    else:
        info_lines.append("Mouse Events: <none>")

    start_pos = log.get('start_cursor_pos', [])
    end_pos = log.get('end_cursor_pos', [])
    if start_pos and end_pos:
        info_lines.append(f"Cursor: {start_pos} → {end_pos}")

    scroll_events = [e for e in log.get('events', []) if e.get('event_type') == 'scroll']
    if scroll_events:
        info_lines.append(f"Scrolls: {len(scroll_events)}")

    start_time = log.get('start_timestamp', '')
    end_time = log.get('end_timestamp', '')
    if start_time and end_time:
        info_lines.append(f"Duration: {start_time} → {end_time}")

    return '\n'.join(info_lines)


def visualize_log(log: dict, index: int, out_dir: Path, prev_end_cursor=None):
    """Create visualization for a single log entry"""
    start_path = Path(log['start_screenshot_path'])
    end_path = Path(log['end_screenshot_path'])

    if not start_path.exists() or not end_path.exists():
        print(f"Warning: Screenshot files not found for log {index + 1}")
        return

    img_start = Image.open(start_path).convert('RGB')
    img_end = Image.open(end_path).convert('RGB')

    monitor = log.get('monitor', {})
    events = log.get('events', [])

    img_start = annotate_image(img_start, events, monitor, prev_end_cursor)

    img_start = draw_border(img_start, color='blue', width=BORDER_WIDTH)
    img_end = draw_border(img_end, color='red', width=BORDER_WIDTH)

    event_info = format_event_info(log)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [4, 1]})
    ax_img, ax_text = axes

    ax_img.axis('off')
    combined = Image.new('RGB', (img_start.width + img_end.width, max(img_start.height, img_end.height)))
    combined.paste(img_start, (0, 0))
    combined.paste(img_end, (img_start.width, 0))
    ax_img.imshow(combined)

    ax_text.axis('off')
    ax_text.text(
        0.02, 0.98, event_info,
        va='top', ha='left',
        family='monospace', fontsize=10,
        transform=ax_text.transAxes,
        wrap=True
    )

    legend_elements = []
    mouse_events = extract_mouse_events(events)
    for button, color in BUTTON_COLORS.items():
        if any(event.get('button') == button for event in mouse_events):
            legend_elements.append(patches.Patch(color=color, label=button))

    if any(e.get('cursor_pos') for e in events):
        legend_elements.append(patches.Patch(color='orange', label='Cursor movement'))
        legend_elements.append(patches.Patch(color='lime', label='Start position'))
        if prev_end_cursor:
            legend_elements.append(patches.Patch(color='magenta', label='Inter-log movement'))

    if legend_elements:
        ax_img.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    out_path = out_dir / f'log_{index + 1:02d}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization: {out_path}")


def get_screen_dimensions(log):
    """Get the screen dimensions for a single log"""
    monitor = log.get('monitor', {})

    for screenshot_key in ['start_screenshot_path', 'end_screenshot_path']:
        screenshot_path = Path(log[screenshot_key])
        if screenshot_path.exists():
            img = Image.open(screenshot_path)
            width, height = img.size
            return width, height

    if monitor:
        width = monitor.get('width', 1920)
        height = monitor.get('height', 1080)
        return width, height

    return 1920, 1080


def group_logs_by_screen_size(logs):
    """Group logs by their screen dimensions"""
    groups = {}

    for i, log in enumerate(logs):
        width, height = get_screen_dimensions(log)
        key = f"{width}x{height}"

        if key not in groups:
            groups[key] = []

        groups[key].append((i, log))

    return groups


def convert_to_separate_videos(logs, output_name, should_annotate=True, seconds_per_frame=1, video_format='mp4'):
    """Convert event logs to separate video files for each screen size"""

    screen_groups = group_logs_by_screen_size(logs)

    print(f"Found {len(screen_groups)} different screen sizes:")
    for size, group_logs in screen_groups.items():
        print(f"  {size}: {len(group_logs)} logs")

    output_paths = []

    for screen_size, group_logs in screen_groups.items():
        print(f"\nProcessing screen size: {screen_size}")

        width, height = map(int, screen_size.split('x'))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frame_paths = []

            prev_end_cursor = None

            for group_idx, (original_idx, log) in enumerate(group_logs):
                monitor = log.get('monitor', {})
                events = log.get('events', [])

                print(f"Processing log {group_idx + 1}/{len(group_logs)} (original index: {original_idx})")

                start_path = Path(log['start_screenshot_path'])
                if start_path.exists():
                    img_start = Image.open(start_path).convert('RGB')

                    if should_annotate:
                        img_start = annotate_image(img_start, events, monitor, prev_end_cursor)

                    start_frame_path = temp_path / f"frame_{group_idx * 2:06d}_start.png"
                    img_start.save(start_frame_path)
                    frame_paths.append(start_frame_path)

                end_path = Path(log['end_screenshot_path'])
                if end_path.exists():
                    img_end = Image.open(end_path).convert('RGB')

                    if should_annotate and group_idx < len(group_logs) - 1:
                        next_log = group_logs[group_idx + 1][1]
                        next_start_cursor = next_log.get('start_cursor_pos', [])
                        current_end_cursor = log.get('end_cursor_pos', [])

                        if (next_start_cursor and current_end_cursor and
                                next_start_cursor != current_end_cursor):
                            img_end = draw_cursor_arrow(img_end, current_end_cursor, next_start_cursor, monitor, 'magenta')

                    end_frame_path = temp_path / f"frame_{group_idx * 2 + 1:06d}_end.png"
                    img_end.save(end_frame_path)
                    frame_paths.append(end_frame_path)

                prev_end_cursor = log.get('end_cursor_pos', [])

            if not frame_paths:
                print(f"No frames to process for screen size {screen_size}")
                continue

            fps = 1.0 / seconds_per_frame

            output_filename = f"{output_name}_{screen_size}.{video_format}"
            output_path = OUTPUT_DIR / output_filename
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if video_format.lower() == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            elif video_format.lower() == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif video_format.lower() == 'mov':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif video_format.lower() == 'webm':
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            if not video_writer.isOpened():
                print(f"Error: Could not open video writer for {screen_size}. Trying alternative codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            try:
                for i, frame_path in enumerate(frame_paths):
                    img = Image.open(frame_path).convert('RGB')
                    img_array = np.array(img)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    video_writer.write(img_bgr)

                    if i % 50 == 0:
                        print(f"Processed {i + 1}/{len(frame_paths)} frames")

            finally:
                video_writer.release()

            print(f"Video saved to: {output_path}")
            print(f"Video specs: {width}x{height}, {fps:.2f} FPS, {len(frame_paths)} frames")
            output_paths.append(output_path)

    return output_paths


def convert_to_single_video_with_transitions(logs, output_name, should_annotate=True, seconds_per_frame=1, video_format='mp4'):
    """Convert event logs to a single video with smooth transitions between different screen sizes"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frame_paths = []

        prev_end_cursor = None

        for i, log in enumerate(logs):
            monitor = log.get('monitor', {})
            events = log.get('events', [])

            print(f"Processing log {i + 1}/{len(logs)}")

            width, height = get_screen_dimensions(log)

            start_path = Path(log['start_screenshot_path'])
            if start_path.exists():
                img_start = Image.open(start_path).convert('RGB')

                if should_annotate:
                    img_start = annotate_image(img_start, events, monitor, prev_end_cursor)

                draw = ImageDraw.Draw(img_start)
                resolution_text = f"{width}x{height}"
                draw.text((10, 10), resolution_text, fill='white', stroke_width=2, stroke_fill='black')

                start_frame_path = temp_path / f"frame_{i * 2:06d}_start.png"
                img_start.save(start_frame_path)
                frame_paths.append((start_frame_path, width, height))

            end_path = Path(log['end_screenshot_path'])
            if end_path.exists():
                img_end = Image.open(end_path).convert('RGB')

                if should_annotate and i < len(logs) - 1:
                    next_log = logs[i + 1]
                    next_start_cursor = next_log.get('start_cursor_pos', [])
                    current_end_cursor = log.get('end_cursor_pos', [])

                    if (next_start_cursor and current_end_cursor and
                            next_start_cursor != current_end_cursor):
                        img_end = draw_cursor_arrow(img_end, current_end_cursor, next_start_cursor, monitor, 'magenta')

                draw = ImageDraw.Draw(img_end)
                resolution_text = f"{width}x{height}"
                draw.text((10, 10), resolution_text, fill='white', stroke_width=2, stroke_fill='black')

                end_frame_path = temp_path / f"frame_{i * 2 + 1:06d}_end.png"
                img_end.save(end_frame_path)
                frame_paths.append((end_frame_path, width, height))

            prev_end_cursor = log.get('end_cursor_pos', [])

        if not frame_paths:
            print("No frames to process")
            return

        max_width = max(w for _, w, h in frame_paths)
        max_height = max(h for _, w, h in frame_paths)

        fps = 1.0 / seconds_per_frame

        output_filename = f"{output_name}_combined.{video_format}"
        output_path = OUTPUT_DIR / output_filename
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if video_format.lower() == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif video_format.lower() == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif video_format.lower() == 'mov':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_format.lower() == 'webm':
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (max_width, max_height))

        if not video_writer.isOpened():
            print("Error: Could not open video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (max_width, max_height))

        try:
            for i, (frame_path, frame_width, frame_height) in enumerate(frame_paths):
                img = Image.open(frame_path).convert('RGB')

                canvas = Image.new('RGB', (max_width, max_height), 'black')
                x_offset = (max_width - frame_width) // 2
                y_offset = (max_height - frame_height) // 2
                canvas.paste(img, (x_offset, y_offset))

                img_array = np.array(canvas)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)

                if i % 50 == 0:
                    print(f"Processed {i + 1}/{len(frame_paths)} frames")

        finally:
            video_writer.release()

        print(f"Video saved to: {output_path}")
        print(f"Video specs: {max_width}x{max_height}, {fps:.2f} FPS, {len(frame_paths)} frames")

        return output_path


def annotate_image(img: Image.Image, events, monitor, prev_end_cursor=None):
    """Annotate image with cursor movements and mouse events"""
    movements = get_cursor_movements(events)
    for movement in movements:
        color = EVENT_COLORS.get(movement['event_type'], 'orange')
        img = draw_cursor_arrow(img, movement['start'], movement['end'], monitor, color)

    mouse_events = extract_mouse_events(events)
    img = draw_clicks(img, mouse_events, monitor, CLICK_MARKER_RADIUS)

    if prev_end_cursor and events:
        first_cursor = events[0].get('cursor_pos', [])
        if first_cursor and first_cursor != prev_end_cursor:
            img = draw_cursor_arrow(img, prev_end_cursor, first_cursor, monitor, 'magenta')

    return img


def main():
    parser = argparse.ArgumentParser(description='Process event logs and create visualizations')
    parser.add_argument('--mode', choices=['video', 'video-separate', 'video-combined', 'analysis'],
                        default='analysis',
                        help='Mode: video-separate (one video per screen size), video-combined (single video with transitions), video (legacy), or analysis')
    parser.add_argument('--annotate', action='store_true', default=True,
                        help='Whether to annotate images with cursor movements and clicks')
    parser.add_argument('--seconds-per-frame', type=float, default=1.0,
                        help='Seconds per frame for video mode (default: 1.0)')
    parser.add_argument('--video-format', choices=['mp4', 'avi', 'mov', 'webm'], default='mp4v',
                        help='Video format (default: mp4)')
    parser.add_argument('--input-json', type=str, default=None,
                        help='Path to input JSON file (overrides default)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides default)')

    args = parser.parse_args()

    input_json = Path(args.input_json) if args.input_json else AGG_JSON
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if not input_json.exists():
        print(f"Error: Input file not found: {input_json}")
        return

    logs = json.loads(input_json.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(logs)} logs in {args.mode} mode")
    print(f"Annotation: {'enabled' if args.annotate else 'disabled'}")
    print(f"Video format: {args.video_format}")

    if args.mode == 'video-separate':
        output_paths = convert_to_separate_videos(logs, "event_logs", args.annotate, args.seconds_per_frame, args.video_format)
        print(f"\nCreated {len(output_paths)} video files:")
        for path in output_paths:
            print(f"  {path}")
    elif args.mode == 'video-combined':
        output_path = convert_to_single_video_with_transitions(logs, "event_logs", args.annotate, args.seconds_per_frame, args.video_format)
        print(f"\nCreated combined video: {output_path}")
    else:
        prev_end_cursor = None
        for idx, log in enumerate(logs):
            visualize_log(log, idx, output_dir, prev_end_cursor)
            prev_end_cursor = log.get('end_cursor_pos', [])


if __name__ == '__main__':
    main()
