import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import tempfile

PERCENTILE = 95

AGG_JSON = Path(__file__).parent.parent / "logs" / 'session_2025-07-03_01-04-03-001589' / f'events.agg_{PERCENTILE}.json'
OUTPUT_DIR = Path(__file__).parent.parent / "logs" / 'session_2025-07-03_01-04-03-001589' / f'agg_{PERCENTILE}_visualizations'
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


def screen_to_image_coords(screen_pos, monitor):
    x, y = screen_pos
    img_x = x - monitor['left']
    img_y = y - monitor['top']
    return img_x, img_y


def draw_border(img: Image.Image, color: str, width: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        rect = [i, i, w - i - 1, h - i - 1]
        draw.rectangle(rect, outline=color)
    return img


def draw_clicks(img: Image.Image, click_positions, monitor, marker_radius: int) -> Image.Image:
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


def draw_cursor_arrow(img: Image.Image, start_pos, end_pos, monitor) -> Image.Image:
    draw = ImageDraw.Draw(img)

    start_x, start_y = screen_to_image_coords(start_pos, monitor)
    end_x, end_y = screen_to_image_coords(end_pos, monitor)

    if (start_x < 0 or start_y < 0 or start_x >= img.width or start_y >= img.height or
            end_x < 0 or end_y < 0 or end_x >= img.width or end_y >= img.height):
        return img

    draw.line([(start_x, start_y), (end_x, end_y)], fill='orange', width=3)

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

    draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill='orange', outline='darkorange')

    draw.ellipse([(start_x - 4, start_y - 4), (start_x + 4, start_y + 4)],
                 fill='lime', outline='darkgreen', width=2)

    return img


def format_event_info(log: dict):
    info_lines = []

    keys = log.get('keys_pressed', [])
    if keys:
        info_lines.append(f"Keys: {' + '.join(keys)}")
    else:
        info_lines.append("Keys: <none>")

    clicks = log.get('click_positions', [])
    if clicks:
        info_lines.append("Clicks:")
        for i, click in enumerate(clicks, 1):
            button = click.get('button', 'unknown')
            position = click.get('position', click)
            is_double = click.get("double_click", False)
            info_lines.append(f"  {i}. {button} at {position} {' (double click)' if is_double else ''}")
    else:
        info_lines.append("Clicks: <none>")

    start_pos = log.get('start_cursor_pos', [])
    end_pos = log.get('end_cursor_pos', [])
    if start_pos and end_pos:
        info_lines.append(f"Cursor: {start_pos} → {end_pos}")

    scrolls = log.get('scroll_directions', [])
    if scrolls:
        info_lines.append(f"Scrolls: {len(scrolls)}")

    start_time = log.get('start_timestamp', '')
    end_time = log.get('end_timestamp', '')
    if start_time and end_time:
        info_lines.append(f"Duration: {start_time} → {end_time}")

    return '\n'.join(info_lines)


def visualize_log(log: dict, index: int, out_dir: Path):
    start_path = Path(log['start_screenshot_path'])
    end_path = Path(log['end_screenshot_path'])
    img_start = Image.open(start_path).convert('RGB')
    img_end = Image.open(end_path).convert('RGB')

    monitor = log.get('monitor', {})

    click_positions = log.get('click_positions', [])
    img_start = draw_clicks(img_start, click_positions, monitor, CLICK_MARKER_RADIUS)

    start_cursor = log.get('start_cursor_pos', [])
    end_cursor = log.get('end_cursor_pos', [])
    if start_cursor and end_cursor:
        img_start = draw_cursor_arrow(img_start, start_cursor, end_cursor, monitor)

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
    for button, color in BUTTON_COLORS.items():
        if any(click.get('button') == button for click in click_positions):
            legend_elements.append(patches.Patch(color=color, label=button))

    if legend_elements:
        legend_elements.append(patches.Patch(color='orange', label='Cursor movement'))
        legend_elements.append(patches.Patch(color='lime', label='Start position'))
        ax_img.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    out_path = out_dir / f'log_{index + 1:02d}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization: {out_path}")


def extrapolate_cursor_event_logs(logs):
    """
    Extrapolate cursor positions from event logs when they are null.

    This function analyzes the events within each log to find cursor positions from:
    1. Mouse movement events
    2. Click events with position data
    3. Scroll events with position data

    Args:
        logs: List of log dictionaries

    Returns:
        List of log dictionaries with extrapolated cursor positions
    """
    updated_logs = []

    for log in logs:
        updated_log = log.copy()
        events = log.get('events', [])

        start_cursor = log.get('start_cursor_pos')
        end_cursor = log.get('end_cursor_pos')

        event_positions = []

        for event in events:
            event_type = event.get('event_type', '')
            details = event.get('details', {})

            if event_type == 'mouse_move':
                if 'x' in details and 'y' in details:
                    event_positions.append([details['x'], details['y']])
                elif 'position' in details:
                    pos = details['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        event_positions.append([pos[0], pos[1]])

            elif event_type in ['mouse_click', 'mouse_press', 'mouse_release']:
                if 'x' in details and 'y' in details:
                    event_positions.append([details['x'], details['y']])
                elif 'position' in details:
                    pos = details['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        event_positions.append([pos[0], pos[1]])

            elif event_type == 'scroll':
                if 'x' in details and 'y' in details:
                    event_positions.append([details['x'], details['y']])
                elif 'position' in details:
                    pos = details['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        event_positions.append([pos[0], pos[1]])

        click_positions = log.get('click_positions', [])
        for click in click_positions:
            if 'position' in click:
                pos = click['position']
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    event_positions.append([pos[0], pos[1]])
            elif isinstance(click, (list, tuple)) and len(click) >= 2:
                event_positions.append([click[0], click[1]])

        if event_positions:
            if start_cursor is None:
                updated_log['start_cursor_pos'] = event_positions[0]
                print(f"Extrapolated start cursor position: {event_positions[0]}")

            if end_cursor is None:
                updated_log['end_cursor_pos'] = event_positions[-1]
                print(f"Extrapolated end cursor position: {event_positions[-1]}")

        if updated_log.get('start_cursor_pos') is None and updated_logs:
            prev_end = updated_logs[-1].get('end_cursor_pos')
            if prev_end is not None:
                updated_log['start_cursor_pos'] = prev_end
                print(f"Used previous log's end cursor position: {prev_end}")

        updated_logs.append(updated_log)

    return updated_logs


def convert_to_video(logs, output_name, should_annotate=True, seconds_per_frame=1):
    """
    Convert event logs to a video file.

    Args:
        logs: List of log dictionaries
        output_name: Name of the output video file (without extension)
        should_annotate: Whether to annotate frames with cursor movements and clicks
        seconds_per_frame: Duration each frame should be displayed in seconds
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frame_paths = []

        for i, log in enumerate(logs):
            monitor = log.get('monitor', {})

            start_cursor = log.get('start_cursor_pos')
            end_cursor = log.get('end_cursor_pos')

            click_positions = log.get('click_positions', [])

            start_path = Path(log['start_screenshot_path'])
            if start_path.exists():
                img_start = Image.open(start_path).convert('RGB')

                if should_annotate:
                    if click_positions:
                        img_start = draw_clicks(img_start, click_positions, monitor, CLICK_MARKER_RADIUS)

                    if (start_cursor and end_cursor and
                        start_cursor != end_cursor and
                            len(start_cursor) >= 2 and len(end_cursor) >= 2):
                        img_start = draw_cursor_arrow(img_start, start_cursor, end_cursor, monitor)

                start_frame_path = temp_path / f"frame_{i * 2:06d}_start.png"
                img_start.save(start_frame_path)
                frame_paths.append(start_frame_path)

            end_path = Path(log['end_screenshot_path'])
            if end_path.exists():
                img_end = Image.open(end_path).convert('RGB')

                if should_annotate:
                    if click_positions:
                        img_end = draw_clicks(img_end, click_positions, monitor, CLICK_MARKER_RADIUS)

                end_frame_path = temp_path / f"frame_{i * 2 + 1:06d}_end.png"
                img_end.save(end_frame_path)
                frame_paths.append(end_frame_path)

        if not frame_paths:
            print("No frames to process")
            return

        first_frame = Image.open(frame_paths[0])
        width, height = first_frame.size

        fps = 1.0 / seconds_per_frame

        output_path = OUTPUT_DIR / f"{output_name}.mp4"
        OUTPUT_DIR.mkdir(exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                video_writer.write(img_bgr)

        finally:
            video_writer.release()

        print(f"Video saved to: {output_path}")
        print(f"Video specs: {width}x{height}, {fps:.2f} FPS, {len(frame_paths)} frames")


def main():
    logs = json.loads(AGG_JSON.read_text())
    OUTPUT_DIR.mkdir(exist_ok=True)

    logs = extrapolate_cursor_event_logs(logs)

    updated_logs_path = AGG_JSON.parent / f'events.agg_{PERCENTILE}_with_cursor.json'
    with open(updated_logs_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Saved logs with extrapolated cursor positions to: {updated_logs_path}")

    # for idx, log in enumerate(logs):
    #     visualize_log(log, idx, OUTPUT_DIR)
    #
    convert_to_video(logs, "event_logs_video", should_annotate=True, seconds_per_frame=1)


if __name__ == '__main__':
    main()
