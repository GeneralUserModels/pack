import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import tempfile
import argparse
from modules import AggregatedLog

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


def get_max_dimensions_from_logs(logs):
    max_width = 0
    max_height = 0

    for log in logs:
        start_path = Path(log.start_screenshot_path)
        if start_path.exists():
            try:
                with Image.open(start_path) as img:
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
            except Exception as e:
                print(f"Warning: Could not read {start_path}: {e}")

        end_path = Path(log.end_screenshot_path)
        if end_path.exists():
            try:
                with Image.open(end_path) as img:
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
            except Exception as e:
                print(f"Warning: Could not read {end_path}: {e}")

    return max_width, max_height


def scale_and_pad_image(img, target_width, target_height, background_color=(0, 0, 0)):
    original_width, original_height = img.size

    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    result = Image.new('RGB', (target_width, target_height), background_color)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    result.paste(scaled_img, (x_offset, y_offset))

    return result, scale, x_offset, y_offset


def screen_to_scaled_coords(screen_pos, monitor, scale, x_offset, y_offset):
    x, y = screen_pos
    img_x = x - monitor['left']
    img_y = y - monitor['top']

    scaled_x = int(img_x * scale) + x_offset
    scaled_y = int(img_y * scale) + y_offset

    return scaled_x, scaled_y


def draw_border(img: Image.Image, color: str, width: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        rect = [i, i, w - i - 1, h - i - 1]
        draw.rectangle(rect, outline=color)
    return img


def draw_clicks(img: Image.Image, click_positions, monitor, marker_radius: int, scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for click in click_positions:
        button = click.get('button', 'Button.left')
        position = click.get('position', click)

        img_x, img_y = screen_to_scaled_coords(position, monitor, scale, x_offset, y_offset)

        if img_x < 0 or img_y < 0 or img_x >= img.width or img_y >= img.height:
            continue

        color = BUTTON_COLORS.get(button, 'yellow')

        scaled_radius = int(marker_radius * scale)

        draw.ellipse(
            [(img_x - scaled_radius, img_y - scaled_radius),
             (img_x + scaled_radius, img_y + scaled_radius)],
            fill=color, outline='black', width=2
        )
    return img


def draw_cursor_arrow(img: Image.Image, start_pos, end_pos, monitor, color='orange', label=None, scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
    draw = ImageDraw.Draw(img)

    start_x, start_y = screen_to_scaled_coords(start_pos, monitor, scale, x_offset, y_offset)
    end_x, end_y = screen_to_scaled_coords(end_pos, monitor, scale, x_offset, y_offset)

    if (start_x < 0 or start_y < 0 or start_x >= img.width or start_y >= img.height or
            end_x < 0 or end_y < 0 or end_x >= img.width or end_y >= img.height):
        return img

    if abs(start_x - end_x) < 2 and abs(start_y - end_y) < 2:
        return img

    line_width = max(1, int(3 * scale))
    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=line_width)

    arrow_length = int(15 * scale)
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

    marker_size = int(4 * scale)
    draw.ellipse([(start_x - marker_size, start_y - marker_size), (start_x + marker_size, start_y + marker_size)],
                 fill='lime', outline='darkgreen', width=2)

    return img


def extract_mouse_events(events):
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
    info_lines = []

    keyboard_events = [e for e in log.events if e.get('event_type', '').startswith('keyboard')]
    keys_str = "Keys: <none>"
    if keyboard_events:
        keys = [e.get('details', {}).get('key', 'unknown') for e in keyboard_events]
        keys = [key if key else "unknown" for key in keys]
        keys_str = f"Keys: {', '.join(keys)}"
    info_lines.append(keys_str)

    mouse_events = extract_mouse_events(log.events)
    if mouse_events:
        info_lines.append("Mouse Events:")
        for i, event in enumerate(mouse_events, 1):
            button = event.get('button', 'unknown')
            position = event.get('position', [])
            event_type = event.get('event_type', '')
            info_lines.append(f"  {i}. {event_type}: {button} at {position}")
    else:
        info_lines.append("Mouse Events: <none>")

    start_pos = log.start_cursor_pos
    end_pos = log.end_cursor_pos
    if start_pos and end_pos:
        info_lines.append(f"Cursor: {start_pos} → {end_pos}")

    scroll_events = [e for e in log.events if e.get('event_type') == 'scroll']
    if scroll_events:
        info_lines.append(f"Scrolls: {len(scroll_events)}")

    start_time = log.start_timestamp
    end_time = log.end_timestamp
    if start_time and end_time:
        info_lines.append(f"Duration: {start_time} → {end_time}")

    return '\n'.join(info_lines)


def visualize_log(log: dict, index: int, out_dir: Path, prev_end_cursor=None, target_width=None, target_height=None):
    start_path = Path(log.start_screenshot_path)
    end_path = Path(log.end_screenshot_path)

    if not start_path.exists() or not end_path.exists():
        print(f"Warning: Screenshot files not found for log {index + 1}")
        return

    img_start = Image.open(start_path).convert('RGB')
    img_end = Image.open(end_path).convert('RGB')

    if target_width and target_height:
        img_start, start_scale, start_x_offset, start_y_offset = scale_and_pad_image(img_start, target_width, target_height)
        img_end, end_scale, end_x_offset, end_y_offset = scale_and_pad_image(img_end, target_width, target_height)
    else:
        start_scale = 1.0
        start_x_offset = start_y_offset = 0

    monitor = log.monitor
    events = log.events

    img_start = annotate_image(img_start, events, monitor, prev_end_cursor, start_scale, start_x_offset, start_y_offset)

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
        if any(event.button == button for event in mouse_events):
            legend_elements.append(patches.Patch(color=color, label=button))

    if any(e.get("cursor_pos", "unknown") for e in events):
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


def convert_to_video(logs, video_name, video_dir, should_annotate=True, seconds_per_frame=1, percentile=95):
    print("Finding maximum dimensions from all screenshots...")
    print(logs[0])
    max_width, max_height = get_max_dimensions_from_logs(logs)
    print(f"Maximum dimensions: {max_width}x{max_height}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frame_paths = []

        prev_end_cursor = None

        for i, log in enumerate(logs):
            monitor = log.monitor
            events = log.events

            print(f"Processing log {i + 1}/{len(logs)}")

            start_path = Path(log.start_screenshot_path)
            if start_path.exists():
                img_start = Image.open(start_path).convert('RGB')
                img_start, start_scale, start_x_offset, start_y_offset = scale_and_pad_image(img_start, max_width, max_height)

                if should_annotate:
                    img_start = annotate_image(img_start, events, monitor, prev_end_cursor, start_scale, start_x_offset, start_y_offset)
            else:
                img_start = Image.new('RGB', (max_width, max_height), (0, 0, 0))
                start_scale, start_x_offset, start_y_offset = 1.0, 0, 0

            start_frame_path = temp_path / f"frame_{i * 2:06d}_start.png"
            img_start.save(start_frame_path)
            frame_paths.append(start_frame_path)

            end_path = Path(log.end_screenshot_path)
            if end_path.exists():
                img_end = Image.open(end_path).convert('RGB')
                img_end, end_scale, end_x_offset, end_y_offset = scale_and_pad_image(img_end, max_width, max_height)

                if should_annotate and i < len(logs) - 1:
                    next_log = logs[i + 1]
                    next_start_cursor = next_log.start_cursor_pos
                    current_end_cursor = log.end_cursor_pos

                    if (next_start_cursor and current_end_cursor and
                            next_start_cursor != current_end_cursor):
                        img_end = draw_cursor_arrow(img_end, current_end_cursor, next_start_cursor, monitor, 'magenta', None, end_scale, end_x_offset, end_y_offset)
            else:
                img_end = Image.new('RGB', (max_width, max_height), (0, 0, 0))

            end_frame_path = temp_path / f"frame_{i * 2 + 1:06d}_end.png"
            img_end.save(end_frame_path)
            frame_paths.append(end_frame_path)

            prev_end_cursor = log.end_cursor_pos

        if not frame_paths:
            print("No frames to process")
            return

        fps = 1.0 / seconds_per_frame

        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{video_name}_{percentile}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (max_width, max_height))

        if not video_writer.isOpened():
            print("Error: Could not open video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (max_width, max_height))

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

        print(f"Video saved to: {video_path}")
        print(f"Video specs: {max_width}x{max_height}, {fps:.2f} FPS, {len(frame_paths)} frames")


def annotate_image(img: Image.Image, events, monitor, prev_end_cursor=None, scale=1.0, x_offset=0, y_offset=0):
    movements = get_cursor_movements(events)
    for movement in movements:
        color = EVENT_COLORS.get(movement['event_type'], 'orange')
        img = draw_cursor_arrow(img, movement['start'], movement['end'], monitor, color, None, scale, x_offset, y_offset)

    mouse_events = extract_mouse_events(events)
    img = draw_clicks(img, mouse_events, monitor, CLICK_MARKER_RADIUS, scale, x_offset, y_offset)

    if prev_end_cursor and events:
        first_cursor = events[0].get('cursor_pos', [])
        if first_cursor and first_cursor != prev_end_cursor:
            img = draw_cursor_arrow(img, prev_end_cursor, first_cursor, monitor, 'magenta', None, scale, x_offset, y_offset)

    return img


def render_images(logs, output_dir):
    max_width, max_height = get_max_dimensions_from_logs(logs)
    print(f"Using maximum dimensions: {max_width}x{max_height}")

    prev_end_cursor = None
    for idx, log in enumerate(logs):
        visualize_log(log, idx, output_dir, prev_end_cursor, max_width, max_height)
        prev_end_cursor = log.end_cursor_pos


def main():
    parser = argparse.ArgumentParser(description='Process event logs and create visualizations')
    parser.add_argument('--mode', choices=['video', 'analysis'], default='analysis',
                        help='Mode: video or analysis (default: analysis)')
    parser.add_argument('--annotate', action='store_true', default=True,
                        help='Whether to annotate images with cursor movements and clicks')
    parser.add_argument('--seconds-per-frame', type=float, default=1.0,
                        help='Seconds per frame for video mode (default: 1.0)')
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
    logs = [AggregatedLog.from_dict(log) for log in logs]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(logs)} logs in {args.mode} mode")
    print(f"Annotation: {'enabled' if args.annotate else 'disabled'}")

    if args.mode == 'video':
        convert_to_video(logs, "event_logs_video", OUTPUT_DIR, args.annotate, args.seconds_per_frame, percentile=PERCENTILE)
    else:
        render_images(logs, output_dir)


if __name__ == '__main__':

    PERCENTILE = 85

    AGG_JSON = Path(__file__).parent.parent / "prompting_benchmark" / "data" / f'aggregated_logs_{PERCENTILE}.json'
    OUTPUT_DIR = Path(__file__).parent.parent / "prompting_benchmark" / "data" / f'agg_{PERCENTILE}_visualizations'
    main()
