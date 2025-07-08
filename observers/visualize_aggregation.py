import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

AGG_JSON = Path(__file__).parent.parent / "logs" / 'session_2025-07-03_01-04-03-001589' / 'events.agg.json'
OUTPUT_DIR = Path(__file__).parent.parent / "logs" / 'session_2025-07-03_01-04-03-001589' / 'agg_visualizations'
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


def load_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()


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
            info_lines.append(f"  {i}. {button} at {position}")
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


def main():
    logs = json.loads(AGG_JSON.read_text())
    OUTPUT_DIR.mkdir(exist_ok=True)

    for idx, log in enumerate(logs):
        visualize_log(log, idx, OUTPUT_DIR)


if __name__ == '__main__':
    main()
