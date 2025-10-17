import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import argparse


class EventAnnotator:
    """Annotates screenshots with event information."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to load a font, fallback to default if not available
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def compress_events(self, events: List[dict]) -> List[dict]:
        """Compress consecutive mouse_move and mouse_scroll events."""
        compressed = []
        i = 0
        n = len(events)

        while i < n:
            e = events[i]
            et = e.get("event_type")

            if et == "mouse_move":
                start_pos = e.get("cursor_position")
                last_pos = start_pos
                j = i + 1
                while j < n and events[j].get("event_type") == "mouse_move":
                    candidate_pos = events[j].get("cursor_position", last_pos)
                    if candidate_pos is not None:
                        last_pos = candidate_pos
                    j += 1
                monitor = e.get("monitor", {})
                compressed.append({
                    "event_type": "mouse_move",
                    "cursor_position": (start_pos, last_pos),
                    "monitor": monitor
                })
                i = j

            elif et == "mouse_scroll":
                j = i
                last_dir = None
                monitor = e.get("monitor", {})
                while j < n and events[j].get("event_type") == "mouse_scroll":
                    details = events[j].get("details", {})
                    dir_ = self._convert_scroll_direction(details)
                    if dir_ != last_dir:
                        compressed.append({
                            "event_type": "mouse_scroll",
                            "details": details,
                            "_direction": dir_,
                            "monitor": events[j].get("monitor", monitor)
                        })
                        last_dir = dir_
                    j += 1
                i = j

            else:
                compressed.append(e)
                i += 1

        return compressed

    def _convert_scroll_direction(self, scroll_data):
        """Convert scroll data to human-readable direction."""
        if isinstance(scroll_data, dict):
            dx = scroll_data.get("dx", 0)
            dy = scroll_data.get("dy", 0)
        elif isinstance(scroll_data, (list, tuple)) and len(scroll_data) >= 2:
            dx, dy = scroll_data[0], scroll_data[1]
        else:
            return "no scroll"

        directions = []
        if dy > 0:
            directions.append("up")
        elif dy < 0:
            directions.append("down")
        if dx > 0:
            directions.append("right")
        elif dx < 0:
            directions.append("left")

        return " ".join(directions) if directions else "no scroll"

    def generate_event_summary(self, events: List[dict], timestamp: str) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
        """
        Generate event summary and extract positions.
        Returns: (summary_text, click_positions, move_paths)
        """
        events = self.compress_events(events)

        actions = []
        keys_pressed = []
        click_positions = []
        move_paths = []

        default_monitor = events[0].get("monitor", {}) if events else {}

        for event in events:
            event_type = event.get("event_type")

            if event_type == "key_press":
                key = event.get("details", {}).get("key", "unknown")
                key = key.replace("Key.", "") if key and isinstance(key, str) and key.startswith("Key.") else key
                if key:
                    keys_pressed.append(key)

            elif event_type == "mouse_down":
                print(event)
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                details = event.get("details", {})
                button = details.get("button", "unknown")
                button = button.replace("Button.", "") if button and isinstance(button, str) and button.startswith("Button.") else button
                cursor_pos = event.get("cursor_position", [])
                double_click = details.get("double_click", False)

                if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) == 2:
                    click_positions.append(tuple(cursor_pos))
                    click_text = f"Click {button}"
                    if double_click:
                        click_text += " (2x)"
                    actions.append(click_text)

            elif event_type == "mouse_scroll":
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                direction = event.get("_direction") or self._convert_scroll_direction(event.get("details", {}))
                actions.append(f"Scroll {direction}")

            elif event_type == "mouse_move":
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'|'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                cursor_pos = event.get("cursor_position", [])

                if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) == 2 and \
                   isinstance(cursor_pos[0], (list, tuple)) and isinstance(cursor_pos[1], (list, tuple)):
                    start_pos, end_pos = cursor_pos[0], cursor_pos[1]
                    move_paths.append((start_pos[0], start_pos[1], end_pos[0], end_pos[1]))
                    actions.append(f"Move mouse")

        if keys_pressed:
            actions.append(
                f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                else f"Keys: {'|'.join(keys_pressed)}"
            )

        summary = f"Event at {timestamp}:\n" + "\n".join([f"  • {action}" for action in actions])

        return summary, click_positions, move_paths

    def annotate_screenshot(self, screenshot_path: Path, events: List[dict],
                            timestamp: str, output_filename: str):
        """Annotate a screenshot with event information and mouse positions."""
        # Load image
        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)

        # Generate summary and positions
        summary, click_positions, move_paths = self.generate_event_summary(events, timestamp)

        # Draw move paths (lines)
        for path in move_paths:
            x1, y1, x2, y2 = path
            # Draw arrow line
            draw.line([(x1, y1), (x2, y2)], fill='blue', width=3)
            # Draw arrowhead
            self._draw_arrow_head(draw, x1, y1, x2, y2, 'blue')

        # Draw click positions (circles)
        for pos in click_positions:
            x, y = pos
            radius = 15
            # Draw outer circle
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         outline='red', width=4)
            # Draw inner circle
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')

        # Draw text box with summary
        self._draw_text_box(draw, img.size, summary)

        # Save annotated image
        output_path = self.output_dir / output_filename
        img.save(output_path)
        print(f"✓ Saved: {output_path}")

    def _draw_arrow_head(self, draw, x1, y1, x2, y2, color):
        """Draw an arrow head at the end of a line."""
        import math

        # Calculate angle
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_length = 15
        arrow_angle = math.pi / 6

        # Calculate arrow points
        point1_x = x2 - arrow_length * math.cos(angle - arrow_angle)
        point1_y = y2 - arrow_length * math.sin(angle - arrow_angle)
        point2_x = x2 - arrow_length * math.cos(angle + arrow_angle)
        point2_y = y2 - arrow_length * math.sin(angle + arrow_angle)

        # Draw arrow head
        draw.polygon([(x2, y2), (point1_x, point1_y), (point2_x, point2_y)],
                     fill=color)

    def _draw_text_box(self, draw, img_size, text):
        """Draw a semi-transparent text box with event summary."""
        width, height = img_size

        # Split text into lines
        lines = text.split('\n')

        # Calculate box dimensions
        max_line_width = max([draw.textlength(line, font=self.font_small) for line in lines])
        padding = 20
        line_height = 20
        box_width = min(max_line_width + 2 * padding, width - 40)
        box_height = len(lines) * line_height + 2 * padding

        # Position box at top-left
        box_x = 10
        box_y = 10

        # Draw semi-transparent background
        overlay = Image.new('RGBA', img_size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            fill=(0, 0, 0, 180)
        )

        # Composite overlay
        img_rgba = draw._image.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw._image.paste(img_rgba.convert('RGB'))

        # Draw text
        y_offset = box_y + padding
        for line in lines:
            draw.text((box_x + padding, y_offset), line,
                      fill='white', font=self.font_small)
            y_offset += line_height


def process_session(session_dir: Path, output_dir: Optional[Path] = None):
    """Process a session directory and annotate all screenshots."""
    session_dir = Path(session_dir)

    if output_dir is None:
        output_dir = session_dir / "annotated_screenshots"
    else:
        output_dir = Path(output_dir)

    # Find aggregations file
    aggregations_file = session_dir / "aggregations.jsonl"
    if not aggregations_file.exists():
        print(f"Error: {aggregations_file} not found")
        return

    screenshots_dir = session_dir / "screenshots"
    if not screenshots_dir.exists():
        print(f"Error: {screenshots_dir} not found")
        return

    # Create annotator
    annotator = EventAnnotator(output_dir)

    # Process each aggregation
    print(f"Processing aggregations from {aggregations_file}")
    count = 0

    with open(aggregations_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            agg = json.loads(line)
            screenshot_path = agg.get("screenshot_path")
            events = agg.get("events", [])
            timestamp = agg.get("timestamp", "unknown")

            if not screenshot_path:
                print(f"⚠ Skipping aggregation: no screenshot_path")
                continue

            # Resolve screenshot path
            screenshot_full_path = Path(screenshot_path)
            if not screenshot_full_path.exists():
                # Try relative to session dir
                screenshot_full_path = session_dir / screenshot_path

            if not screenshot_full_path.exists():
                print(f"⚠ Screenshot not found: {screenshot_path}")
                continue

            # Generate output filename
            screenshot_name = screenshot_full_path.name
            output_filename = f"annotated_{count:04d}_{screenshot_name}"

            # Annotate screenshot
            annotator.annotate_screenshot(
                screenshot_full_path,
                events,
                str(timestamp),
                output_filename
            )

            count += 1

    print(f"\n✓ Processed {count} screenshots")
    print(f"✓ Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate session screenshots with event information"
    )
    parser.add_argument(
        "session_dir",
        type=str,
        help="Path to session directory containing aggregations.jsonl and screenshots/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotated screenshots (default: session_dir/annotated_screenshots)"
    )

    args = parser.parse_args()
    process_session(args.session_dir, args.output_dir)


if __name__ == "__main__":
    main()
