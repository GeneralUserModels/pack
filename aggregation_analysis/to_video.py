import json
import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
import shutil
import argparse


def load_json_data(json_file):
    """Load and return the aggregated actions JSON data."""
    with open(json_file, 'r') as f:
        return json.load(f)


def get_screenshot_files(screenshot_dir):
    """Get all screenshot files and extract their timestamps."""
    screenshot_dir = Path(screenshot_dir)
    screenshots = []

    # Pattern to match buffer_active_TIMESTAMP.jpg
    pattern = re.compile(r'buffer_active_(\d+\.\d+)\.jpg')

    for file_path in screenshot_dir.glob('buffer_active_*.jpg'):
        match = pattern.match(file_path.name)
        if match:
            timestamp = float(match.group(1))
            screenshots.append((timestamp, file_path))

    # Sort by timestamp
    screenshots.sort(key=lambda x: x[0])
    return screenshots


def find_closest_screenshot(target_timestamp, screenshots):
    """Find the screenshot with timestamp closest to target_timestamp."""
    if not screenshots:
        return None

    closest_screenshot = min(screenshots, key=lambda x: abs(x[0] - target_timestamp))
    return closest_screenshot


def _load_font_prefer_truetype(size=55):
    """
    Try to load a TrueType font at requested size.
    Tries common font names/paths; falls back to load_default() if none found.
    """
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue

    # Last resort: default bitmap font (size cannot be changed)
    return ImageFont.load_default()


def add_label_and_border(image_path, output_path, action_type, border_color, label_text):
    """Add label and colored border to an image and save to output_path."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Create a copy to work with
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        # Add colored border
        border_width = 20
        img_width, img_height = img_copy.size

        # Draw border rectangles (one-pixel outline repeated for border_width)
        for i in range(border_width):
            draw.rectangle([i, i, img_width - 1 - i, img_height - 1 - i], outline=border_color, width=1)

        font_size = 55
        font = _load_font_prefer_truetype(font_size)

        # Add label text at bottom center
        if font:
            # Use textbbox to measure text size (works with TrueType and default)
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text in bottom center, above the border with some padding
            padding_bottom = 12
            text_x = (img_width - text_width) // 2
            text_y = img_height - border_width - text_height - padding_bottom
            if text_y < border_width + 5:
                # ensure we don't overlap border; push up if needed
                text_y = border_width + 5

            # Draw background rectangle behind text for readability
            bg_padding_x = 12
            bg_padding_y = 8
            bg_left = text_x - bg_padding_x
            bg_top = text_y - bg_padding_y
            bg_right = text_x + text_width + bg_padding_x
            bg_bottom = text_y + text_height + bg_padding_y

            # Keep background inside image bounds
            bg_left = max(bg_left, 0)
            bg_top = max(bg_top, 0)
            bg_right = min(bg_right, img_width)
            bg_bottom = min(bg_bottom, img_height)

            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill='black', outline='white', width=2)

            # Draw the text
            draw.text((text_x, text_y), label_text, fill='white', font=font)

        # Save the modified image
        img_copy.save(output_path)


def create_video_with_ffmpeg(image_dir, output_video_path, fps=3):
    """Create video from images using ffmpeg at given fps (default 3)."""
    # Create a text file listing all images in order
    image_list_file = os.path.join(image_dir, 'image_list.txt')

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    # If no images, bail out
    if not image_files:
        print("No images found in directory for ffmpeg.")
        return False

    with open(image_list_file, 'w') as f:
        for img_file in image_files:
            f.write(f"file '{img_file}'\n")
            # keep each image visible for 1 second (you can change this if desired)
            f.write("duration 1.0\n")

    # ffmpeg requires the last file to be listed twice for correct concat behavior in some builds
    # append the last file again
    with open(image_list_file, 'a') as f:
        f.write(f"file '{image_files[-1]}'\n")

    # Create temporary output file in the image directory first
    temp_output = os.path.join(image_dir, 'temp_video.mp4')

    # Run ffmpeg command
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-f', 'concat',
        '-safe', '0',
        '-i', image_list_file,
        '-vf', f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        temp_output
    ]

    try:
        subprocess.run(cmd, check=True, cwd=image_dir)
        # Copy the video to the final destination
        shutil.copy2(temp_output, output_video_path)
        print(f"Video created successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False
    except Exception as e:
        print(f"Error copying video to final location: {e}")
        return False

    return True


def main(create_video=False):
    """
    Main processing pipeline.
    If create_video is True, the script will create a video at output_video path.
    If create_video is False, the script will write annotated images to annotated_dir.
    """
    # Configuration
    json_file = './aggregation_analysis/aggregated_actions.json'
    screenshot_dir = './aggregation_analysis/session_6/buffer_screenshots'
    output_video = './aggregation_analysis/video.mp4'
    annotated_dir = './aggregation_analysis/annotated_imgs'  # where annotated images will be placed if not creating video

    # Create output directory if it doesn't exist
    output_parent_dir = os.path.dirname(output_video)
    if output_parent_dir and not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir, exist_ok=True)

    # Ensure annotated directory exists (we will copy there when not creating a video or always keep annotated images)
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir, exist_ok=True)

    # Check if files/directories exist
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        return

    if not os.path.exists(screenshot_dir):
        print(f"Error: {screenshot_dir} not found")
        return

    # Load data
    print("Loading JSON data...")
    actions = load_json_data(json_file)

    print("Loading screenshot files...")
    screenshots = get_screenshot_files(screenshot_dir)

    if not screenshots:
        print("No screenshot files found!")
        return

    print(f"Found {len(screenshots)} screenshots")
    print(f"Processing {len(actions)} actions")

    # Create temporary directory for processed images
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Processing images in temporary directory: {temp_dir}")

        processed_images = []

        for i, action in enumerate(actions):
            action_type = action.get('type', 'unknown')
            start_screenshot_info = action.get('start_screenshot') or {}
            end_screenshot_info = action.get('end_screenshot') or {}
            start_time = start_screenshot_info.get('timestamp')
            end_time = end_screenshot_info.get('timestamp')

            if start_time is None or end_time is None:
                continue

            # Find closest screenshots for start and end
            start_screenshot = find_closest_screenshot(start_time, screenshots)
            end_screenshot = find_closest_screenshot(end_time, screenshots)

            if start_screenshot:
                # Process start image
                start_output = os.path.join(temp_dir, f"{i:04d}_start_{action_type}.jpg")
                label = f"START: {action_type.upper()}"
                add_label_and_border(start_screenshot[1], start_output, action_type, 'red', label)
                processed_images.append((start_time, start_output))
                print(f"Processed start image for {action_type}: {start_screenshot[0]} -> {start_output}")

            if end_screenshot and end_screenshot != start_screenshot:
                # Process end image
                end_output = os.path.join(temp_dir, f"{i:04d}_end_{action_type}.jpg")
                label = f"END: {action_type.upper()}"
                add_label_and_border(end_screenshot[1], end_output, action_type, 'blue', label)
                processed_images.append((end_time, end_output))
                print(f"Processed end image for {action_type}: {end_screenshot[0]} -> {end_output}")

        if not processed_images:
            print("No images to process!")
            return

        # Sort processed images by timestamp to maintain chronological order
        processed_images.sort(key=lambda x: x[0])

        # Rename files to ensure proper ordering for ffmpeg / final output
        final_temp_dir = os.path.join(temp_dir, 'final')
        os.makedirs(final_temp_dir)

        for idx, (timestamp, image_path) in enumerate(processed_images):
            final_path = os.path.join(final_temp_dir, f"{idx:04d}.jpg")
            shutil.copy2(image_path, final_path)

        # Always copy annotated images to annotated_dir (so you can inspect them even if you also make a video)
        for fname in sorted(os.listdir(final_temp_dir)):
            if not fname.endswith('.jpg'):
                continue
            src = os.path.join(final_temp_dir, fname)
            dst = os.path.join(annotated_dir, fname)
            shutil.copy2(src, dst)

        print(f"Annotated images written to: {os.path.abspath(annotated_dir)}")

        if create_video:
            # Create video with absolute path at chosen fps (0.5 fps used previously)
            print(f"Creating video with {len(processed_images)} images...")
            absolute_output_path = os.path.abspath(output_video)
            success = create_video_with_ffmpeg(final_temp_dir, absolute_output_path, fps=0.5)

            if success:
                print(f"Video saved as: {absolute_output_path}")
            else:
                print("Failed to create video")
        else:
            print("create_video is False — skipped video creation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate screenshots and optionally create a video.")
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="Create a video from the annotated images. If not set, the script only writes annotated images to annotated_imgs.",
    )
    args = parser.parse_args()

    main(create_video=args.create_video)
