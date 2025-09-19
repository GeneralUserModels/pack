import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
from pathlib import Path
import threading
import time
from collections import deque
import re


class ImageLabelingTool:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.buffer_dir = self.session_dir / "buffer_screenshots"
        self.labels_file = self.session_dir / "manual_labels.jsonl"

        # Load images and existing labels
        self.images = self._load_images()
        self.labels = self._load_labels()

        # Find first unlabeled image
        self.current_index = self._find_first_unlabeled()

        # Image cache for fast loading
        self.image_cache = {}
        self.cache_size = 100
        self.cache_thread = None
        self.cache_stop_event = threading.Event()

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Manual Image Labeling Tool")
        self.root.configure(bg='black')

        # Make window much larger for bigger images
        self.root.geometry("5080x3328")
        self.root.resizable(True, True)

        # Setup key bindings
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()

        self.setup_gui()
        self.start_cache_preloader()
        self.update_display()

        print(f"Loaded {len(self.images)} images")
        print(f"Found {len(self.labels)} existing labels")
        print(f"Starting at image {self.current_index + 1}/{len(self.images)}")

    def _load_images(self):
        """Load all buffer images sorted by timestamp"""
        if not self.buffer_dir.exists():
            print(f"Buffer directory not found: {self.buffer_dir}")
            return []

        images = []
        pattern = re.compile(r'buffer_(?:active_)?(\d+(?:\.\d+)?)\.jpg')

        for img_file in sorted(self.buffer_dir.glob("buffer_*.jpg")):
            match = pattern.match(img_file.name)
            if match:
                timestamp = float(match.group(1))
                images.append({
                    'filename': img_file.name,
                    'path': img_file,
                    'timestamp': timestamp
                })

        # Sort by timestamp
        images.sort(key=lambda x: x['timestamp'])
        return images

    def _load_labels(self):
        """Load existing labels from JSONL file"""
        labels = {}
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line.strip())
                            labels[entry['filename']] = entry
            except Exception as e:
                print(f"Error loading labels: {e}")
        return labels

    def _find_first_unlabeled(self):
        """Find the first image without a label"""
        for i, img in enumerate(self.images):
            if img['filename'] not in self.labels:
                return i
        return len(self.images) - 1  # All labeled, go to last

    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frame
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Info frame at top
        info_frame = tk.Frame(main_frame, bg='black')
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = tk.Label(info_frame,
                                   text="",
                                   bg='black',
                                   fg='white',
                                   font=('Arial', 14, 'bold'))
        self.info_label.pack()

        self.status_label = tk.Label(info_frame,
                                     text="",
                                     bg='black',
                                     fg='cyan',
                                     font=('Arial', 12))
        self.status_label.pack()

        # Images frame
        images_frame = tk.Frame(main_frame, bg='black')
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - current image (60% width)
        left_frame = tk.Frame(images_frame, bg='black')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        current_label = tk.Label(left_frame,
                                 text="Current Image",
                                 bg='black',
                                 fg='white',
                                 font=('Arial', 16, 'bold'))
        current_label.pack(pady=(0, 5))

        self.current_image_label = tk.Label(left_frame, bg='black')
        self.current_image_label.pack(expand=True)

        self.current_info_label = tk.Label(left_frame,
                                           text="",
                                           bg='black',
                                           fg='yellow',
                                           font=('Arial', 12))
        self.current_info_label.pack(pady=(5, 0))

        # Right side - next images (larger width for bigger images)
        right_frame = tk.Frame(images_frame, bg='black', width=900)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)  # Maintain fixed width

        next_label = tk.Label(right_frame,
                              text="Next Images",
                              bg='black',
                              fg='white',
                              font=('Arial', 14, 'bold'))
        next_label.pack(pady=(0, 15))

        self.next_image_labels = []
        for i in range(2):
            frame = tk.Frame(right_frame, bg='black')
            frame.pack(pady=10, fill=tk.X)

            info = tk.Label(frame,
                            text=f"Next {i + 1}",
                            bg='black',
                            fg='lightgray',
                            font=('Arial', 12))
            info.pack(pady=(0, 5))

            img_label = tk.Label(frame, bg='black')
            img_label.pack()

            self.next_image_labels.append((info, img_label))

        # Controls frame at bottom
        controls_frame = tk.Frame(main_frame, bg='black')
        controls_frame.pack(fill=tk.X, pady=(10, 0))

        controls_text = tk.Label(controls_frame,
                                 text="ENTER: Save (True) | X: Don't Save (False) | ←→: Navigate | ESC: Quit",
                                 bg='black',
                                 fg='lightgreen',
                                 font=('Arial', 14, 'bold'))
        controls_text.pack()

        # Progress bar
        progress_frame = tk.Frame(main_frame, bg='black')
        progress_frame.pack(fill=tk.X, pady=(5, 0))

        self.progress = ttk.Progressbar(progress_frame,
                                        length=400,
                                        mode='determinate')
        self.progress.pack()

        self.progress_label = tk.Label(progress_frame,
                                       text="",
                                       bg='black',
                                       fg='white',
                                       font=('Arial', 10))
        self.progress_label.pack()

    def start_cache_preloader(self):
        """Start background thread to preload images"""
        self.cache_stop_event.clear()
        self.cache_thread = threading.Thread(target=self._cache_preloader, daemon=True)
        self.cache_thread.start()

    def _cache_preloader(self):
        """Background thread that preloads images around current position"""
        while not self.cache_stop_event.is_set():
            try:
                # Cache images around current position
                start_idx = max(0, self.current_index - 10)
                end_idx = min(len(self.images), self.current_index + self.cache_size)

                for i in range(start_idx, end_idx):
                    if self.cache_stop_event.is_set():
                        break

                    if i not in self.image_cache:
                        try:
                            img_path = self.images[i]['path']
                            with Image.open(img_path) as img:
                                # Cache different sizes
                                self.image_cache[i] = {
                                    'large': img.copy(),
                                    'small': img.copy()
                                }
                        except Exception as e:
                            print(f"Error caching image {i}: {e}")

                # Clean cache if too large
                if len(self.image_cache) > self.cache_size:
                    # Remove images far from current position
                    to_remove = []
                    for idx in self.image_cache:
                        if abs(idx - self.current_index) > self.cache_size // 2:
                            to_remove.append(idx)

                    for idx in to_remove[:len(to_remove) // 2]:  # Remove half
                        del self.image_cache[idx]

                time.sleep(0.1)  # Small delay to not overwhelm

            except Exception as e:
                print(f"Cache preloader error: {e}")
                time.sleep(1)

    def get_image(self, index, size='large'):
        """Get image from cache or load it"""
        if index < 0 or index >= len(self.images):
            return None

        # Try cache first
        if index in self.image_cache:
            return self.image_cache[index][size].copy()

        # Load directly if not in cache
        try:
            img_path = self.images[index]['path']
            return Image.open(img_path)
        except Exception as e:
            print(f"Error loading image {index}: {e}")
            return None

    def resize_image_for_display(self, img, max_width, max_height):
        """Resize image to fit display area while maintaining aspect ratio"""
        if not img:
            return None

        img_width, img_height = img.size

        # Calculate scaling factor
        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y)

        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def update_display(self):
        """Update the display with current images"""
        if not self.images:
            return

        # Update info labels
        current_img = self.images[self.current_index]
        filename = current_img['filename']

        # Check if labeled
        label_status = ""
        if filename in self.labels:
            should_save = self.labels[filename]['should_save']
            label_status = f" - LABELED: {'SAVE' if should_save else 'SKIP'}"
            label_color = 'lightgreen' if should_save else 'red'
        else:
            label_status = " - UNLABELED"
            label_color = 'orange'

        self.info_label.config(text=f"Image {self.current_index + 1}/{len(self.images)}: {filename}")
        self.status_label.config(text=label_status, fg=label_color)

        # Update progress
        progress_value = ((self.current_index + 1) / len(self.images)) * 100
        self.progress.config(value=progress_value)

        labeled_count = len(self.labels)
        self.progress_label.config(text=f"Progress: {labeled_count}/{len(self.images)} labeled ({progress_value:.1f}%)")

        # Load and display current image
        current_img_pil = self.get_image(self.current_index, 'large')
        if current_img_pil:
            # Resize for main display (left side gets more space)
            resized_img = self.resize_image_for_display(current_img_pil, 900, 600)
            if resized_img:
                photo = ImageTk.PhotoImage(resized_img)
                self.current_image_label.config(image=photo)
                self.current_image_label.image = photo  # Keep reference

        # Update current image info
        timestamp_str = time.strftime('%H:%M:%S', time.localtime(current_img['timestamp']))
        self.current_info_label.config(text=f"Time: {timestamp_str}")

        # Load and display next images (larger now)
        for i, (info_label, img_label) in enumerate(self.next_image_labels):
            next_idx = self.current_index + i + 1
            if next_idx < len(self.images):
                next_img = self.images[next_idx]
                next_img_pil = self.get_image(next_idx, 'small')

                if next_img_pil:
                    # Resize for preview (much larger than before)
                    resized_next = self.resize_image_for_display(next_img_pil, 550, 300)
                    if resized_next:
                        next_photo = ImageTk.PhotoImage(resized_next)
                        img_label.config(image=next_photo)
                        img_label.image = next_photo  # Keep reference

                # Update info
                next_timestamp = time.strftime('%H:%M:%S', time.localtime(next_img['timestamp']))
                info_label.config(text=f"Next {i + 1}: {next_timestamp}")
            else:
                img_label.config(image='')
                img_label.image = None
                info_label.config(text=f"Next {i + 1}: -")

    def save_label(self, should_save):
        """Save label for current image"""
        if not self.images:
            return

        current_img = self.images[self.current_index]
        filename = current_img['filename']

        label_entry = {
            'filename': filename,
            'timestamp': time.time(),
            'unix_timestamp': current_img['timestamp'],
            'should_save': should_save,
            'image_index': self.current_index
        }

        # Update in-memory labels
        self.labels[filename] = label_entry

        # Append to file
        try:
            with open(self.labels_file, 'a') as f:
                json.dump(label_entry, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving label: {e}")

    def on_key_press(self, event):
        """Handle key press events"""
        key = event.keysym.lower()

        if key == 'return':  # Enter - Save (True)
            self.save_label(True)
            self.move_next()
        elif key == 'x':  # X - Don't save (False)
            self.save_label(False)
            self.move_next()
        elif key == 'right':  # Right arrow - Next
            self.move_next()
        elif key == 'left':  # Left arrow - Previous
            self.move_previous()
        elif key == 'escape':  # Escape - Quit
            self.quit()

    def move_next(self):
        """Move to next image"""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.update_display()

    def move_previous(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def quit(self):
        """Quit the application"""
        self.cache_stop_event.set()
        if self.cache_thread:
            self.cache_thread.join(timeout=1)
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()


def main():
    session_path = Path(__file__).parent / "session"
    if not session_path.exists():
        print(f"Session directory not found: {session_path}")
        return

    buffer_dir = session_path / "buffer_screenshots"
    if not buffer_dir.exists():
        print(f"Buffer screenshots directory not found: {buffer_dir}")
        return

    print(f"Starting manual labeling tool for session: {session_path}")
    print("Controls:")
    print("  ENTER: Label current image as 'should_save = True'")
    print("  X: Label current image as 'should_save = False'")
    print("  ← →: Navigate between images")
    print("  ESC: Quit")
    print()

    app = ImageLabelingTool(session_path)
    app.run()


if __name__ == "__main__":
    main()
