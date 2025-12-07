import os
from sys import argv
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import typing_extensions as typing

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# --- Configuration ---
# Switching to 1.5-pro for best instruction following on complex cleaning
MODEL_NAME = "gemini-3-pro-preview"
OUTPUT_FILE = "extracted_benchmarks.json"


class TaskExtractor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.raw_data = self._load_data()

        # We will update these captions in the sanitization phase
        self.sanitized_data = []
        self.transcript = ""

    def _load_data(self):
        """Reads the JSONL file and sorts by timestamp."""
        data = []
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Error: {self.input_file} not found.")
            return []

        # Sort by start time to ensure linearity
        data.sort(key=lambda x: x['start_time'])
        return data

    def _create_transcript(self, data_source):
        """Creates a simplified text representation for the LLM."""
        lines = []
        base_time = data_source[0]['start_time'] if data_source else 0

        for idx, event in enumerate(data_source):
            rel_seconds = event['start_time'] - base_time
            rel_time_str = f"{int(rel_seconds // 60):02d}:{int(rel_seconds % 60):02d}"
            # Use the global index from the raw data
            lines.append(f"[ID: {idx} | Time: {rel_time_str}] {event.get('caption', 'No caption')}")

        return "\n".join(lines)

    def sanitize_data(self):
        """Phase 0: Remove PII from captions using LLM."""
        print("🛡️ Phase 0: Sanitizing PII (Names, Emails) from captions...")

        # We process in batches if data is huge, but for now we assume it fits context
        transcript = self._create_transcript(self.raw_data)

        model = genai.GenerativeModel(MODEL_NAME)

        prompt = f"""
        You are a Privacy Officer. Your job is to scrub PII (Personally Identifiable Information) from a log of computer interactions.

        Instructions:
        1. Replace names with roles in brackets, e.g., 'Julian' -> '[COLLEAGUE]', 'Dr. Smith' -> '[PROFESSOR]', 'Mom' -> '[FAMILY]'.
        2. Replace emails/phone numbers with '[EMAIL]' or '[PHONE]'.
        3. Replace specific private addresses with '[ADDRESS]'.
        4. **DO NOT** change the ID or Time stamps.
        5. **DO NOT** change technical details (file names like 'video.py', library names, code snippets).
        6. Return the list of cleaned captions preserving the exact order and IDs.

        Input Transcript:
        {transcript}

        Output format: JSON list of objects: {{"id": int, "clean_caption": str}}
        """

        class CleanEntry(typing.TypedDict):
            id: int
            clean_caption: str

        class CleanResult(typing.TypedDict):
            entries: list[CleanEntry]

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=CleanResult
            )
        )

        try:
            cleaned_entries = json.loads(response.text)['entries']

            # Create a deep copy of raw data to sanitize
            self.sanitized_data = [dict(d) for d in self.raw_data]

            for entry in cleaned_entries:
                idx = entry['id']
                if idx < len(self.sanitized_data):
                    self.sanitized_data[idx]['caption'] = entry['clean_caption']

            # Re-generate transcript based on sanitized data
            self.transcript = self._create_transcript(self.sanitized_data)
            print("✅ Data sanitized.")

        except Exception as e:
            print(f"Error in sanitization: {e}")
            # Fallback to raw data if sanitization fails
            self.sanitized_data = self.raw_data
            self.transcript = self._create_transcript(self.raw_data)

    def segment_tasks(self):
        """Phase 1: Segmentation."""
        print("🤖 Phase 1: Analyzing session for task segmentation...")

        class EventTag(typing.TypedDict):
            event_id: int
            task_ids: list[str]  # Allow multiple tasks

        class SegmentationResult(typing.TypedDict):
            events: list[EventTag]

        model = genai.GenerativeModel(MODEL_NAME)

        prompt = f"""
        Analyze this computer session log. Identify distinct high-level goals.
        A task can span multiple events. Some events may not belong to any task (idle browsing, distractions).
        Not all tasks are considered especially productive, e.g. coordinating dinner can be a task on it's own.
        If the user spends some events on related thinks, they are not noise but their own task.


        Rules:
        1. Assign a generic ID (TASK_A, TASK_B) to broad activities.
        2. Events can belong to multiple tasks if they are relevant to both.
        3. Mark idle/irrelevant browsing as empty task_ids [].

        Transcript:
        {self.transcript}
        """

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SegmentationResult
            )
        )
        try:
            return json.loads(response.text)['events']
        except Exception:
            return []

    def describe_and_filter_tasks(self, segmentation_map):
        """Phase 2: Generate Generalized Benchmarks & Denoise."""
        print("🤖 Phase 2: Generating Abstract Descriptions & Denoising...")

        # Group by Task ID
        task_groups = {}
        for tag in segmentation_map:
            event_idx = tag['event_id']
            if event_idx >= len(self.sanitized_data):
                continue

            # We use sanitized data here
            event_obj = self.sanitized_data[event_idx]
            # Add the ID to the object so the LLM knows which is which
            event_obj['_temp_id'] = event_idx

            for t_id in tag['task_ids']:
                if t_id not in task_groups:
                    task_groups[t_id] = []
                task_groups[t_id].append(event_obj)

        final_benchmarks = []
        model = genai.GenerativeModel(MODEL_NAME)

        for t_id, events in task_groups.items():
            # Create numbered list for the LLM to pick from
            task_text = "\n".join([f"ID {e['_temp_id']}: {e['caption']}" for e in events])

            prompt = f"""
            You are creating a generalized AI benchmark based on this human recording.

            1. **Generalize the Goal:** don't say "Message Julian", say "Coordinate lunch with a colleague". Don't say "Click link 3", say "Find a relevant search result".
            2. **Denoise:** Look at the events. Which ID numbers are actually useful for this goal? Exclude accidental clicks, typos, or checking unrelated notifications.

            Valid Categories: [Software Development, Information Retrieval, Communication, System Administration, Data Processing, Personal Planning].

            Input Actions:
            {task_text}

            Output JSON:
            - caption: The generalized task instruction.
            - category: One of the valid categories.
            - relevant_event_ids: A list of integers (the IDs from the Input Actions) that are valid steps.
            """

            class TaskRefinement(typing.TypedDict):
                caption: str
                category: str
                relevant_event_ids: list[int]

            res = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TaskRefinement
                )
            )

            try:
                data = json.loads(res.text)

                # Filter the events based on LLM's selection
                valid_ids = set(data['relevant_event_ids'])
                clean_events = [e for e in events if e['_temp_id'] in valid_ids]

                # Remove the temp_id before saving
                for e in clean_events:
                    if '_temp_id' in e:
                        del e['_temp_id']

                final_benchmarks.append({
                    "task_id_internal": t_id,
                    "caption": data['caption'],
                    "category": data['category'],
                    "events": clean_events
                })
                print(f"   > Processed {t_id}: {data['category']}")
            except Exception as e:
                print(f"Failed to process task {t_id}: {e}")

        return final_benchmarks

    def visualize(self, benchmarks):
        """Phase 3: Visualization including Noise."""
        print("📊 Phase 3: Generating Visualization with Noise Tracking...")

        if not benchmarks:
            return

        # 1. Calculate Global Noise (Events present in NO benchmark)
        all_event_ids = set(range(len(self.sanitized_data)))
        included_ids = set()

        # We need to recover IDs. In a real scenario, keep IDs in the object.
        # Since we stripped them, we match by object identity or timestamp is safer.
        # But for this script, let's rely on the fact we didn't deep copy raw_data in a way that lost order completely.
        # Actually, let's use the start_time as a unique key for matching "Noise"

        benchmark_timestamps = set()
        for b in benchmarks:
            for e in b['events']:
                benchmark_timestamps.add(e['start_time'])

        noise_events = []
        for e in self.sanitized_data:
            if e['start_time'] not in benchmark_timestamps:
                noise_events.append(e)

        # 2. Plotting
        fig, ax = plt.subplots(figsize=(14, 8))

        categories = list(set(b['category'] for b in benchmarks))
        categories.sort()
        # Add Noise category
        categories.append("Noise / Idle")

        # Assign colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        cat_color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
        cat_color_map["Noise / Idle"] = "#d3d3d3"  # Grey for noise

        # We plot rows by Benchmark Items + 1 for Noise
        yticks = []
        yticklabels = []

        row_idx = 0

        # Plot Noise first (at the bottom or top)
        if noise_events:
            yticks.append(row_idx)
            yticklabels.append("Background Noise / Unused")
            for event in noise_events:
                start = event['start_time']
                duration = max(event['end_time'] - event['start_time'], 0.5)
                ax.barh(row_idx, duration, left=start, height=0.6, align='center',
                        color=cat_color_map["Noise / Idle"], alpha=0.5)
            row_idx += 1

        # Plot Benchmarks
        for task in benchmarks:
            cat = task['category']
            color = cat_color_map.get(cat, "#333333")
            label = f"[{cat}] {task['caption'][:40]}..."

            yticks.append(row_idx)
            yticklabels.append(label)

            for event in task['events']:
                start = event['start_time']
                duration = max(event['end_time'] - event['start_time'], 0.5)
                ax.barh(row_idx, duration, left=start, height=0.6, align='center', color=color, alpha=0.9)

            row_idx += 1

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Timestamp')
        ax.set_title('Extracted Agentic Tasks (Denoised & Anonymized)')
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig("task_visualization_v2.png")
        print("Visualization saved to task_visualization_v2.png")
        plt.show()


def main():
    input_path = argv[1] if len(argv) > 1 else "data.jsonl"
    extractor = TaskExtractor(input_path)

    # 1. Sanitize
    extractor.sanitize_data()

    # 2. Segment
    segmentation_map = extractor.segment_tasks()

    # 3. Refine & Denoise
    benchmarks = extractor.describe_and_filter_tasks(segmentation_map)

    # 4. Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmarks, f, indent=4)

    # 5. Visualize
    extractor.visualize(benchmarks)


if __name__ == "__main__":
    main()
