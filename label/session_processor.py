from __future__ import annotations
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from tqdm import tqdm

from label.clients import PromptClient
from record.models.aggregation import AggregationRequest, ProcessedAggregation


@dataclass
class SessionTask:
    """Represents a single chunk processing task."""
    session_id: str
    chunk_index: int
    video_path: Path
    prompt: str
    aggregations: List[Any]
    output_dir: Path


@dataclass
class SessionConfig:
    """Configuration for a single session."""
    session_folder: Path
    agg_jsonl: Path
    out_chunks_dir: Path


class SessionProcessor:
    """Handles parallel processing of multiple sessions with different strategies."""

    def __init__(
        self,
        prompt_client: PromptClient,
        num_workers: int = 4,
        batch_size: int = 8,
        use_batching: bool = False,
    ):
        """
        Initialize parallel processor.

        Args:
            prompt_client: The VLM client to use
            num_workers: Number of concurrent workers (for Gemini)
            batch_size: Batch size for vLLM batch processing
            use_batching: Whether to use batch processing (vLLM) or concurrent workers (Gemini)
        """
        self.prompt_client = prompt_client
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_batching = use_batching

    def process_multiple_sessions(
        self,
        session_configs: List[SessionConfig],
        chunk_duration: int = 60,
        fps: int = 1,
        label_video: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Process multiple sessions in parallel.

        Args:
            session_configs: List of session configurations
            chunk_duration: Duration of each video chunk in seconds
            fps: Frames per second for video creation
            label_video: Whether to annotate video frames

        Returns:
            Dictionary mapping session_id to list of chunk results
        """
        from label.video import (
            compute_max_image_size,
            create_video_from_images,
            split_video,
        )

        print(f"[ParallelProcessor] Processing {len(session_configs)} sessions...")
        print(f"[ParallelProcessor] Mode: {'Batching' if self.use_batching else 'Concurrent'}")
        print(f"[ParallelProcessor] Workers/Batch: {self.batch_size if self.use_batching else self.num_workers}")

        # Step 1: Prepare all sessions (create videos and split into chunks)
        all_tasks = []

        for config in tqdm(session_configs, desc="Preparing sessions"):
            session_id = config.session_folder.name

            # List and process screenshots
            screenshots = self.list_screenshots(config.session_folder)
            if not screenshots:
                print(f"[Warning] No screenshots in {session_id}, skipping...")
                continue

            images_and_ts = [(p, self.extract_timestamp_from_filename(p)) for p in screenshots]
            images_and_ts = [it for it in images_and_ts if it[1] is not None]
            images_and_ts.sort(key=lambda x: x[1])

            global_start = images_and_ts[0][1]

            # Create master video
            master_video = config.out_chunks_dir / "full_session.mp4"
            pad_to = compute_max_image_size([p for p, _ in images_and_ts])
            image_paths = [p for p, _ in images_and_ts]

            if not master_video.exists():
                aggs = self.load_aggregations_jsonl(config.agg_jsonl)
                create_video_from_images(
                    image_paths,
                    master_video,
                    fps=fps,
                    pad_to=pad_to,
                    label_video=label_video,
                    aggregations=aggs if label_video else None
                )

            # Split into chunks
            chunks_dir = config.out_chunks_dir / "video_chunks"
            video_chunks = split_video(master_video, chunk_duration, chunks_dir)

            # Load and chunk aggregations
            aggs = self.load_aggregations_jsonl(config.agg_jsonl)
            agg_chunks = self.chunk_aggregations(aggs, chunk_start=global_start, chunk_duration=chunk_duration)

            # Create tasks for each chunk
            with open(Path(__file__).parent / "prompt.txt", 'r') as f:
                prompt_template = f.read()

            for i, vpath in enumerate(video_chunks):
                chunk_out_dir = config.out_chunks_dir / f"chunk_{i:03d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                this_aggs = agg_chunks[i] if i < len(agg_chunks) else []

                # Build prompt
                prompts = []
                for j, a in enumerate(this_aggs):
                    mss = f"{j // 60:02}:{j % 60:02}"
                    prompts.append(a.to_prompt(mss))
                full_prompt = "".join(prompts)
                full_prompt = prompt_template.replace("{{LOGS}}", full_prompt)

                task = SessionTask(
                    session_id=session_id,
                    chunk_index=i,
                    video_path=vpath,
                    prompt=full_prompt,
                    aggregations=this_aggs,
                    output_dir=chunk_out_dir,
                )
                all_tasks.append(task)

        print(f"[ParallelProcessor] Total tasks: {len(all_tasks)}")

        # Step 2: Process all tasks in parallel
        if self.use_batching:
            results = self._process_with_batching(all_tasks)
        else:
            results = self._process_with_workers(all_tasks)

        # Step 3: Group results by session and save summaries
        return self._save_results(results, session_configs)

    def _process_with_workers(self, tasks: List[SessionTask]) -> List[Tuple[SessionTask, Any]]:
        """Process tasks using concurrent workers (for Gemini)."""
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_task, task): task
                for task in tasks
            }

            # Collect results with progress bar
            with tqdm(total=len(tasks), desc="Processing chunks") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append((task, result))
                    except Exception as e:
                        print(f"\n[Error] Task {task.session_id}/chunk_{task.chunk_index} failed: {e}")
                        results.append((task, {"error": str(e)}))
                    pbar.update(1)

        return results

    def _process_with_batching(self, tasks: List[SessionTask]) -> List[Tuple[SessionTask, Any]]:
        """Process tasks using batch generation (for vLLM)."""
        results = []

        # Process in batches
        for i in tqdm(range(0, len(tasks), self.batch_size), desc="Processing batches"):
            batch = tasks[i:i + self.batch_size]

            # Prepare batch inputs
            prompts = [task.prompt for task in batch]
            file_descriptors = []

            for task in batch:
                try:
                    file_desc = self.prompt_client.upload_file(str(task.video_path))
                    file_descriptors.append(file_desc)
                except Exception as e:
                    print(f"\n[Warning] Upload failed for {task.session_id}/chunk_{task.chunk_index}: {e}")
                    file_descriptors.append(None)

            # Batch generation
            try:
                responses = self.prompt_client.generate_content(
                    prompts,
                    file_descriptor=file_descriptors,
                    response_mime_type="application/json"
                )

                # Parse responses
                for task, resp in zip(batch, responses):
                    try:
                        if hasattr(resp, 'json'):
                            body = resp.json if not callable(resp.json) else resp.json()
                        else:
                            body = json.loads(getattr(resp, 'text', str(resp)))
                        results.append((task, body))
                    except Exception as e:
                        print(f"\n[Error] Failed to parse response for {task.session_id}/chunk_{task.chunk_index}: {e}")
                        results.append((task, {"error": str(e)}))

            except Exception as e:
                print(f"\n[Error] Batch generation failed: {e}")
                for task in batch:
                    results.append((task, {"error": str(e)}))

        return results

    def _process_single_task(self, task: SessionTask) -> Any:
        """Process a single task (used by worker threads)."""
        # Upload file
        file_descriptor = None
        try:
            file_descriptor = self.prompt_client.upload_file(str(task.video_path))
        except Exception as e:
            print(f"\n[Warning] Upload failed for {task.session_id}/chunk_{task.chunk_index}: {e}")

        # Generate
        try:
            resp = self.prompt_client.generate_content(
                task.prompt,
                file_descriptor=file_descriptor,
                response_mime_type="application/json"
            )

            if hasattr(resp, 'json'):
                body = resp.json if not callable(resp.json) else resp.json()
            else:
                body = json.loads(getattr(resp, 'text', str(resp)))
            return body

        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def _save_results(
        self,
        results: List[Tuple[SessionTask, Any]],
        session_configs: List[SessionConfig]
    ) -> Dict[str, List[Dict]]:
        """Save results and create summaries for each session."""
        # Group by session
        session_results = {}

        for task, result in results:
            if task.session_id not in session_results:
                session_results[task.session_id] = []

            # Save individual chunk results
            agg_output = task.output_dir / "aggregations.json"
            with open(agg_output, 'w') as f:
                json.dump([a.to_dict() for a in task.aggregations], f, indent=2, ensure_ascii=False)

            result_output = task.output_dir / "generation_result.json"
            with open(result_output, 'w') as f:
                json.dump({
                    "chunk_index": task.chunk_index,
                    "video_chunk": str(task.video_path),
                    "aggregations_count": len(task.aggregations),
                    "result": result,
                }, f, indent=2, ensure_ascii=False)

            prompt_output = task.output_dir / "prompt.txt"
            with open(prompt_output, 'w') as f:
                f.write(task.prompt)

            session_results[task.session_id].append({
                "chunk_index": task.chunk_index,
                "video_chunk": str(task.video_path),
                "aggregations_count": len(task.aggregations),
                "aggregations_file": str(agg_output),
                "result_file": str(result_output),
                "prompt_file": str(prompt_output),
            })

        # Save session summaries
        for config in session_configs:
            session_id = config.session_folder.name
            if session_id in session_results:
                summary_path = config.out_chunks_dir / "all_chunks_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(
                        sorted(session_results[session_id], key=lambda x: x["chunk_index"]),
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                print(f"[ParallelProcessor] Saved summary for {session_id}: {summary_path}")

        return session_results

    def load_aggregations_jsonl(self, path: Path) -> List[ProcessedAggregation]:
        """Load aggregations from a JSONL file."""
        out = []
        with open(path, 'r') as f:
            for raw_line in f:
                try:
                    line = json.loads(raw_line.strip())
                except Exception:
                    print("Skipping invalid json line")
                    continue

                r = AggregationRequest(
                    timestamp=line['timestamp'],
                    end_timestamp=None,
                    reason=line['reason'],
                    event_type=line['event_type'],
                    is_start=line['is_start'],
                    screenshot=None,
                    screenshot_path=line['screenshot_path'],
                    monitor=line.get('monitor'),
                    burst_id=line.get('burst_id'),
                )
                events = line['events']
                out.append(ProcessedAggregation(request=r, events=events))
        return out

    def chunk_aggregations(
        self,
        aggs: List[ProcessedAggregation],
        chunk_start: float,
        chunk_duration: int
    ) -> List[List[ProcessedAggregation]]:
        """Partition aggregations into time-based chunks."""
        if not aggs:
            return []
        min_ts = min(a.request.timestamp for a in aggs)
        max_ts = max(a.request.timestamp for a in aggs)
        if chunk_start is None:
            chunk_start = min_ts
        num_chunks = max(1, math.ceil((max_ts - chunk_start) / float(chunk_duration)))
        chunks: List[List[ProcessedAggregation]] = [[] for _ in range(num_chunks)]
        for a in aggs:
            idx = int(math.floor((a.request.timestamp - chunk_start) / float(chunk_duration)))
            if idx < 0:
                idx = 0
            if idx >= len(chunks):
                extend_by = idx - len(chunks) + 1
                chunks.extend([[] for _ in range(extend_by)])
            chunks[idx].append(a)
        return chunks

    def list_screenshots(self, session_folder: Path) -> List[Path]:
        """List all screenshot files in the session folder."""
        screenshots = []
        for p in sorted((session_folder / "screenshots").iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                screenshots.append(p)
        return screenshots

    def extract_timestamp_from_filename(self, p: Path) -> Optional[float]:
        """Extract timestamp from filename or use file modification time."""
        _timestamp_re = re.compile(r"(\d+\.\d+)")
        m = _timestamp_re.search(p.name)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        try:
            return p.stat().st_mtime
        except Exception:
            return None
