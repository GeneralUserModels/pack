from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from tqdm import tqdm

from label.clients import PromptClient
from record.models.aggregation import ProcessedAggregation, AggregationRequest
from label.utils import SessionTask, CaptionEntry


class SessionProcessor:
    """Handles parallel processing of sessions with VLM."""

    def __init__(
        self,
        prompt_client: PromptClient,
        num_workers: int = 4,
        batch_size: int = 8,
        use_batching: bool = False,
        video_only_mode: bool = False,
        video_only_prompt_file: str = "prompts/video_only_prompt.txt",
    ):
        self.prompt_client = prompt_client
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_batching = use_batching
        self.video_only_mode = video_only_mode
        self.video_only_prompt_file = video_only_prompt_file

    def process_multiple_sessions(
        self,
        session_configs: List,
        chunk_duration: int = 60,
        fps: int = 1,
        label_video: bool = False,
    ) -> Dict[str, List[Dict]]:
        """Process multiple sessions in parallel."""
        if self.video_only_mode:
            tasks = self._prepare_video_only_tasks(session_configs, chunk_duration)
        else:
            tasks = self._prepare_standard_tasks(session_configs, chunk_duration, fps, label_video)

        print(f"[Processor] Total tasks: {len(tasks)}")
        print(f"[Processor] Mode: {'Batching' if self.use_batching else 'Concurrent'}")

        if self.use_batching:
            results = self._process_with_batching(tasks)
        else:
            results = self._process_with_workers(tasks)

        return self._save_results(results, session_configs)

    def _prepare_standard_tasks(
        self,
        session_configs: List,
        chunk_duration: int,
        fps: int,
        label_video: bool,
    ) -> List[SessionTask]:
        """Prepare tasks for standard mode (screenshots + logs)."""
        from label.video import (
            compute_max_image_size,
            create_video_from_images,
            split_video,
        )
        from label.utils import list_screenshots, extract_timestamp_from_filename

        tasks = []
        prompt_template = self._load_prompt_template("prompts/prompt.txt")

        for config in tqdm(session_configs, desc="Preparing sessions"):
            session_id = config.session_folder.name
            screenshots = list_screenshots(config.session_folder)

            if not screenshots:
                print(f"[Warning] No screenshots in {session_id}")
                continue

            # Sort screenshots by timestamp
            images_and_ts = [
                (p, extract_timestamp_from_filename(p))
                for p in screenshots
            ]
            images_and_ts = [(p, ts) for p, ts in images_and_ts if ts is not None]
            images_and_ts.sort(key=lambda x: x[1])

            global_start = images_and_ts[0][1]
            image_paths = [p for p, _ in images_and_ts]

            # Create master video
            master_video = config.out_chunks_dir / "full_session.mp4"
            if not master_video.exists():
                pad_to = compute_max_image_size(image_paths)
                aggs = self._load_aggregations(config.agg_jsonl)
                create_video_from_images(
                    image_paths, master_video, fps=fps, pad_to=pad_to,
                    label_video=label_video,
                    aggregations=aggs if label_video else None
                )

            # Split into chunks
            chunks_dir = config.out_chunks_dir / "video_chunks"
            video_chunks = split_video(master_video, chunk_duration, chunks_dir)

            # Chunk aggregations
            aggs = self._load_aggregations(config.agg_jsonl)
            agg_chunks = self._chunk_aggregations(aggs, global_start, chunk_duration)

            # Create tasks
            for i, vpath in enumerate(video_chunks):
                chunk_out_dir = config.out_chunks_dir / f"chunk_{i:03d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                this_aggs = agg_chunks[i] if i < len(agg_chunks) else []

                # Build prompt
                prompts = []
                for j, a in enumerate(this_aggs):
                    mss = f"{j // 60:02}:{j % 60:02}"
                    prompts.append(a.to_prompt(mss))
                full_prompt = prompt_template.replace("{{LOGS}}", "".join(prompts))

                tasks.append(SessionTask(
                    session_id=session_id,
                    chunk_index=i,
                    video_path=vpath,
                    prompt=full_prompt,
                    aggregations=this_aggs,
                    output_dir=chunk_out_dir,
                    chunk_start_time=i * chunk_duration,
                    chunk_duration=chunk_duration,
                ))

        return tasks

    def _prepare_video_only_tasks(
        self,
        session_configs: List,
        chunk_duration: int,
    ) -> List[SessionTask]:
        """Prepare tasks for video-only mode."""
        from label.video import split_video

        tasks = []
        prompt_template = self._load_prompt_template(self.video_only_prompt_file)

        for config in tqdm(session_configs, desc="Preparing video sessions"):
            session_id = config.session_folder.name

            if not config.video_path or not config.video_path.exists():
                print(f"[Warning] No video for {session_id}")
                continue

            # Split video
            chunks_dir = config.out_chunks_dir / "video_chunks"
            video_chunks = split_video(config.video_path, chunk_duration, chunks_dir)

            # Create tasks
            for i, vpath in enumerate(video_chunks):
                chunk_out_dir = config.out_chunks_dir / f"chunk_{i:03d}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)

                tasks.append(SessionTask(
                    session_id=session_id,
                    chunk_index=i,
                    video_path=vpath,
                    prompt=prompt_template,
                    aggregations=[],
                    output_dir=chunk_out_dir,
                    chunk_start_time=i * chunk_duration,
                    chunk_duration=chunk_duration,
                ))

        return tasks

    def _process_with_workers(self, tasks: List[SessionTask]) -> List[Tuple[SessionTask, Any]]:
        """Process using concurrent workers (Gemini)."""
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_task, task): task
                for task in tasks
            }

            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append((task, result))
                    except Exception as e:
                        print(f"\n[Error] {task.session_id}/chunk_{task.chunk_index}: {e}")
                        results.append((task, {"error": str(e)}))
                    pbar.update(1)

        return results

    def _process_with_batching(self, tasks: List[SessionTask]) -> List[Tuple[SessionTask, Any]]:
        """Process using batch generation (vLLM)."""
        results = []

        for i in tqdm(range(0, len(tasks), self.batch_size), desc="Processing batches"):
            batch = tasks[i:i + self.batch_size]

            prompts = [task.prompt for task in batch]
            file_descriptors = []

            for task in batch:
                try:
                    file_desc = self.prompt_client.upload_file(str(task.video_path))
                    file_descriptors.append(file_desc)
                except Exception as e:
                    print(f"\n[Warning] Upload failed: {e}")
                    file_descriptors.append(None)

            try:
                responses = self.prompt_client.generate_content(
                    prompts,
                    file_descriptor=file_descriptors,
                    response_mime_type="application/json"
                )

                for task, resp in zip(batch, responses):
                    try:
                        body = resp.json if not callable(resp.json) else resp.json()
                        results.append((task, body))
                    except Exception as e:
                        print(f"\n[Error] Parse failed: {e}")
                        results.append((task, {"error": str(e)}))

            except Exception as e:
                print(f"\n[Error] Batch failed: {e}")
                for task in batch:
                    results.append((task, {"error": str(e)}))

        return results

    def _process_single_task(self, task: SessionTask) -> Any:
        """Process single task (for workers)."""
        file_descriptor = None
        try:
            file_descriptor = self.prompt_client.upload_file(str(task.video_path))
        except Exception as e:
            print(f"\n[Warning] Upload failed: {e}")

        resp = self.prompt_client.generate_content(
            task.prompt,
            file_descriptor=file_descriptor,
            response_mime_type="application/json"
        )

        return json.loads(getattr(resp, 'text', str(resp)))

    def _save_results(
        self,
        results: List[Tuple[SessionTask, Any]],
        session_configs: List
    ) -> Dict[str, List[Dict]]:
        """Save results and create summaries."""
        session_results = {}
        session_captions = {}

        for task, result in results:
            if task.session_id not in session_results:
                session_results[task.session_id] = []
                session_captions[task.session_id] = []

            # Save aggregations (skip in video-only mode)
            if not self.video_only_mode:
                agg_output = task.output_dir / "aggregations.json"
                with open(agg_output, 'w') as f:
                    json.dump([a.to_dict() for a in task.aggregations], f, indent=2)

            # Save generation result
            result_output = task.output_dir / "generation_result.json"
            with open(result_output, 'w') as f:
                json.dump({
                    "chunk_index": task.chunk_index,
                    "video_chunk": str(task.video_path),
                    "result": result,
                }, f, indent=2)

            # Save prompt
            prompt_output = task.output_dir / "prompt.txt"
            with open(prompt_output, 'w') as f:
                f.write(task.prompt)

            # Extract captions
            captions = self._extract_captions(result, task)
            session_captions[task.session_id].extend(captions)

            session_results[task.session_id].append({
                "chunk_index": task.chunk_index,
                "video_chunk": str(task.video_path),
                "result_file": str(result_output),
            })

        # Save session summaries and captions
        for config in session_configs:
            session_id = config.session_folder.name
            if session_id in session_results:
                # Summary
                summary_path = config.out_chunks_dir / "all_chunks_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(
                        sorted(session_results[session_id], key=lambda x: x["chunk_index"]),
                        f, indent=2
                    )

                # Captions
                captions_path = config.session_folder / "captions.jsonl"
                self._save_captions(session_captions[session_id], captions_path)

                print(f"[Processor] ✓ Saved {session_id}: {summary_path}")

        return session_results

    def _extract_captions(self, result: Any, task: SessionTask) -> List[CaptionEntry]:
        """Extract and adjust caption timestamps."""
        captions = []
        if isinstance(result, str):
            return captions

        for item in result:
            start_str = item.get("start", "00:00")
            end_str = item.get("end", start_str)

            try:
                mins, secs = map(int, start_str.split(":"))
                rel_start = mins * 60 + secs
            except:
                rel_start = 0

            try:
                mins, secs = map(int, end_str.split(":"))
                rel_end = mins * 60 + secs
            except:
                rel_end = rel_start

            # Adjust to absolute time
            abs_start = task.chunk_start_time + rel_start
            abs_end = task.chunk_start_time + rel_end

            captions.append(CaptionEntry(
                timestamp_seconds=abs_start,
                timestamp_formatted=f"{int(abs_start // 60):02d}:{int(abs_start % 60):02d}",
                caption=item.get("caption", item.get("description", "")),
                chunk_index=task.chunk_index,
                metadata={
                    "end_timestamp": f"{int(abs_end // 60):02d}:{int(abs_end % 60):02d}",
                    "end_timestamp_seconds": abs_end,
                }
            ))

        return captions

    def _save_captions(self, captions: List[CaptionEntry], output_path: Path):
        """Save captions to JSONL."""
        captions.sort(key=lambda c: c.timestamp_seconds)

        with open(output_path, 'w', encoding='utf-8') as f:
            for caption in captions:
                entry = {
                    "start": caption.timestamp_formatted,
                    "end": caption.metadata.get("end_timestamp", caption.timestamp_formatted),
                    "start_seconds": caption.timestamp_seconds,
                    "end_seconds": caption.metadata.get("end_timestamp_seconds", caption.timestamp_seconds),
                    "caption": caption.caption,
                    "chunk_index": caption.chunk_index,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _load_aggregations(self, path: Path) -> List[ProcessedAggregation]:
        """Load aggregations from JSONL."""
        out = []
        with open(path, 'r') as f:
            for raw_line in f:
                try:
                    line = json.loads(raw_line.strip())
                except:
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
                out.append(ProcessedAggregation(request=r, events=line['events']))
        return out

    def _chunk_aggregations(
        self,
        aggs: List[ProcessedAggregation],
        chunk_start: float,
        chunk_duration: int
    ) -> List[List[ProcessedAggregation]]:
        """Partition aggregations into time chunks."""
        if not aggs:
            return []

        min_ts = min(a.request.timestamp for a in aggs)
        max_ts = max(a.request.timestamp for a in aggs)
        num_chunks = max(1, math.ceil((max_ts - chunk_start) / float(chunk_duration)))

        chunks = [[] for _ in range(num_chunks)]

        for a in aggs:
            idx = int((a.request.timestamp - chunk_start) / float(chunk_duration))
            idx = max(0, idx)

            if idx >= len(chunks):
                extend_by = idx - len(chunks) + 1
                chunks.extend([[] for _ in range(extend_by)])

            chunks[idx].append(a)

        return chunks

    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path(__file__).parent / prompt_file
        if not prompt_path.exists():
            raise RuntimeError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            return f.read()
