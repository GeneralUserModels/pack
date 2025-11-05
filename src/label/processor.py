import re
import json
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from label.models import SessionConfig, ChunkTask, Caption, Aggregation, VideoPath, MatchedCaption
from label.video import create_video, split_video, compute_max_size
from label.clients import VLMClient, CAPTION_SCHEMA


class Processor:
    def __init__(
        self,
        client: VLMClient,
        num_workers: int = 4,
        video_only: bool = False,
        prompt_file: str = "prompts/default.txt",
    ):
        self.client = client
        self.num_workers = num_workers
        self.video_only = video_only
        self.prompt = self._load_prompt(prompt_file)

    def _load_prompt(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        return p.read_text()

    def process_sessions(
        self,
        configs: List[SessionConfig],
        fps: int = 1,
        annotate: bool = False
    ) -> dict:

        tasks = []

        for config in tqdm(configs, desc="Preparing"):
            config.ensure_dirs()

            if self.video_only:
                tasks.extend(self._prepare_video_only(config))
            else:
                tasks.extend(self._prepare_standard(config, fps, annotate))

        print(f"[Processor] Processing {len(tasks)} chunks with {min(self.num_workers, len(tasks))} concurrent workers")

        results = self._process_tasks(tasks)

        return self._save_results(results, configs, fps)

    def _prepare_standard(
        self,
        config: SessionConfig,
        fps: int,
        annotate: bool
    ) -> List[ChunkTask]:

        aggs = config.load_aggregations() if annotate else None
        image_paths = [Path(agg.screenshot_path) for agg in aggs if agg.screenshot_path and Path(agg.screenshot_path).exists()]
        global_start = self._extract_timestamp(image_paths[0]) if image_paths else 0.0
        if not config.master_video_path.exists():
            pad_to = compute_max_size(image_paths)

            create_video(
                image_paths, config.master_video_path, fps=fps,
                pad_to=pad_to, annotate=annotate, aggregations=aggs,
                session_dir=config.session_folder
            )

        chunks = split_video(config.master_video_path, config.chunk_duration, config.chunks_dir)
        aggs = config.load_aggregations()
        agg_chunks = self._chunk_aggregations(aggs, global_start, config.chunk_duration)

        tasks = []
        for i, video_path in enumerate(chunks):
            chunk_aggs = agg_chunks[i] if i < len(agg_chunks) else []

            prompt_lines = []
            for j, agg in enumerate(chunk_aggs):
                time_str = f"{j // 60:02}:{j % 60:02}"
                prompt_lines.append(agg.to_prompt(time_str))

            full_prompt = self.prompt.replace("{{LOGS}}", "".join(prompt_lines))
            with open(video_path.parent / f"prompt_{i:03d}.txt", 'w') as f:
                f.write(full_prompt)

            tasks.append(ChunkTask(
                session_id=config.session_id,
                chunk_index=i,
                video_path=VideoPath(video_path),
                prompt=full_prompt,
                aggregations=chunk_aggs,
                chunk_start_time=i * config.chunk_duration,
                chunk_duration=config.chunk_duration
            ))

        return tasks

    def _prepare_video_only(self, config: SessionConfig) -> List[ChunkTask]:
        if not config.video_path or not config.video_path.exists():
            return []

        chunks = split_video(config.video_path.resolve(), config.chunk_duration, config.chunks_dir)

        tasks = []
        for i, video_path in enumerate(chunks):
            tasks.append(ChunkTask(
                session_id=config.session_id,
                chunk_index=i,
                video_path=VideoPath(video_path),
                prompt=self.prompt,
                aggregations=[],
                chunk_start_time=i * config.chunk_duration,
                chunk_duration=config.chunk_duration
            ))

        return tasks

    def _process_tasks(self, tasks: List[ChunkTask]) -> List[Tuple[ChunkTask, any]]:
        """Process tasks with configurable concurrency using num_workers."""
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

    def _process_single_task(self, task: ChunkTask) -> any:
        """Process single task with schema."""
        file_desc = self.client.upload_file(str(task.video_path.resolve()))
        response = self.client.generate(task.prompt, file_desc, schema=CAPTION_SCHEMA)

        result = response.json if not callable(response.json) else response.json()
        return result

    def _save_results(
        self, results: List[Tuple[ChunkTask, any]],
        configs: List[SessionConfig],
        fps: int
    ) -> dict:

        session_captions = {}

        for task, result in results:
            if task.session_id not in session_captions:
                session_captions[task.session_id] = []

            config = next(c for c in configs if c.session_id == task.session_id)

            if not self.video_only and task.aggregations:
                agg_file = config.aggregations_dir / f"{task.chunk_index:03d}.json"
                with open(agg_file, 'w') as f:
                    json.dump([a.to_dict() for a in task.aggregations], f, indent=2)

            result_file = config.captions_dir / f"{task.chunk_index:03d}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    "chunk_index": task.chunk_index,
                    "result": result
                }, f, indent=2)

            captions = self._extract_captions(result, task)
            session_captions[task.session_id].extend(captions)

        for config in configs:
            if config.session_id in session_captions:
                captions = session_captions[config.session_id]
                captions.sort(key=lambda c: c.start_seconds)
                config.save_captions(captions)

                if not self.video_only:
                    self._create_matched_captions(config, captions, fps)

        return {sid: len(caps) for sid, caps in session_captions.items()}

    def _extract_captions(self, result: any, task: ChunkTask) -> List[Caption]:
        captions = []

        if isinstance(result, str) or not isinstance(result, list):
            return captions

        for item in result:
            start_str = item.get("start", "00:00")
            end_str = item.get("end", start_str)

            try:
                mins, secs = map(int, start_str.split(":"))
                rel_start = mins * 60 + secs
            except Exception:
                rel_start = 0

            try:
                mins, secs = map(int, end_str.split(":"))
                rel_end = mins * 60 + secs
            except Exception:
                rel_end = rel_start

            abs_start = task.chunk_start_time + rel_start
            abs_end = task.chunk_start_time + rel_end

            captions.append(Caption(
                start_seconds=abs_start,
                end_seconds=abs_end,
                text=item.get("caption", item.get("description", "")),
                chunk_index=task.chunk_index
            ))

        return captions

    def _create_matched_captions(
        self,
        config: SessionConfig,
        captions: List[Caption],
        fps: int
    ):
        aggs = config.load_aggregations()
        if not aggs:
            return
        aggs.sort(key=lambda x: x.timestamp)
        matched = []

        for caption in captions:
            start_idx = int(caption.start_seconds * fps)
            end_idx = int(caption.end_seconds * fps)
            start_idx = max(0, min(start_idx, len(aggs) - 1))
            end_idx = max(start_idx, min(end_idx, len(aggs) - 1))
            matched_aggs = aggs[start_idx:end_idx + 1]
            matched.append(MatchedCaption(
                caption=caption,
                aggregations=matched_aggs,
                start_index=start_idx,
                end_index=end_idx
            ))

        if matched:
            covered_indices = set()
            for match in matched:
                covered_indices.update(range(match.start_index, match.end_index + 1))

            remaining_indices = [i for i in range(len(aggs)) if i not in covered_indices]

            if remaining_indices:
                last_match = matched[-1]
                remaining_aggs = [aggs[i] for i in remaining_indices if i > last_match.end_index]

                if remaining_aggs:
                    last_match.aggregations.extend(remaining_aggs)
                    last_match.end_index = len(aggs) - 1

        config.save_matched_captions(matched)

    def _chunk_aggregations(
        self,
        aggs: List[Aggregation],
        start_time: float,
        chunk_duration: int
    ) -> List[List[Aggregation]]:

        if not aggs:
            return []
        return [aggs[i:i + chunk_duration] for i in range(0, len(aggs), chunk_duration)]

    def _extract_timestamp(self, path: Path) -> float:
        m = re.search(r'(\d+\.\d+)', path.name)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

        try:
            return path.stat().st_mtime
        except Exception:
            return None
