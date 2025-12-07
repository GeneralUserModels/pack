import re
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from label.models import SessionConfig, ChunkTask, Caption, Aggregation, VideoPath, MatchedCaption
from label.video import create_video, split_video, compute_max_size
from label.clients import VLMClient, CAPTION_SCHEMA


class Processor:
    def __init__(
        self,
        client: VLMClient,
        encode_workers: int = 8,
        label_workers: int = 4,
        screenshots_only: bool = False,
        prompt_file: str = "prompts/default.txt",
        max_time_gap: float = 300.0,
        hash_cache_path: Optional[str] = None,
        dedupe_threshold: int = 1,
    ):
        self.client = client
        self.encode_workers = encode_workers
        self.label_workers = label_workers
        self.screenshots_only = screenshots_only
        self.prompt = self._load_prompt(prompt_file)
        self.max_time_gap = max_time_gap
        self.dedupe_threshold = dedupe_threshold
        self.hash_map = self._load_hash_cache(hash_cache_path) if hash_cache_path else None

    def _load_prompt(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent / path
        return p.read_text()

    def _load_hash_cache(self, cache_path: str) -> Optional[Dict[str, int]]:
        """Load hash cache from JSON file. Returns path -> hash_int mapping.
        
        Normalizes paths to use the last 3 components (user/screenshots/filename)
        as the key for robust matching regardless of absolute/relative paths.
        """
        p = Path(cache_path)
        if not p.exists():
            print(f"[Warning] Hash cache not found: {cache_path}")
            return None
        
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            entries = data.get("entries", {})
            hash_map = {}
            for path, info in entries.items():
                hash_int = info.get("hash_int")
                if hash_int is not None:
                    # Normalize: use last 3 path components (user/screenshots/filename)
                    parts = Path(path).parts
                    if len(parts) >= 3:
                        key = str(Path(*parts[-3:]))
                    else:
                        key = path
                    hash_map[key] = hash_int
            
            print(f"[Hash] Loaded {len(hash_map):,} hashes from cache")
            return hash_map
        except Exception as e:
            print(f"[Warning] Failed to load hash cache: {e}")
            return None
    
    def _get_hash_key(self, img_path: Path) -> str:
        """Get normalized key for hash lookup (last 3 path components)."""
        parts = img_path.parts
        if len(parts) >= 3:
            return str(Path(*parts[-3:]))
        return str(img_path)

    def _hamming_distance(self, h1: int, h2: int) -> int:
        """Compute hamming distance between two integer hashes."""
        return (h1 ^ h2).bit_count()

    def _dedupe_images_by_hash(self, image_files: List[Path]) -> List[Path]:
        """
        Filter consecutive images that are too similar (hamming distance <= dedupe_threshold).
        
        Only keeps images where the hash difference from the previous kept image is > threshold.
        This collapses runs of near-identical images into a single representative.
        """
        if not self.hash_map or not image_files:
            return image_files
        
        original_count = len(image_files)
        kept_images = []
        last_kept_hash = None
        
        for img_path in image_files:
            hash_key = self._get_hash_key(img_path)
            curr_hash = self.hash_map.get(hash_key)
            
            if curr_hash is None:
                # No hash available, keep the image
                kept_images.append(img_path)
                last_kept_hash = None
                continue
            
            if last_kept_hash is None:
                # First image with a hash, always keep
                kept_images.append(img_path)
                last_kept_hash = curr_hash
                continue
            
            distance = self._hamming_distance(last_kept_hash, curr_hash)
            
            if distance > self.dedupe_threshold:
                # Different enough, keep it
                kept_images.append(img_path)
                last_kept_hash = curr_hash
            # else: skip this image (too similar to last kept)
        
        kept_count = len(kept_images)
        dropped_count = original_count - kept_count
        pct_saved = (dropped_count / original_count * 100) if original_count > 0 else 0
        
        print(f"[Dedupe] Kept {kept_count:,} of {original_count:,} images "
              f"(dropped {dropped_count:,}, saved {pct_saved:.1f}%) "
              f"[threshold={self.dedupe_threshold}]")
        

        
        return kept_images

    def process_sessions(
        self,
        configs: List[SessionConfig],
        fps: int = 1,
        annotate: bool = False,
        encode_only: bool = False
    ) -> dict:

        tasks = []
        config_map = {c.session_id: c for c in configs}

        for config in tqdm(configs, desc="Preparing"):
            config.ensure_dirs()

            if self.screenshots_only:
                tasks.extend(self._prepare_screenshots_only(config, fps))
            else:
                tasks.extend(self._prepare_standard(config, fps, annotate))

        print(f"[Processor] Prepared {len(tasks)} chunks (encode: {self.encode_workers} workers)")

        if encode_only:
            print(f"[Encode-only] Stopping after encoding. Run again without --encode-only to label.")
            return {c.session_id: 0 for c in configs}

        print(f"[Processor] Labeling with {self.label_workers} workers")
        results = self._process_tasks(tasks, config_map)

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

    def _prepare_screenshots_only(self, config: SessionConfig, fps: int) -> List[ChunkTask]:
        if not config.screenshots_dir or not config.screenshots_dir.exists():
            return []

        # Get all image files from the screenshots directory
        image_files = sorted([
            f for f in config.screenshots_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not image_files:
            return []

        # Deduplicate consecutive similar images using hash cache
        if self.hash_map:
            image_files = self._dedupe_images_by_hash(image_files)

        if not image_files:
            return []

        # Group images by time gaps (split if > max_time_gap seconds apart)
        image_segments = self._split_images_by_time_gap(image_files, max_gap_seconds=self.max_time_gap)
        
        print(f"\n[Segments] Created {len(image_segments)} segment(s) from {len(image_files)} images (max gap: {self.max_time_gap}s)")
        for idx, seg in enumerate(image_segments):
            print(f"  Segment {idx}: {len(seg)} images")
        
        # At 1 fps, chunk_duration seconds = chunk_duration images
        images_per_chunk = config.chunk_duration * fps

        # Build list of all chunks to create (across all segments)
        chunk_jobs = []
        chunk_index = 0
        cumulative_time = 0

        for segment_idx, segment_images in enumerate(image_segments):
            if not segment_images:
                continue

            segment_duration = len(segment_images) / fps
            num_chunks = math.ceil(len(segment_images) / images_per_chunk)
            
            print(f"[Chunk] Segment {segment_idx}: Preparing {num_chunks} chunk(s) from {len(segment_images)} images")

            for i in range(num_chunks):
                start_img_idx = i * images_per_chunk
                end_img_idx = min((i + 1) * images_per_chunk, len(segment_images))
                chunk_images = segment_images[start_img_idx:end_img_idx]
                
                if not chunk_images:
                    continue

                chunk_video_path = config.chunks_dir / f"{chunk_index:03d}.mp4"
                chunk_start_in_segment = i * config.chunk_duration
                actual_chunk_duration = len(chunk_images) / fps
                
                chunk_jobs.append({
                    'chunk_index': chunk_index,
                    'chunk_images': chunk_images,
                    'chunk_video_path': chunk_video_path,
                    'chunk_start_time': cumulative_time + chunk_start_in_segment,
                    'actual_duration': actual_chunk_duration,
                    'fps': fps
                })
                chunk_index += 1

            cumulative_time += segment_duration

        # Parallel video creation
        def create_chunk_video(job):
            if not job['chunk_video_path'].exists():
                create_video(
                    job['chunk_images'],
                    job['chunk_video_path'],
                    fps=job['fps'],
                    pad_to=None,
                    annotate=False,
                    aggregations=None,
                    session_dir=None
                )
            return job

        print(f"[Encode] Creating {len(chunk_jobs)} chunk videos with {self.encode_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=self.encode_workers) as executor:
            completed_jobs = list(tqdm(
                executor.map(create_chunk_video, chunk_jobs),
                total=len(chunk_jobs),
                desc="Encoding chunks"
            ))

        # Build tasks from completed jobs
        tasks = []
        for job in completed_jobs:
            tasks.append(ChunkTask(
                session_id=config.session_id,
                chunk_index=job['chunk_index'],
                video_path=VideoPath(job['chunk_video_path']),
                prompt=self.prompt,
                aggregations=[],
                chunk_start_time=job['chunk_start_time'],
                chunk_duration=int(job['actual_duration'])
            ))

        return tasks
    
    def _split_images_by_time_gap(self, image_files: List[Path], max_gap_seconds: float = 30) -> List[List[Path]]:
        """
        Split images into segments based on time gaps between consecutive images.
        If two images are more than max_gap_seconds apart, start a new segment.
        
        Args:
            image_files: List of image file paths (should be sorted)
            max_gap_seconds: Maximum time gap in seconds before forcing a split
            
        Returns:
            List of image segments, where each segment is a list of consecutive images
        """
        if not image_files:
            return []
        
        segments = []
        current_segment = [image_files[0]]
        prev_timestamp = self._extract_timestamp_from_filename(image_files[0])
        
        for img_path in image_files[1:]:
            curr_timestamp = self._extract_timestamp_from_filename(img_path)
            
            # If we can't parse timestamps, just keep adding to current segment
            if prev_timestamp is None or curr_timestamp is None:
                current_segment.append(img_path)
                continue
            
            # Check time gap
            time_gap = abs(curr_timestamp - prev_timestamp)
            
            if time_gap > max_gap_seconds:
                # Start a new segment
                print(f"[Split] Time gap detected: {time_gap:.1f}s between screenshots (threshold: {max_gap_seconds}s)")
                print(f"  Previous: {image_files[image_files.index(img_path)-1].name}")
                print(f"  Current:  {img_path.name}")
                segments.append(current_segment)
                current_segment = [img_path]
            else:
                current_segment.append(img_path)
            
            prev_timestamp = curr_timestamp
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _extract_timestamp_from_filename(self, path: Path) -> Optional[float]:
        """
        Extract timestamp from filename. Supports multiple formats:
        1. Float timestamp: 1760702571.228687_reason_move_start.jpg
        2. DateTime format: w5_6713_sstetler1@msn.com20200810004157314.jpg (YYYYMMDDHHMMSSmmm)
        
        Returns timestamp as float (seconds since epoch) or None if unable to parse
        """
        filename = path.name
        
        # Try format 1: float timestamp at the beginning
        m = re.search(r'^(\d+\.\d+)', filename)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        
        # Try format 2: YYYYMMDDHHMMSSmmm datetime format
        # Look for 17 consecutive digits (YYYYMMDDHHMMSSMMM)
        m = re.search(r'(\d{17})', filename)
        if m:
            try:
                timestamp_str = m.group(1)
                # Parse: YYYYMMDDHHMMSSMMM
                year = int(timestamp_str[0:4])
                month = int(timestamp_str[4:6])
                day = int(timestamp_str[6:8])
                hour = int(timestamp_str[8:10])
                minute = int(timestamp_str[10:12])
                second = int(timestamp_str[12:14])
                millisecond = int(timestamp_str[14:17])
                
                dt = datetime(year, month, day, hour, minute, second, millisecond * 1000)
                return dt.timestamp()
            except Exception:
                pass
        
        # Fallback: try file modification time
        try:
            return path.stat().st_mtime
        except Exception:
            return None

    def _process_tasks(self, tasks: List[ChunkTask], config_map: dict) -> List[Tuple[ChunkTask, any]]:
        """Process tasks with configurable concurrency using num_workers.
        
        Supports resume: skips tasks that already have caption files.
        Saves incrementally: writes caption file immediately after each task completes.
        """
        results = []
        tasks_to_process = []
        
        # Check for existing captions (resume support)
        for task in tasks:
            config = config_map[task.session_id]
            result_file = config.captions_dir / f"{task.chunk_index:03d}.json"
            
            if result_file.exists():
                # Load existing result
                with open(result_file, 'r') as f:
                    data = json.load(f)
                results.append((task, data.get('result', {})))
            else:
                tasks_to_process.append(task)
        
        if results:
            print(f"[Resume] Loaded {len(results)} existing captions, {len(tasks_to_process)} remaining")
        
        if not tasks_to_process:
            print("[Resume] All chunks already captioned")
            return results

        with ThreadPoolExecutor(max_workers=self.label_workers) as executor:
            future_to_task = {
                executor.submit(self._process_single_task, task): task
                for task in tasks_to_process
            }

            with tqdm(total=len(tasks_to_process), desc="Processing") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    config = config_map[task.session_id]
                    try:
                        result = future.result()
                        results.append((task, result))
                        # Save immediately after processing
                        self._save_chunk_result(config, task, result)
                    except Exception as e:
                        print(f"\n[Error] {task.session_id}/chunk_{task.chunk_index}: {e}")
                        results.append((task, {"error": str(e)}))
                    pbar.update(1)

        return results

    def _process_single_task(self, task: ChunkTask) -> any:
        """Process single task with schema."""
        file_desc = self.client.upload_file(str(task.video_path.resolve()), session_id=task.session_id)
        response = self.client.generate(task.prompt, file_desc, schema=CAPTION_SCHEMA)

        result = response.json if not callable(response.json) else response.json()
        return result

    def _save_chunk_result(self, config: SessionConfig, task: ChunkTask, result: any):
        """Save a single chunk's result immediately after processing."""
        # Save aggregations if applicable (standard mode only)
        if not self.screenshots_only and task.aggregations:
            agg_file = config.aggregations_dir / f"{task.chunk_index:03d}.json"
            with open(agg_file, 'w') as f:
                json.dump([a.to_dict() for a in task.aggregations], f, indent=2)

        # Save caption result
        result_file = config.captions_dir / f"{task.chunk_index:03d}.json"
        with open(result_file, 'w') as f:
            json.dump({
                "chunk_index": task.chunk_index,
                "result": result
            }, f, indent=2)

    def _save_results(
        self, results: List[Tuple[ChunkTask, any]],
        configs: List[SessionConfig],
        fps: int
    ) -> dict:
        """Merge all chunk results and save final captions.
        
        Note: Individual chunk files are already saved incrementally by _save_chunk_result
        or loaded from existing files during resume.
        """
        session_captions = {}

        for task, result in results:
            if task.session_id not in session_captions:
                session_captions[task.session_id] = []

            captions = self._extract_captions(result, task)
            session_captions[task.session_id].extend(captions)

        for config in configs:
            if config.session_id in session_captions:
                captions = session_captions[config.session_id]
                captions.sort(key=lambda c: c.start_seconds)
                config.save_captions(captions)

                if not self.screenshots_only:
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
                end_index=end_idx,
                screenshot_scale_factor=matched_aggs[0].scale_factor if matched_aggs else 1.0
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
