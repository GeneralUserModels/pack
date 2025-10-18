# Pack

**Pack** records and aggregates your computer use — screenshots plus input events (click, keypress, scroll, cursor move). It groups activity into event bursts and uses a VLM pipeline to generate human-readable captions describing what happened.

<img alt="pack_overview" src="https://github.com/user-attachments/assets/5bab2b31-0564-448c-b286-de06859b7c97" />

---

# Quickstart

## Requirements

* Python **3.11+** (3.12.7 recommended)
* `ffmpeg` (for video generation)
* [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Install

```shell
https://github.com/GeneralUserModels/pack.git  # Clone repo
cd pack
uv sync --python 3.12.7  # Install dependencies
cp .env.example .env  #  Optionally add your Gemini API key here
```

---

# Usage

Two main entry points:

* **Record a session** — capture screenshots + input events
  `uv run -m record`

* **Label / process a session** — aggregate events, run VLM labeling, optionally create annotated videos
  `uv run -m label`

Below are the flags for each command (concise, non-code descriptions).

---

## `uv run -m record` — Record a session

**What it does:** records screen activity and user input events into a session folder.

**Flags**

* `-f, --fps` — frames per second to capture. **Type:** int — **Default:** `16`
* `-s, --buffer-seconds` — seconds to keep in the in-memory cyclic buffer. **Type:** int — **Default:** `12`
* `-b, --buffer-all-images` — save all buffered images to disk (flag). **Default:** off
* `-m, --monitor` — enable real-time monitoring of the last session (flag). **Default:** off
* `-r, --max-res` — maximal resolution for screenshots: provide `width height`. **Default:** no scaling

**Record output (per session folder, saved under `logs/`)**

* `screenshots/` — captured frames
* `buffer_imgs/` — exported buffer images (if `--buffer-all-images`)
* `aggregations.jsonl` — aggregated event bursts
* `events.jsonl` — all raw input events
* `screenshots.jsonl` — screenshot metadata

---

## `uv run -m label` — Label a recorded session (VLM)

**What it does:** loads recorded sessions (or raw video), chunks and formats them, runs VLM labeling to produce captions, and optionally renders annotated videos.

### Session selection (mutually exclusive)

* `--session <PATH>` — single session folder.
* `--sessions-root <PATH>` — process all session folders under this root.

### Session processing options

* `--agg-jsonl` — aggregation filename to read/write. **Default:** `aggregations.jsonl`
* `--chunk-duration` — chunk duration in seconds for processing. **Type:** int — **Default:** `60`
* `--fps` — FPS for sampling frames during labeling. **Type:** int — **Default:** `1`

### Video-only / video handling

* `--video-only` — process video files without input logs (flag).
* `--video-only-prompt` — prompt file for video-only mode. **Default:** `prompts/video_only_prompt.txt`
* `--video-extensions` — recognized video extensions. **Default:** `[".mp4", ".avi", ".mov", ".mkv"]`
* `--label-video` — annotate video frames (flag).
* `--skip-existing` — skip sessions that already have `matched_captions.jsonl` (flag).

### Visualization

* `--visualize-session` — produce an annotated video with captions and events (`annotated_session.mp4`).

### VLM client & model selection

* `--client` — VLM backend: `gemini` or `qwen3vl`. **Default:** `gemini`
* `--model-id` — model path or ID to use.

### Parallelization / performance

* `--num-workers` — concurrent workers (Gemini). **Type:** int — **Default:** `4`
* `--batch-size` — batch size for Qwen3VL inference. **Type:** int — **Default:** `8`

### Qwen3VL server options (when using `qwen3vl`)

* `--vllm-url` — use an existing vLLM server (e.g. `http://localhost:8000`) instead of launching one.
* `--vllm-port` — port when starting a new vLLM server. **Default:** `8000`
* `--tensor-parallel-size` — tensor parallelism (GPUs/shards). **Default:** `1`
* `--gpu-memory-utilization` — fraction of GPU memory to reserve for the server. **Default:** `0.9`
* `--max-model-len` — max token length for the model.
* `--server-startup-timeout` — seconds to wait for server startup. **Default:** `600`
* `--enable-expert-parallel` — enable MoE expert-parallel mode (flag).

---

## Examples

Gemini + annotated video:

```bash
uv run -m label \
  --session logs/session_xyz \
  --client gemini \
  --model-id gemini-2.5-pro \
  --label-video \
  --visualize-session
```

Process all sessions in a folder:

```bash
uv run -m label \
  --sessions-root logs/ \
  --client gemini \
  --label-video
```

Video-only labeling with Qwen3VL on 4 GPUs:

```bash
uv run -m label \
  --session path_with_video \
  --label-video \
  --client qwen3vl \
  --model-id Qwen/Qwen3-VL-8B-Thinking-FP8 \
  --video-only \
  --tensor-parallel-size 4
```

Launch a Qwen vLLM instance automatically (example):

```bash
uv run -m label \
  --session logs/session_xyz \
  --client qwen3vl \
  --model-id Qwen/Qwen3-VL-8B-Thinking-FP8 \
  --label-video \
  --tensor-parallel-size 4
```

Use an existing vLLM server:

```bash
uv run -m label \
  --session logs/session_xyz \
  --label-video \
  --client qwen3vl \
  --vllm-url http://127.0.0.1:8000
```

---

## Label output (per processed session)

* Screenshot video and chunks
* `captions.jsonl` and caption chunk files — VLM-generated action captions with relative timestamps
* `matched_captions.jsonl` — captions matched to screenshots with recorded timestamps and input events
* `annotated_session.mp4` — annotated video (when `--visualize-session` is used)

---

# Method

## Record

The `record` module captures screenshots and user input events (mouse_move, mouse_scroll, mouse_up, mouse_down, key_press, key_release) and organizes them into per-category buffers and a global chronological buffer.

Key behavior:

* Screenshots are captured at `1 / fps` seconds (default `fps=16` → ~0.0625 s between frames). The recorder maintains a cyclic buffer retaining the last `buffer_seconds` (default 12s → 192 frames at 16 fps). The `--buffer-all-images` (`-b`) flag exports all buffer frames to disk (off by default).
* Input events are categorized into **click**, **move**, **scroll**, and **key** buffers while also appended to a shared chronological buffer (used later for alignment).

Burst detection and aggregation:

1. New events are appended to the category buffer and the shared buffer.
2. If a new event occurs within `gap_threshold` seconds of the previous event in that category, it is considered part of the current burst. If the category buffer reaches `total_threshold` events, it is split: the first half is sent to aggregation and the remainder stays buffered. Otherwise the event is appended.
3. If the new event is outside `gap_threshold`, the existing burst is closed (aggregated) and the new event starts a fresh burst.
4. A background worker runs every second to close bursts whose most recent event is older than `gap_threshold`, ensuring no events are lost when screenshots roll out of the cyclic buffer.
5. If the monitor of the cursor position changed compared to the last event, a new burst is started automatically.

Aggregation flow (high level):

1. Aggregation requests are queued for the screenshots immediately before and after a burst (the recorder picks screenshots at ±75 ms around the burst edges).
2. A worker ensures no intervening requests could alter the burst end time (bounded by `total_threshold`); when safe, the following request’s start time can be used to set the current burst end.
3. All events between burst start and end are pulled from the shared buffer and saved alongside the before/after screenshots into `aggregations.jsonl`.
   All disk writes are performed asynchronously so the recorder loop stays responsive.

## Label

The `label` module:

* Loads sessions or raw video, chunks them and their logs, and prepares inputs for the VLM.
* Uses prompts (in `label/prompts`) to instruct the VLM to generate captions that describe the user’s actions and context.
* Produces `captions.jsonl` and `matched_captions.jsonl` (captions aligned to screenshots and events).
* Optionally renders an annotated video (`annotated_session.mp4`) showing captions and event visualizations overlayed on frames.

The label step performs a second layer of aggregation: it uses the bursts detected at recording time and further refines and annotates them with VLM outputs to create final human-readable summaries.
