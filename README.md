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
git clone https://github.com/GeneralUserModels/pack.git  # Clone repo
cd pack
cp .env.example .env  #  Optionally add your Gemini API key here
```

## Usage

Two main entry points:

* **Record a session**
  ```shell
  uv run -m record --monitor  # start and monitor recording. Use --accessibility to capture accessibility info.
  ```
  ```shell
  CTRL+C  # stop recording
  ```
* **Label a session**
  ```shell
  uv run -m label \
    --sessions-root logs/ `# label all sessions in logs dir` \
    --skip-existing `# skip sessions already processed` \
    --client gemini \
    --model gemini-2.5-pro \
    --annotate `# visualize mouse movement and click positions` \
    --visualize `# final video creation`
  ```

---

# Detailed Commands


## `uv run -m record` — Record a session

**What it does:** Records screen activity and user input events into a session folder.

### Flags

| Flag                             | Type                 | Default    | Description                                     |
| -------------------------------- | -------------------- | ---------- | ----------------------------------------------- |
| `-f, --fps`                      | int                  | `30`       | Frames per second to capture                    |
| `-s, --buffer-seconds`           | int                  | `12`       | Seconds to keep in buffer                       |
| `-b, --buffer-all-images`        | flag                 | off        | Save all buffer images to disk                  |
| `-m, --monitor`                  | flag                 | off        | Enable real-time monitoring of the last session |
| `-r, --max-res <width> <height>` | int int              | none       | Maximal resolution for screenshots              |
| `-p, --precision`                | `accurate` / `rough` | `accurate` | Precision level for event aggregation (presets) |
| `-c, --compression-quality`      | int                  | 70         | JPEG compression quality                        |

### Output

```shell
logs/session_name
├── aggregations.jsonl
├── events.jsonl
├── screenshots
│   └── 1760971355.978042_reason_key_start.jpg
├── screenshots.jsonl
└── summary.png
```


---

## `uv run -m label` — Label a session

**What it does:** Loads recorded sessions or raw screenshots, chunks and formats them, runs VLM labeling, and optionally renders annotated videos.

### Session selection (required)

* `--session <PATH>` — single session folder
* `--sessions-root <PATH>` — process all sessions under this root

### Processing options

| Flag               | Type | Default | Description                     |
| ------------------ | ---- | ------- | ------------------------------- |
| `--chunk-duration` | int  | `60`    | Chunk duration in seconds       |
| `--fps`            | int  | `1`     | Frame sampling rate             |
| `--skip-existing`  | flag | off     | Skip already processed sessions |

### Screenshots-only mode

| Flag                               | Description                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--screenshots-only`               | Process raw screenshots without aggregations or input event annotations                                       |
| `--video-extensions .mp4 .avi ...` | Recognized video extensions                                                                                   |
| `--prompt-file`                    | Custom prompt file (defaults to `prompts/screenshots_only.txt` in screenshots-only mode or `prompts/default.txt`) |

> [!NOTE]
> Screenshots-only mode requires your session folder to contain a `screenshots/` subdirectory with image files (.jpg, .jpeg, or .png).

### Visualization & annotations

| Flag          | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `--annotate`  | Overlay cursor and click markers (only for standard processing) |
| `--visualize` | Create annotated video visualizations                           |

### VLM client

| Flag                     | Default                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| `--client gemini / vllm` | `gemini`                                                            |
| `--model`                | auto-selects: `gemini-2.5-flash` or `Qwen/Qwen3-VL-8B-Thinking-FP8` |
| `--num-workers`          | `4`                                                                 |
| `--vllm-url`             | none                                                                |

### Output

```shell
logs/session_name
├── aggregations
│   └── 000.json  # chunked aggregations used for LLM prompting
├── annotated.mp4  # final video showing captions and input events
├── captions
│   └── 000.json  # generated captions of chunk
├── captions.jsonl  # summarized generated captions
├── chunks
│   ├── 000.mp4  # video chunk used for LLM prompting
│   ├── master.mp4
│   └── prompt_000.txt  # prompt used for LLM prompting
└── data.jsonl  # final data containing raw input events and LLM generated captions
```

---

## Examples

Gemini + annotated video:

```bash
uv run -m label \
  --session logs/session_xyz \
  --client gemini \
  --model gemini-2.5-flash \
  --annotate \
  --visualize
```

Process all sessions in a folder:

```bash
uv run -m label \
  --sessions-root logs/ \
  --client gemini \
  --annotate
```

Screenshots-only labeling (with vLLM):

```bash
uv run -m label \
  --session path_with_screenshots \
  --screenshots-only \
  --client vllm \
  --model Qwen/Qwen3-VL-8B-Thinking-FP8 \
  --vllm-url http://localhost:8000/v1
```

> [!NOTE]
> For deploying a vllm server, see the [vLLM documentation](https://vllm.ai/).
> E.g. you can run:
> ```shell
> vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 --host 127.0.0.1 --port 8000 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --guided-decoding-backend outlines --enable-expert-parallel --enforce-eager
>vllm serve Qwen/Qwen3-VL-8B-Thinking-FP8 --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --guided-decoding-backend outlines
>```

---

# Method

## Record

The `record` module captures screenshots and user input events (mouse_move, mouse_scroll, mouse_up, mouse_down, key_press, key_release) and organizes them into per-category buffers and a global chronological buffer.

Key behavior:

* Screenshots are captured at `1 / fps` seconds (default `fps=30` → ~0.0625 s between frames). The recorder maintains a cyclic buffer retaining the last `buffer_seconds` (default 12s → 360 frames at 320 fps). The `--buffer-all-images` (`-b`) flag exports all buffer frames to disk (off by default).
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

* Loads sessions or raw screenshots, chunks them and their logs, and prepares inputs for the VLM.
* Uses prompts (in `label/prompts`) to instruct the VLM to generate captions that describe the user's actions and context.
* Produces `captions.jsonl` and `data.jsonl` (captions aligned to screenshots and events).
* Optionally renders an annotated video (`annotated.mp4`) showing captions and event visualizations overlayed on frames.

The label step performs a second layer of aggregation: it uses the bursts detected at recording time and further refines and annotates them with VLM outputs to create final human-readable summaries.
