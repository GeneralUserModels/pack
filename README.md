# Pack of GUM

## Setup

- prerequisite: `python 3.11+` (`python3.12.7` recommended), [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
# setup once
git clone git@github.com:GeneralUserModels/pack.git
uv sync --python 3.12.7
```

### Activate environment
```shell
source .venv/bin/activate
```

### run record
```shell
uv run -m record
```

## Methodology

### Recording

The `record` module captures user interactions on the computer in the form of aggregated input events (click, move, scroll, key) and corresponding screenshots. To achieve this, the system employs the following approach:

We record a screenshot every `1 / fps = 16` seconds (spawning a new process for each) and store it in a cyclic memory buffer that retains the last `12` seconds of activity (equivalent to 192 screenshots). A run flag `-b` can be set to export all screenshots to disk, but this is disabled (`false`) by default.

In parallel, we log user input events (`mouse_move`, `mouse_scroll`, `mouse_up`, `mouse_down`, `key_press`, `key_release`), grouped into four categories: click, move, scroll, and key. Each category maintains its own buffer for event aggregation, while an additional shared buffer holds **all** events in chronological order.
Aggregations are triggered by the category-specific buffers, whereas the shared buffer is used to align events with the corresponding screenshots. The process works as follows:

1. When a new event is captured, it is added to both the corresponding category-specific buffer and the shared buffer.
2. The system then checks whether the new event occurred within `gap_threshold` seconds of the previous event in the buffer:

   * If **yes**, the event is considered part of the current burst. If the buffer reaches `total_threshold` events, it is split in half: the first half is passed to the aggregation callback, and the remaining events stay in the buffer. Otherwise, the event is simply appended.
   * If **no**, the aggregation callback is triggered with the existing burst, and the new event initializes a fresh buffer.
3. Every second, a background worker checks for bursts where the most recent event is older than `gap_threshold` seconds. In such cases, the burst is aggregated and the buffer is cleared. This ensures that events are not lost when the screenshot buffer no longer holds the corresponding images.

Triggering an aggregation involves the following steps:

1. The first and last event in the burst are used to locate the closest screenshots in the buffer. A padding of `±75 ms` ensures that the screenshots capture the state both before and after the event (`-75 ms` for before, `+75 ms` for after).
2. All events occurring between the burst’s first and last event are retrieved from the shared buffer and saved together with the screenshots in a JSONL file.

All saving operations are performed asynchronously to ensure that the main recording loop remains non-blocking and responsive.
