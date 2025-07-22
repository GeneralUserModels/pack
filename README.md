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

### run aggregation / evaluation
```shell
uv run -m pipeline
# for label aggregation, 1 to create aggregated logs and 3 to create a video
```

### label video in chunks (Gemini)
```shell
# adjust SESSION variable of the prompt/prompt.py to match your session directory 
uv run -m prompt.prompt
```
