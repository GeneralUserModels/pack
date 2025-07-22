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
```
