# plf-env

RL environment for [Protocols for Loanable Funds](https://arxiv.org/abs/2006.13922)

## Setup

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
```

or follow the step-by-step instructions below

```
git clone https://github.com/danhper/plf-env.git
cd plf-env
pip install -e ".[dev]"
```

## Running the simulation

```
plf-env --debug process-events data/aave-v2-events/*.jsonl.gz
```

use `--include-agents` flag to simulate extra agents, and `-o` to save output in a pickle file

```
plf-env process-events data/aave-v2-events/*.jsonl.gz --include-agents -o data/agents.pkl
```

drop files to eliminate historical events

```
plf-env process-events --include-agents -o data/agents.pkl
```

## Git Large File Storage (Git LFS)

All files in [`data/`](data/) are stored with `lfs`:

```
git lfs track data/**/*
```

## Plotting the data

```
plf-env plot -s data/agents.pkl agent-wealths
```

`-o FILENAME.pdf` can be added to save the plot.

## Profiling the code

```
python -m cProfile -o program.prof scripts/run.py
snakeviz program.prof
```

## Test the code

```zsh
pytest
```
# auto-gov
