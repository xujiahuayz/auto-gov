# plf-env

RL environment for [Protocols for Loanable Funds](https://arxiv.org/abs/2006.13922)

## Setup

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below

```
git clone https://github.com/xujiahuayz/auto-gov.git
cd auto-gov
pip install -e ".[dev]"
```

## Running the simulation

<!-- TODO -->

## Git Large File Storage (Git LFS)

All files in [`data/`](data/) are stored with `lfs`:

```
git lfs track data/**/*
```

## Test the code

```zsh
pytest
```
