# auto.gov

[![python](https://img.shields.io/badge/Python-v3.11.3-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## Setup

```
git clone https://github.com/xujiahuayz/auto-gov.git
cd auto-gov
```

## Installation

```
pip install toml
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- iOS

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install the project in editable mode

```bash
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
