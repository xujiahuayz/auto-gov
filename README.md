# auto.gov

![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/xujiahuayz/auto-gov)

## Setup

```
git clone https://github.com/xujiahuayz/auto-gov.git
cd auto-gov
```

### Give execute permission to your script and then run `setup_repo.sh`

<!-- TODO: use non-deprecated setup -->

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
