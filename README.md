# auto.gov

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
