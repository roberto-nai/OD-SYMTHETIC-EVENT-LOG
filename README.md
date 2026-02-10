# Hybrid synthetic event log generation

## Setup

1) Create and activate a Python 3.10 virtual environment (example):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies from setup.py:

```bash
pip install -e .
```

3) Set the CSV column names in `log_to_synth.py` (`CASEID_COL`, `ACTIVITY_COL`, `TIMESTAMP_COL`) to match your input file.

## Running

1) Save the event log in the `event_log` folder.

2) Run the generator:

```bash
python log_to_synth.py
```

## Info
The event log is not public due to privacy constraints; contact the corresponding author for access (roberto.nai@unito.it)