# Hybrid synthetic event log generation

## Setup

Create and activate a Python 3.10 virtual environment (example):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies from setup.py:

```bash
pip install -e .
```

Set the CSV column names in `log_to_synth.py` (`CASEID_COL`, `ACTIVITY_COL`, `TIMESTAMP_COL`) to match your input file.

## Running

Save the event log in the "event_log" folder.

Run the generator:

```bash
python log_to_synth.py
```

The event log is not public due to privacy constraints; contact the corresponding author for access (roberto.nai@unito.it)