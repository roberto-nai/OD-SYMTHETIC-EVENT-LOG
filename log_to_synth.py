"""
log_to_synth.py

Hybrid synthetic event log generation (LSTM + constraint mask) + evaluation.

User requirements implemented:
- Uses the *same column names* as the provided CSV (orbassano.csv).
- Main parameters are constants in UPPERCASE at the top of the file.
- Input folder is `event_log/`
- Output is written in the *same folder* with suffix `_synth.csv`
  (and an additional `_synth_eval.json` for metrics).

Baseline constraints (expandable):
- TRIAGE must be the first event of every synthetic trace
- DISCHARGE must be the last event (generation stops when DISCHARGE is sampled)

Timestamp synthesis:
- Generates ONLY the `timestamp` column for synthetic events, sampling inter-event deltas
  from the original log (optionally per transition activity_a -> activity_b).

Evaluation (one-shot overview):
- Constraint validity rates
- Activity distribution similarity (Jensen–Shannon distance)
- DFG edge Jaccard overlap
- Top-k variants Jaccard overlap
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONSTANTS (edit here)
# =========================================================
EVENT_LOG_DIR = Path("event_log")
INPUT_FILENAME = "orbassano.csv"

SEED = 42
DEVICE = "auto"          # "auto", "cpu", "cuda"

# Model / training
WINDOW = 12
EMB_DIM = 64
HID_DIM = 128
NUM_LAYERS = 1
EPOCHS = 6
BATCH_SIZE = 256
LR = 1e-3

# Generation
MAX_LEN = 25
TEMPERATURE = 1.0

# Constraints
FIRST_EVENT = "TRIAGE"
LAST_EVENT = "DISCHARGE"

# Evaluation
TOPK_VARIANTS = 20

# Timestamp synthesis
MAX_DELTA_SEC = 24 * 3600        # sanity cap for inter-event gaps
DEFAULT_DELTA_SEC = 5 * 60       # fallback gap if no empirical data
DEFAULT_START_TOD_SEC = 9 * 3600 # fallback start time-of-day (09:00:00)


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Metrics helpers
# =========================================================
def normalise(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {k: 0.0 for k in counter}
    return {k: v / total for k, v in counter.items()}


def js_distance(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_arr = np.array([q.get(k, 0.0) for k in keys], dtype=float)

    p_arr = np.clip(p_arr, eps, 1.0)
    q_arr = np.clip(q_arr, eps, 1.0)
    m = 0.5 * (p_arr + q_arr)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    js_div = 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)
    return float(math.sqrt(max(js_div, 0.0)))


def dfg_edges(traces: List[List[str]]) -> Counter:
    edges = Counter()
    for tr in traces:
        for a, b in zip(tr[:-1], tr[1:]):
            edges[(a, b)] += 1
    return edges


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def topk_variants(traces: List[List[str]], k: int = 20) -> List[Tuple[Tuple[str, ...], int]]:
    c = Counter(tuple(t) for t in traces)
    return c.most_common(k)


# =========================================================
# Data loading (column names based on orbassano.csv)
# =========================================================
def read_event_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"case", "activity", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)}. Found: {list(df.columns)}")

    df["case"] = df["case"].astype(str)
    df["activity"] = df["activity"].astype(str)

    # Sort by timestamp to reconstruct traces
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["case", "timestamp"])

    return df


def build_traces(df: pd.DataFrame) -> List[List[str]]:
    traces = []
    for _, g in df.groupby("case", sort=False):
        traces.append(g["activity"].tolist())
    return traces


# =========================================================
# Timestamp models (learn from original log)
# =========================================================
def build_timestamp_models(
    df: pd.DataFrame,
) -> tuple[Dict[tuple[str, str], np.ndarray], np.ndarray, np.ndarray]:
    """
    Learns empirical distributions from the real log:
    - delta seconds between consecutive event timestamps, per transition (a -> b)
    - fallback delta seconds distribution (all transitions)
    - distribution of start time-of-day (seconds from midnight) for first events
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values(["case", "timestamp"])

    trans_deltas: Dict[tuple[str, str], list[float]] = {}
    all_deltas: list[float] = []
    start_tod: list[float] = []

    for _, g in work.groupby("case", sort=False):
        g = g.reset_index(drop=True)

        ts0 = g.loc[0, "timestamp"]
        start_tod.append(ts0.hour * 3600 + ts0.minute * 60 + ts0.second)

        for i in range(len(g) - 1):
            a = str(g.loc[i, "activity"])
            b = str(g.loc[i + 1, "activity"])
            t1 = g.loc[i, "timestamp"]
            t2 = g.loc[i + 1, "timestamp"]

            delta = (t2 - t1).total_seconds()
            if 0 <= delta <= MAX_DELTA_SEC:
                trans_deltas.setdefault((a, b), []).append(float(delta))
                all_deltas.append(float(delta))

    trans_arr = {k: np.array(v, dtype=float) for k, v in trans_deltas.items() if len(v) > 0}
    all_arr = np.array(all_deltas, dtype=float) if len(all_deltas) > 0 else np.array([DEFAULT_DELTA_SEC], dtype=float)
    start_tod_arr = np.array(start_tod, dtype=float) if len(start_tod) > 0 else np.array([DEFAULT_START_TOD_SEC], dtype=float)

    return trans_arr, all_arr, start_tod_arr


def sample_start_time(start_tod_arr: np.ndarray) -> float:
    if start_tod_arr is None or len(start_tod_arr) == 0:
        return DEFAULT_START_TOD_SEC
    x = float(np.random.choice(start_tod_arr))
    if 0 <= x < 24 * 3600:
        return x
    return DEFAULT_START_TOD_SEC


def sample_delta(
    trans_arr: Dict[tuple[str, str], np.ndarray],
    all_arr: np.ndarray,
    a: str,
    b: str,
) -> float:
    arr = trans_arr.get((a, b))
    if arr is not None and len(arr) > 0:
        x = float(np.random.choice(arr))
        if 0 <= x <= MAX_DELTA_SEC:
            return x

    if all_arr is not None and len(all_arr) > 0:
        x = float(np.random.choice(all_arr))
        if 0 <= x <= MAX_DELTA_SEC:
            return x

    return DEFAULT_DELTA_SEC


def assign_timestamps_to_trace(
    trace: List[str],
    trans_arr: Dict[tuple[str, str], np.ndarray],
    all_arr: np.ndarray,
    start_tod_arr: np.ndarray,
    anchor_date: pd.Timestamp,
) -> List[pd.Timestamp]:
    """
    Returns one timestamp per event in the trace.
    First timestamp uses a sampled real start time-of-day;
    next timestamps add sampled deltas (per transition if available).
    """
    if not trace:
        return []

    t = anchor_date.normalize() + pd.to_timedelta(sample_start_time(start_tod_arr), unit="s")
    ts_list: List[pd.Timestamp] = [t]

    for i in range(len(trace) - 1):
        a = trace[i]
        b = trace[i + 1]
        t = t + pd.to_timedelta(sample_delta(trans_arr, all_arr, a, b), unit="s")
        ts_list.append(t)

    return ts_list


# =========================================================
# Vocabulary
# =========================================================
@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_id: int
    unk_id: int

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]


def make_vocab(traces: List[List[str]]) -> Vocab:
    special = ["<PAD>", "<UNK>", "<START>"]
    activities = sorted(set(a for t in traces for a in t))
    all_tokens = special + activities
    stoi = {t: i for i, t in enumerate(all_tokens)}
    itos = {i: t for t, i in stoi.items()}
    return Vocab(stoi=stoi, itos=itos, pad_id=stoi["<PAD>"], unk_id=stoi["<UNK>"])


# =========================================================
# Dataset: next-event prediction
# =========================================================
class NextEventDataset(Dataset):
    """
    Builds (prefix -> next_event) samples from traces.
    Uses a fixed window length for prefixes; padding on the left.
    """

    def __init__(self, traces: List[List[str]], vocab: Vocab, window: int):
        self.vocab = vocab
        self.window = window
        self.samples: List[Tuple[List[int], int]] = []

        start = "<START>"
        for tr in traces:
            tr2 = [start] + tr
            ids = vocab.encode(tr2)
            for t in range(len(ids) - 1):
                prefix = ids[: t + 1]
                nxt = ids[t + 1]
                self.samples.append((prefix, nxt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix, nxt = self.samples[idx]
        if len(prefix) > self.window:
            prefix = prefix[-self.window:]
        pad_len = self.window - len(prefix)
        x = [self.vocab.pad_id] * pad_len + prefix
        return torch.tensor(x, dtype=torch.long), torch.tensor(nxt, dtype=torch.long)


# =========================================================
# Model
# =========================================================
class LSTMNextEvent(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)             # [B, T, E]
        out, _ = self.lstm(e)       # [B, T, H]
        last = out[:, -1, :]        # [B, H]
        return self.fc(last)        # [B, V]


# =========================================================
# Constraint mask (expandable)
# =========================================================
class ConstraintMask:
    """
    Expandable constraint masking.

    Baseline constraints:
      - TRIAGE is forced at step 0 (first generated event).
      - DISCHARGE is enforced as last event by stopping when sampled.
    """

    def __init__(self, vocab: Vocab, first_event: str, last_event: str):
        self.vocab = vocab

        if first_event not in vocab.stoi:
            raise ValueError(f"FIRST_EVENT '{first_event}' not found in vocabulary.")
        if last_event not in vocab.stoi:
            raise ValueError(f"LAST_EVENT '{last_event}' not found in vocabulary.")

        self.first_id = vocab.stoi[first_event]
        self.last_id = vocab.stoi[last_event]

        self.forbidden_always = {vocab.stoi["<PAD>"], vocab.stoi["<START>"]}

    def apply(self, logits: torch.Tensor, step: int, history: List[int]) -> torch.Tensor:
        masked = logits.clone()

        # Never generate special tokens.
        for tid in self.forbidden_always:
            masked[tid] = -1e9

        # Force TRIAGE as first event.
        if step == 0:
            masked[:] = -1e9
            masked[self.first_id] = 0.0
            return masked

        # (DISCHARGE as last event is handled by stopping when sampled.)
        return masked


# =========================================================
# Training
# =========================================================
def train_model(
    model: LSTMNextEvent,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(f"Epoch {ep}/{epochs} - loss: {total_loss / max(n, 1):.4f}")


# =========================================================
# Generation
# =========================================================
@torch.no_grad()
def generate_trace(
    model: LSTMNextEvent,
    vocab: Vocab,
    masker: ConstraintMask,
    window: int,
    max_len: int,
    temperature: float,
    device: str,
) -> List[str]:
    model.eval()

    start_id = vocab.stoi["<START>"]
    history: List[int] = []

    for step in range(max_len):
        prefix_ids = [start_id] + history
        if len(prefix_ids) > window:
            prefix_ids = prefix_ids[-window:]
        pad_len = window - len(prefix_ids)
        x = [vocab.pad_id] * pad_len + prefix_ids
        x_t = torch.tensor([x], dtype=torch.long, device=device)

        logits = model(x_t)[0] / max(temperature, 1e-6)
        logits = masker.apply(logits, step=step, history=history)

        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        history.append(next_id)

        # Enforce DISCHARGE as last event by stopping.
        if next_id == masker.last_id:
            break

    return vocab.decode(history)


@torch.no_grad()
def generate_log(
    model: LSTMNextEvent,
    vocab: Vocab,
    masker: ConstraintMask,
    window: int,
    n_traces: int,
    max_len: int,
    temperature: float,
    device: str,
) -> List[List[str]]:
    traces = []
    for _ in range(n_traces):
        traces.append(
            generate_trace(
                model=model,
                vocab=vocab,
                masker=masker,
                window=window,
                max_len=max_len,
                temperature=temperature,
                device=device,
            )
        )
    return traces


# =========================================================
# Save synthetic log (same schema as input + timestamp synthesis)
# =========================================================
def save_synth_log_like_input(
    input_df: pd.DataFrame,
    synth_traces: List[List[str]],
    out_csv: Path,
) -> None:
    """
    Saves a synthetic log with the SAME columns as the input.
    Populates:
      - case
      - activity
      - timestamp  (synthetic, sampled from original deltas)

    Leaves other fields as NaN (except resource='-' if present).
    """
    input_columns = list(input_df.columns)

    # Learn timestamp deltas from the original log
    trans_arr, all_arr, start_tod_arr = build_timestamp_models(input_df)

    # Anchor date: use DATE column if present, else use the first real timestamp date
    if "DATE" in input_df.columns and input_df["DATE"].notna().any():
        date_choices = pd.to_datetime(input_df["DATE"], errors="coerce").dropna().unique()
        anchor_date = pd.Timestamp(np.random.choice(date_choices))
    else:
        anchor_date = pd.to_datetime(input_df["timestamp"], errors="coerce").dropna().iloc[0].normalize()

    rows = []
    for i, tr in enumerate(synth_traces, start=1):
        case_id = f"SYN_{i:06d}"
        ts_list = assign_timestamps_to_trace(
            trace=tr,
            trans_arr=trans_arr,
            all_arr=all_arr,
            start_tod_arr=start_tod_arr,
            anchor_date=anchor_date,
        )

        for act, ts in zip(tr, ts_list):
            row = {c: np.nan for c in input_columns}
            row["case"] = case_id
            row["activity"] = act
            row["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")

            if "DATE" in row:
                row["DATE"] = ts.strftime("%Y-%m-%d")
            if "resource" in row:
                row["resource"] = "-"

            rows.append(row)

    out_df = pd.DataFrame(rows, columns=input_columns)

    # Keep the log ordered by (case, timestamp) like the real one
    if "timestamp" in out_df.columns:
        out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce")
        out_df = out_df.sort_values(["case", "timestamp"])
        out_df["timestamp"] = out_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_df.to_csv(out_csv, index=False)


# =========================================================
# Evaluation
# =========================================================
def evaluate(
    real_traces: List[List[str]],
    synth_traces: List[List[str]],
    first_event: str,
    last_event: str,
    topk: int,
) -> Dict:
    first_ok = sum(1 for t in synth_traces if t and t[0] == first_event)
    last_ok = sum(1 for t in synth_traces if t and t[-1] == last_event)
    both_ok = sum(1 for t in synth_traces if t and t[0] == first_event and t[-1] == last_event)

    real_act = Counter(a for t in real_traces for a in t)
    syn_act = Counter(a for t in synth_traces for a in t)
    jsd = js_distance(normalise(real_act), normalise(syn_act))

    real_dfg = dfg_edges(real_traces)
    syn_dfg = dfg_edges(synth_traces)
    dfg_jacc = jaccard(set(real_dfg.keys()), set(syn_dfg.keys()))

    real_top = topk_variants(real_traces, k=topk)
    syn_top = topk_variants(synth_traces, k=topk)
    var_jacc = jaccard(set(v for v, _ in real_top), set(v for v, _ in syn_top))

    return {
        "constraints": {
            "n_synthetic_traces": len(synth_traces),
            "first_event": first_event,
            "last_event": last_event,
            "first_event_ok_rate": first_ok / max(len(synth_traces), 1),
            "last_event_ok_rate": last_ok / max(len(synth_traces), 1),
            "both_ok_rate": both_ok / max(len(synth_traces), 1),
        },
        "process_fidelity": {
            "activity_js_distance": jsd,          # lower is better
            "dfg_edge_jaccard": dfg_jacc,         # higher is better
            "topk_variants_jaccard": var_jacc,    # higher is better
            "topk": topk,
        },
        "diagnostics": {
            "real_num_traces": len(real_traces),
            "real_num_events": sum(len(t) for t in real_traces),
            "syn_num_events": sum(len(t) for t in synth_traces),
            "real_num_activities": len(set(a for t in real_traces for a in t)),
            "syn_num_activities": len(set(a for t in synth_traces for a in t)),
        },
    }


# =========================================================
# Main
# =========================================================
def main() -> None:
    set_seed(SEED)

    input_path = (EVENT_LOG_DIR / INPUT_FILENAME).resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Put '{INPUT_FILENAME}' inside the folder '{EVENT_LOG_DIR}'."
        )

    # Output in the same folder with suffix _synth.csv
    out_csv = input_path.with_name(f"{input_path.stem}_synth{input_path.suffix}")
    out_eval = input_path.with_name(f"{input_path.stem}_synth_eval.json")

    # Device resolution
    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE

    print(f"Input:  {input_path}")
    print(f"Output: {out_csv}")
    print(f"Device: {device}")

    df = read_event_log(input_path)
    real_traces = build_traces(df)

    vocab = make_vocab(real_traces)
    dataset = NextEventDataset(real_traces, vocab=vocab, window=WINDOW)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMNextEvent(
        vocab_size=len(vocab.stoi),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_layers=NUM_LAYERS,
    )

    print(f"Training traces: {len(real_traces)} | Samples: {len(dataset)} | Vocab: {len(vocab.stoi)}")
    train_model(model, loader, device=device, epochs=EPOCHS, lr=LR)

    masker = ConstraintMask(vocab=vocab, first_event=FIRST_EVENT, last_event=LAST_EVENT)

    # Generate as many synthetic traces as real traces (easy baseline)
    synth_traces = generate_log(
        model=model,
        vocab=vocab,
        masker=masker,
        window=WINDOW,
        n_traces=len(real_traces),
        max_len=MAX_LEN,
        temperature=TEMPERATURE,
        device=device,
    )

    # Save synthetic log with the same schema as input, now including timestamp
    save_synth_log_like_input(df, synth_traces, out_csv)
    print(f"Saved synthetic log: {out_csv}")

    metrics = evaluate(
        real_traces=real_traces,
        synth_traces=synth_traces,
        first_event=FIRST_EVENT,
        last_event=LAST_EVENT,
        topk=TOPK_VARIANTS,
    )
    out_eval.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {out_eval}")

    print("\n=== Evaluation summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()