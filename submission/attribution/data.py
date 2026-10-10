"""Data checks, leakage-safe sampling and text encoding for the midterm project.

This module never downloads or modifies the supplied CSV files.  All fitted text
objects (the vocabulary) use training rows only.  Output-only and input+output
conditions use exactly the same output token budget.
"""
from __future__ import annotations

from array import array
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<sep>": 2, "<nl>": 3, "<identity>": 4}
EXPECTED_LABELS = {0: "GPT", 1: "Claude", 2: "Gemini", 3: "Qwen", 4: "DeepSeek"}
REQUIRED_COLUMNS = {
    "record_id", "prompt_id", "group_id", "LLM_name", "model_id", "label",
    "LLM_Input", "LLM_output", "task_domain", "benchmark", "split",
}
# Unicode letter/digit sequences, underscores, punctuation and explicit newlines.
# Special markers are recognized before the punctuation alternative.
TOKEN_RE = re.compile(r"<identity>|<nl>|\r\n|\r|\n|[^\W_]+|_+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\S+")
IDENTITY_RE = re.compile(
    r"(?<!\w)(?:(?:openai|anthropic|google|alibaba|qwen|deepseek)[/ :]+)?"
    r"(?:chatgpt|gpt|claude|gemini|qwen|deepseek)"
    r"(?:-?\d[\w.]*)?"
    r"(?:[-_/ ](?:chat|sonnet|opus|haiku|flash|pro|turbo|instruct|reasoner|"
    r"v?\d[\w.]*|a\d[\w.]*))*"
    r"(?!\w)|(?<!\w)(?:openai|anthropic|google|alibaba)(?!\w)",
    re.IGNORECASE,
)
MODES = {"input_only", "output_only", "input_and_output"}
TRANSFORMS = {"none", "no_format", "mask_identity", "fixed_length"}


def normalized_input(text: str) -> str:
    """Conservative exact-duplicate normalization (not semantic similarity)."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read prepared/{train,validation,test}.csv with text preserved verbatim.

    Pass either the prepared directory itself or its parent project directory.
    A literal answer such as 'NA' is retained, not treated as a missing value.
    """
    folder = Path(data_dir).expanduser()
    if not (folder / "train.csv").is_file() and (folder / "prepared" / "train.csv").is_file():
        folder = folder / "prepared"
    splits: dict[str, pd.DataFrame] = {}
    text_columns = {column: str for column in REQUIRED_COLUMNS if column != "label"}
    for name in ("train", "validation", "test"):
        path = folder / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}. Keep all three CSV files in prepared/.")
        frame = pd.read_csv(path, dtype=text_columns, keep_default_na=False, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        labels = pd.to_numeric(frame["label"], errors="raise")
        if not np.isfinite(labels.to_numpy(dtype=float)).all() or (labels % 1 != 0).any():
            raise ValueError(f"{path.name}: label must contain finite integers.")
        frame["label"] = labels.astype("int64")
        splits[name] = frame
    return splits


def audit_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, Any]:
    """Check labels, complete matched prompts and leakage before any fitting.

    Known near-duplicate groups must stay in one split.  The additional text
    check detects normalized exact duplicates; it does NOT claim to detect all
    semantic paraphrases.  Failures are saved before an exception is raised.
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    counts: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    label_names: dict[int, set[str]] = {label: set() for label in EXPECTED_LABELS}
    label_models: dict[int, set[str]] = {label: set() for label in EXPECTED_LABELS}
    seen: dict[str, dict[str, str]] = {key: {} for key in ("record_id", "prompt_id", "group_id", "normalized_input")}
    for split in ("train", "validation", "test"):
        if split not in splits:
            errors.append(f"Missing split: {split}")
            continue
        frame = splits[split]
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            errors.append(f"{split}: missing columns {sorted(missing)}")
            continue
        if frame.empty:
            errors.append(f"{split}: no rows")
            continue
        for column in REQUIRED_COLUMNS - {"label"}:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                errors.append(f"{split}: empty values in {column}")
        if frame["record_id"].duplicated().any():
            errors.append(f"{split}: duplicate record_id values")
        labels = set(frame["label"].tolist())
        if labels != set(EXPECTED_LABELS):
            errors.append(f"{split}: expected labels 0..4, found {sorted(map(str, labels))}")
        actual_splits = set(frame["split"].astype(str).str.lower().replace({"val": "validation", "dev": "validation"}))
        if actual_splits != {split}:
            errors.append(f"{split}: split column contains {sorted(actual_splits)}")
        pairs = frame.groupby("prompt_id", sort=False).agg(
            n=("label", "size"), n_labels=("label", "nunique"),
            n_inputs=("LLM_Input", "nunique"), n_groups=("group_id", "nunique"),
            n_domains=("task_domain", "nunique"), n_benchmarks=("benchmark", "nunique"),
        )
        bad = pairs.loc[(pairs["n"] != 5) | (pairs["n_labels"] != 5) |
                        (pairs["n_inputs"] != 1) | (pairs["n_groups"] != 1) |
                        (pairs["n_domains"] != 1) | (pairs["n_benchmarks"] != 1)]
        if not bad.empty:
            errors.append(f"{split}: {len(bad)} prompts lack five matched, distinct labels and identical input/domain/benchmark/group")
        per_class = frame.groupby("label").size()
        if per_class.nunique() != 1:
            errors.append(f"{split}: class counts are not equal")
        for label, part in frame.groupby("label", sort=True):
            if label in label_names:
                label_names[label].update(part["LLM_name"].astype(str).unique())
                label_models[label].update(part["model_id"].astype(str).unique())
        for (label, name, domain), count in frame.groupby(["label", "LLM_name", "task_domain"]).size().items():
            counts.append({"split": split, "label": int(label), "LLM_name": name, "task_domain": domain, "n_rows": int(count)})
        for key in ("record_id", "prompt_id", "group_id"):
            for value in frame[key].astype(str).unique():
                previous = seen[key].get(value)
                if previous is not None and previous != split:
                    errors.append(f"Cross-split {key} overlap: {previous} and {split} (example {value[:80]})")
                    break
            seen[key].update({value: split for value in frame[key].astype(str).unique()})
        prompt_texts = frame.drop_duplicates("prompt_id")["LLM_Input"]
        digests = {sha256(normalized_input(str(value)).encode("utf-8")).hexdigest() for value in prompt_texts}
        overlaps = {seen["normalized_input"][value] for value in digests if value in seen["normalized_input"]}
        if overlaps:
            errors.append(f"Cross-split normalized exact input overlap: {split} and {sorted(overlaps)}")
        seen["normalized_input"].update({value: split for value in digests})
        summary[split] = {"rows": int(len(frame)), "prompts": int(frame["prompt_id"].nunique()),
                          "groups": int(frame["group_id"].nunique()), "per_label": {str(k): int(v) for k, v in per_class.items()}}
    for label, expected_family in EXPECTED_LABELS.items():
        names = label_names[label]
        models = label_models[label]
        if len(names) != 1 or len(models) != 1:
            errors.append(f"Label {label}: expected one consistent LLM_name and model_id, found names={sorted(names)}, model_ids={sorted(models)}")
        elif expected_family.casefold() not in next(iter(names)).casefold():
            errors.append(f"Label {label}: expected {expected_family} family, found {sorted(names)}")
    used_names = [next(iter(value)) for value in label_names.values() if len(value) == 1]
    used_models = [next(iter(value)) for value in label_models.values() if len(value) == 1]
    if len(used_names) != len(set(used_names)) or len(used_models) != len(set(used_models)):
        errors.append("Multiple labels map to the same LLM_name or model_id")
    audit = {"passed": not errors, "splits": summary,
             "label_mapping": {str(k): {"LLM_name": sorted(label_names[k]), "model_id": sorted(label_models[k])} for k in EXPECTED_LABELS},
             "errors": errors,
             "near_duplicate_scope": "Checks the supplied group_id assignments and normalized exact input equality; does not newly detect every semantic paraphrase."}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(counts).to_csv(output / "dataset_counts.csv", index=False, encoding="utf-8-sig")
    if errors:
        raise ValueError("Data audit failed. See data_audit.json:\n" + "\n".join(errors))
    return audit


def sample_groups(frame: pd.DataFrame, max_groups: int | None, seed: int) -> pd.DataFrame:
    """Sample approximately by task, keeping every row in each chosen group_id.

    max_groups counts connected groups, not rows and not individual prompts.
    A connected group can contain several prompts, so its row count can exceed 5.
    """
    if max_groups is None or int(max_groups) <= 0:
        return frame.copy().reset_index(drop=True)
    unique = frame[["group_id", "task_domain"]].drop_duplicates().copy()
    # A rare group spanning domains receives a stable combined-domain stratum.
    grouped = unique.groupby("group_id", sort=True)["task_domain"].agg(lambda values: " | ".join(sorted(set(map(str, values)))))
    target = min(int(max_groups), len(grouped))
    if target == len(grouped):
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    strata = [(name, sorted(grouped.index[grouped == name].tolist())) for name in sorted(grouped.unique())]
    sizes = np.array([len(values) for _, values in strata], dtype=int)
    # Preserve every stratum when the requested sample is large enough.
    allocations = np.ones(len(strata), dtype=int) if target >= len(strata) else np.zeros(len(strata), dtype=int)
    capacity = sizes - allocations
    remaining = target - int(allocations.sum())
    if remaining:
        ideal = remaining * capacity / capacity.sum()
        floor = np.floor(ideal).astype(int)
        allocations += floor
        remainder = remaining - int(floor.sum())
        order = np.argsort(-(ideal - floor), kind="stable")
        allocations[order[:remainder]] += 1
    selected: set[str] = set()
    for (_, group_ids), count in zip(strata, allocations):
        if count:
            positions = rng.choice(len(group_ids), size=int(count), replace=False)
            selected.update(group_ids[int(position)] for position in positions)
    return frame.loc[frame["group_id"].isin(selected)].copy().reset_index(drop=True)


def tokenize(text: str, limit: int | None = None) -> list[str]:
    """Lowercase Unicode tokens; stop scanning as soon as limit is reached."""
    if limit is not None and limit <= 0:
        return []
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        tokens.append("<nl>" if token in ("\r\n", "\r", "\n") else token.lower())
        if limit is not None and len(tokens) >= limit:
            break
    return tokens


def _original_span(text: str, token_limit: int) -> str:
    """Select the original prefix BEFORE any transform removes/replaces tokens."""
    if token_limit <= 0:
        return ""
    end = 0
    for count, match in enumerate(TOKEN_RE.finditer(text), start=1):
        end = match.end()
        if count == token_limit:
            return text[:end]
    return text[:end]


def first_words(text: str, count: int) -> str:
    """Original text prefix ending at its count-th whitespace-delimited word."""
    if count <= 0:
        return ""
    for index, match in enumerate(WORD_RE.finditer(text), start=1):
        if index == count:
            return text[:match.end()]
    return text


def remove_format(text: str) -> str:
    """Approximate Markdown/newline removal, preserving ordinary text/code.

    This is a rule-based formatting ablation, not a lossless Markdown parser.
    Some syntax or linguistic information may also be removed; report that limit.
    """
    text = re.sub(r"(?m)^\s*(?:`{3,}|~{3,})[^\r\n]*$", " ", text)
    text = re.sub(r"(?m)^\s{0,3}(?:[-*_]\s*){3,}$", " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:>\s*)+", "", text)
    text = re.sub(r"!?\[([^\]\n]+)\]\([^\)\n]*\)", r"\1", text)
    text = re.sub(r"(`+)(.*?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"(\*\*|__|~~)(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)(\*|_)(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
    return " ".join(text.split())


def _budgets(cfg: dict[str, Any]) -> tuple[int, int]:
    prompt_budget = int(cfg.get("input_max_tokens", 256))
    output_budget = int(cfg.get("output_max_tokens", 512))
    if prompt_budget < 1 or output_budget < 1:
        raise ValueError("input_max_tokens and output_max_tokens must be positive")
    return prompt_budget, output_budget


def output_tokens(text: str, cfg: dict[str, Any], transform: str = "none", length_k: int | None = None) -> list[str]:
    """Transform only the same original answer prefix used by the baseline."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; choose {sorted(TRANSFORMS)}")
    _, budget = _budgets(cfg)
    if transform == "none":
        return tokenize(text, budget)
    if transform == "fixed_length":
        if length_k is None or int(length_k) <= 0:
            raise ValueError("fixed_length requires a positive length_k")
        if int(length_k) > budget:
            raise ValueError("length_k cannot exceed output_max_tokens")
        # The caller retains complete five-response groups with every answer >= K
        # actual tokenizer tokens. This controls the length the network observes.
        tokens = tokenize(text, int(length_k))
        if len(tokens) != int(length_k):
            raise ValueError("fixed_length received an answer shorter than K tokenizer tokens; filter whole prompt groups before encoding")
        return tokens
    span = _original_span(text, budget)
    if transform == "no_format":
        span = remove_format(span)
    elif transform == "mask_identity":
        span = IDENTITY_RE.sub("<identity>", span)
    return tokenize(span, budget)


def build_vocab(train_df: pd.DataFrame, cfg: dict[str, Any], transform: str = "none", length_k: int | None = None) -> dict[str, int]:
    """Fit a stable vocabulary using only budgeted TRAINING text.

    Count each question once per prompt_id rather than repeating it five times.
    The same vocabulary can be shared by all three input conditions within a run.
    """
    prompt_budget, _ = _budgets(cfg)
    vocabulary_size = int(cfg.get("vocabulary_size", cfg.get("vocab_size", 20000)))
    if vocabulary_size < len(SPECIAL_TOKENS):
        raise ValueError(f"vocabulary_size must be at least {len(SPECIAL_TOKENS)}")
    minimum = int(cfg.get("min_token_frequency", 1))
    counts: Counter[str] = Counter()
    for text in train_df.drop_duplicates("prompt_id")["LLM_Input"]:
        counts.update(tokenize(str(text), prompt_budget))
    for text in train_df["LLM_output"]:
        counts.update(output_tokens(str(text), cfg, transform, length_k))
    vocab = dict(SPECIAL_TOKENS)
    ranked = sorted(((token, count) for token, count in counts.items() if token not in vocab and count >= minimum), key=lambda item: (-item[1], item[0]))
    for token, _ in ranked[:vocabulary_size - len(vocab)]:
        vocab[token] = len(vocab)
    return vocab


class EncodedDataset(Dataset):
    """Encode once to compact uint32 arrays; return row IDs for prediction export."""

    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int], mode: str,
                 cfg: dict[str, Any], transform: str = "none", length_k: int | None = None):
        if mode not in MODES:
            raise ValueError(f"Unknown input mode {mode!r}; choose {sorted(MODES)}")
        if transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform {transform!r}")
        self.frame = frame.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)
        self.encoded: list[array] = []
        prompt_budget, _ = _budgets(cfg)
        unknown = int(vocab.get("<unk>", 1))
        prompt_cache: dict[str, list[int]] = {}
        for prompt, answer in self.frame[["LLM_Input", "LLM_output"]].itertuples(index=False, name=None):
            ids: list[int] = []
            if mode != "output_only":
                if prompt not in prompt_cache:
                    prompt_cache[prompt] = [vocab.get(token, unknown) for token in tokenize(prompt, prompt_budget)]
                ids.extend(prompt_cache[prompt])
            if mode == "input_and_output":
                ids.append(int(vocab["<sep>"]))
            if mode != "input_only":
                ids.extend(vocab.get(token, unknown) for token in output_tokens(answer, cfg, transform, length_k))
            # Empty transformed responses remain valid examples, represented UNK.
            self.encoded.append(array("I", ids or [unknown]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int, int]:
        return list(self.encoded[index]), int(self.labels[index]), index


def pad_batch(batch: Iterable[tuple[list[int], int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic right-padding: input_ids, lengths, labels, original row indices."""
    items = list(batch)
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(item[0]) for item in items], dtype=torch.long)
    padded = torch.zeros((len(items), int(lengths.max())), dtype=torch.long)
    for row, (ids, _, _) in enumerate(items):
        padded[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor([item[1] for item in items], dtype=torch.long)
    indices = torch.tensor([item[2] for item in items], dtype=torch.long)
    return padded, lengths, labels, indices


def output_word_counts(frame: pd.DataFrame) -> pd.Series:
    """Count raw whitespace words (recompute instead of trusting old metadata)."""
    return frame["LLM_output"].map(lambda text: len(str(text).split())).astype("int64")


def token_lengths(frame: pd.DataFrame, max_tokens: int = 512) -> np.ndarray:
    """Capped original-response token lengths for the fixed-length ablation.

    Select K using training groups only. Keep a prompt only if ALL its responses
    have length >= K, then apply that same fixed eligibility rule to other splits.
    """
    return np.fromiter((len(tokenize(str(text), max_tokens)) for text in frame["LLM_output"]), dtype=np.int64, count=len(frame))
