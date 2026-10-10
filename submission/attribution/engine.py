"""Train and evaluate TextCNN/BiLSTM without exposing test data to training.

Checkpoints in this project are local files produced by this program. Do not
replace them with checkpoint files downloaded from untrusted sources.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from .data import EncodedDataset, build_vocab, pad_batch


ENGINE_VERSION = "1.0.0"


class TextCNN(nn.Module):
    """Kim-style convolutional text classifier with padding-aware max pooling.

    For a sequence shorter than a kernel, one right-padded window is used.
    Additional windows that exist only because of batch padding are excluded.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_classes: int = 5, filters: int = 128,
                 kernels: tuple[int, ...] | list[int] = (3, 4, 5),
                 dropout: float = 0.3, padding_idx: int = 0) -> None:
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        if not self.kernels or min(self.kernels) < 1:
            raise ValueError("cnn_kernels must contain positive integers.")
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, kernel_size=k) for k in self.kernels
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * len(self.kernels), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] < max(self.kernels):
            input_ids = F.pad(input_ids, (0, max(self.kernels) - input_ids.shape[1]),
                              value=self.padding_idx)
        embeddings = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernels, self.convolutions):
            activations = F.relu(convolution(embeddings))
            valid_windows = (lengths.to(input_ids.device) - kernel + 1).clamp(min=1)
            positions = torch.arange(activations.shape[2], device=input_ids.device)
            valid = positions[None, :] < valid_windows[:, None]
            activations = activations.masked_fill(~valid[:, None, :], float("-inf"))
            pooled.append(activations.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """One-layer bidirectional LSTM; padding never enters the recurrent state."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_classes: int = 5, hidden_dim: int = 128,
                 dropout: float = 0.3, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.detach().cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


def seed_everything(seed: int, deterministic: bool = True) -> None:
    # This must be set before the first CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use device='cpu' or install "
                           "a PyTorch build compatible with your NVIDIA GPU.")
    return device


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _frame_hash(frame: pd.DataFrame) -> str:
    # Include actual text and row order, not just IDs, so a data edit cannot
    # silently reuse a completed result from an older experiment.
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("data.py", "engine.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _get_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_checkpoint(path: Path) -> Any:
    # The locally produced resume checkpoint includes Python/NumPy RNG states.
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | str,
             num_classes: int = 5) -> dict[str, Any]:
    """Deterministic evaluation; CE is weighted by samples, not by batches."""
    model.eval()
    all_probabilities, all_labels, all_indices = [], [], []
    summed_loss = 0.0
    with torch.inference_mode():
        for input_ids, lengths, labels, row_indices in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids, lengths)
            summed_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(row_indices.cpu().numpy())
    if not all_labels:
        raise ValueError("An evaluation split is empty.")
    probabilities = np.concatenate(all_probabilities)
    labels_array = np.concatenate(all_labels)
    indices = np.concatenate(all_indices)
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": summed_loss / len(labels_array),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "macro_f1": float(f1_score(labels_array, predictions, labels=list(range(num_classes)),
                                    average="macro", zero_division=0)),
        "probabilities": probabilities, "labels": labels_array,
        "predictions": predictions, "row_indices": indices,
    }


def _scores_from_confusion(matrix: np.ndarray) -> tuple[float, float]:
    true_positive = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    class_f1 = np.divide(2 * true_positive, denominator,
                         out=np.zeros_like(true_positive), where=denominator > 0)
    return float(true_positive.sum() / matrix.sum()), float(class_f1.mean())


def grouped_bootstrap(labels: np.ndarray, predictions: np.ndarray,
                      groups: np.ndarray | pd.Series, num_classes: int = 5,
                      repetitions: int = 200, seed: int = 42) -> dict[str, Any]:
    """Percentile CIs resampling whole leakage groups, including all model answers.

    These CIs describe test-sample uncertainty conditional on one fitted model.
    Variation across training seeds is reported separately by the report module.
    """
    if repetitions <= 0:
        return {"bootstrap_repetitions": 0}
    group_codes, unique_groups = pd.factorize(np.asarray(groups), sort=True)
    if (group_codes < 0).any():
        raise ValueError("Bootstrap group IDs contain missing values.")
    n_groups = len(unique_groups)
    if n_groups == 0:
        raise ValueError("Cannot bootstrap an empty test split.")
    flat = group_codes * num_classes ** 2 + labels * num_classes + predictions
    per_group = np.bincount(flat, minlength=n_groups * num_classes ** 2).reshape(
        n_groups, num_classes ** 2)
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, 2))
    for index in range(repetitions):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        matrix = (counts @ per_group).reshape(num_classes, num_classes)
        samples[index] = _scores_from_confusion(matrix)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "accuracy_ci_low": float(low[0]), "accuracy_ci_high": float(high[0]),
        "macro_f1_ci_low": float(low[1]), "macro_f1_ci_high": float(high[1]),
        "bootstrap_repetitions": repetitions, "bootstrap_groups": n_groups,
        "ci_method": "95% percentile bootstrap over group_id; conditional on fitted model",
    }


def _label_names(train_df: pd.DataFrame, num_classes: int) -> list[str]:
    if "label" not in train_df or "LLM_name" not in train_df:
        raise ValueError("Training data must contain integer 'label' and readable 'LLM_name'.")
    pairs = train_df[["label", "LLM_name"]].drop_duplicates()
    if set(pairs["label"].astype(int)) != set(range(num_classes)) or len(pairs) != num_classes:
        raise ValueError("Training labels must map one-to-one to LLM_name and cover 0..4.")
    return pairs.sort_values("label")["LLM_name"].astype(str).tolist()


def _write_evaluation(run_dir: Path, frame: pd.DataFrame, evaluated: dict[str, Any],
                       names: list[str]) -> None:
    index_order = np.argsort(evaluated["row_indices"])
    indices = evaluated["row_indices"][index_order]
    rows = frame.iloc[indices].reset_index(drop=True)
    labels = evaluated["labels"][index_order]
    predictions = evaluated["predictions"][index_order]
    probabilities = evaluated["probabilities"][index_order]
    metadata = [column for column in ("record_id", "prompt_id", "group_id", "task_domain", "benchmark")
                if column in rows]
    output = rows[metadata].copy()
    output["true_label_id"] = labels
    output["pred_label_id"] = predictions
    output["true_label"] = [names[int(value)] for value in labels]
    output["pred_label"] = [names[int(value)] for value in predictions]
    output["correct"] = labels == predictions
    for label in range(len(names)):
        output[f"prob_{label}"] = probabilities[:, label]
    output.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    report = classification_report(labels, predictions, labels=list(range(len(names))),
                                    target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(run_dir / "classification_report.csv", encoding="utf-8-sig",
                                 index_label="class")
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(names))))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        run_dir / "confusion_matrix.csv", encoding="utf-8-sig", index_label="true_label")
    for field, filename in (("task_domain", "metrics_by_domain.csv"),
                            ("benchmark", "metrics_by_benchmark.csv")):
        records = []
        if field in rows:
            for value, subset in rows.groupby(field, dropna=False, sort=True):
                selected = subset.index.to_numpy()
                records.append({field: str(value), "n": len(selected),
                    "accuracy": float(accuracy_score(labels[selected], predictions[selected])),
                    "macro_f1": float(f1_score(labels[selected], predictions[selected],
                        labels=list(range(len(names))), average="macro", zero_division=0))})
        pd.DataFrame(records, columns=[field, "n", "accuracy", "macro_f1"]).to_csv(
            run_dir / filename, index=False, encoding="utf-8-sig")


def run_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   cfg: dict[str, Any], run_dir: str | Path, architecture: str = "cnn",
                   mode: str = "output_only", seed: int = 42, transform: str = "none",
                   length_k: int | None = None, resume: bool = True) -> dict[str, Any]:
    """Fit one experiment, choose an epoch on validation F1, and test once.

    Resuming restores the last *completed* epoch, optimizer state, and RNG state.
    A partially finished epoch is replayed. A run directory is never silently
    reused after data, settings, code, or dependency versions have changed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if architecture not in {"cnn", "bilstm"}:
        raise ValueError("architecture must be 'cnn' or 'bilstm'.")
    if mode not in {"input_only", "output_only", "input_and_output"}:
        raise ValueError("mode must be input_only, output_only, or input_and_output.")
    for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if frame.empty:
            raise ValueError(f"{name} is empty: cannot run this experiment.")
    if int(cfg.get("epochs", 15)) < 1 or int(cfg.get("patience", 3)) < 1:
        raise ValueError("epochs and patience must be at least one.")
    num_classes = 5
    names = _label_names(train_df, num_classes)
    device = choose_device(str(cfg.get("device", "auto")))
    torch.set_num_threads(max(1, int(cfg.get("threads", 4))))
    seed_everything(seed, bool(cfg.get("deterministic", True)))
    signature_payload = {
        "engine_version": ENGINE_VERSION, "code_sha256": _code_hash(),
        "config": cfg, "architecture": architecture, "mode": mode,
        "seed": seed, "transform": transform, "length_k": length_k,
        "data_sha256": {"train": _frame_hash(train_df), "validation": _frame_hash(val_df),
                         "test": _frame_hash(test_df)},
        "label_names": names, "device": str(device),
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True,
                                          ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = _read_json(manifest_path)
        if previous.get("signature") != signature:
            raise ValueError(f"Existing run {run_dir} has different data, settings, code, or "
                             "software versions. Choose a NEW output root/run directory. "
                             "The existing result has been preserved.")
        if not resume:
            raise ValueError(f"{run_dir} already contains this run. Enable resume or choose "
                             "a new output directory.")
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            previous_result = _read_json(metrics_path)
            required = ("test_predictions.csv", "classification_report.csv", "confusion_matrix.csv",
                        "metrics_by_domain.csv", "metrics_by_benchmark.csv", "history.csv", "best.pt")
            if previous_result.get("status") == "complete" and all(
                    (run_dir / filename).exists() for filename in required):
                print(f"[{run_dir.name}] already complete; verified matching fingerprint.", flush=True)
                return previous_result
    elif any((run_dir / filename).exists() for filename in ("last.pt", "best.pt", "metrics.json")):
        raise ValueError(f"{run_dir} has checkpoint/results but no manifest. Use a new directory.")
    _atomic_json(manifest_path, {"signature": signature, **signature_payload})
    _atomic_json(run_dir / "config.json", cfg)
    _atomic_json(run_dir / "label_map.json", {str(i): name for i, name in enumerate(names)})
    vocabulary_path = run_dir / "vocabulary.json"
    if vocabulary_path.exists():
        vocab = _read_json(vocabulary_path)
    else:
        vocab = build_vocab(train_df, cfg, transform=transform, length_k=length_k)
        _atomic_json(vocabulary_path, vocab)
    datasets = [EncodedDataset(frame, vocab, mode, cfg, transform=transform, length_k=length_k)
                for frame in (train_df, val_df, test_df)]
    batch_size = int(cfg.get("batch_size", 64))
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    gradient_clip = float(cfg.get("gradient_clip", 1.0))
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive.")
    loader_options = {"batch_size": batch_size, "num_workers": int(cfg.get("num_workers", 0)),
                      "collate_fn": pad_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(datasets[0], shuffle=True, **loader_options)
    val_loader = DataLoader(datasets[1], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets[2], shuffle=False, **loader_options)
    common = {"vocab_size": max(vocab.values()) + 1,
              "embedding_dim": int(cfg.get("embedding_dim", 128)),
              "num_classes": num_classes, "dropout": float(cfg.get("dropout", 0.3))}
    if architecture == "cnn":
        model = TextCNN(**common, filters=int(cfg.get("cnn_filters", 128)),
                        kernels=cfg.get("cnn_kernels", [3, 4, 5]))
    else:
        model = BiLSTM(**common, hidden_dim=int(cfg.get("lstm_hidden", 128)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)))
    history: list[dict[str, Any]] = []
    start_epoch, stale, best_epoch = 0, 0, 0
    best_f1, elapsed_before = -1.0, 0.0
    started = time.perf_counter()
    last_path = run_dir / "last.pt"
    if last_path.exists():
        checkpoint = _load_checkpoint(last_path)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint fingerprint differs from its manifest; use a new output directory.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_epoch, best_f1 = int(checkpoint["best_epoch"]), float(checkpoint["best_f1"])
        stale, history = int(checkpoint["stale"]), checkpoint["history"]
        elapsed_before = float(checkpoint["elapsed_seconds"])
        _set_rng(checkpoint["rng"])
        print(f"[{run_dir.name}] resuming after completed epoch {start_epoch}.", flush=True)
    else:
        # A first-epoch interruption can also resume exactly from initialization.
        _atomic_checkpoint(last_path, {"signature": signature, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": 0, "best_epoch": 0,
            "best_f1": best_f1, "stale": 0, "history": [], "rng": _get_rng(),
            "elapsed_seconds": 0.0})
    try:
        for epoch in range(start_epoch + 1, int(cfg.get("epochs", 15)) + 1):
            if stale >= int(cfg.get("patience", 3)):
                break
            epoch_started = time.perf_counter()
            model.train()
            train_loss_sum, train_count = 0.0, 0
            for input_ids, lengths, labels, _ in train_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, lengths)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; check data and learning rate.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(labels)
                train_count += len(labels)
            validation = evaluate(model, val_loader, device, num_classes)
            if validation["macro_f1"] > best_f1 + 1e-12:
                best_f1, best_epoch, stale = validation["macro_f1"], epoch, 0
                _atomic_checkpoint(run_dir / "best.pt", {"signature": signature,
                    "model": model.state_dict(), "epoch": epoch, "validation_macro_f1": best_f1})
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": train_loss_sum / train_count,
                "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "epoch_seconds": time.perf_counter() - epoch_started, "best_epoch_so_far": best_epoch})
            _atomic_checkpoint(last_path, {"signature": signature, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                "best_f1": best_f1, "stale": stale, "history": history, "rng": _get_rng(),
                "elapsed_seconds": elapsed_before + time.perf_counter() - started})
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
            print(f"[{run_dir.name}] epoch {epoch:02d}: train loss={history[-1]['train_loss']:.4f}; "
                  f"val F1={validation['macro_f1']:.4f}; val acc={validation['accuracy']:.4f}; "
                  f"{history[-1]['epoch_seconds']:.1f}s", flush=True)
        best_checkpoint = _load_checkpoint(run_dir / "best.pt")
        if best_checkpoint["signature"] != signature:
            raise ValueError("Best checkpoint fingerprint differs from this run.")
        model.load_state_dict(best_checkpoint["model"])
        # Test labels/results are used only after all epoch selection is finished.
        evaluated = evaluate(model, test_loader, device, num_classes)
        _write_evaluation(run_dir, datasets[2].frame, evaluated, names)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        result = {"run_id": run_dir.name, "architecture": architecture, "mode": mode,
            "seed": seed, "transform": transform, "length_k": length_k,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "accuracy": evaluated["accuracy"], "macro_f1": evaluated["macro_f1"],
            "loss": evaluated["loss"], "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1, "epochs_completed": len(history),
            "seconds": elapsed_before + time.perf_counter() - started,
            "device": str(device), "status": "complete", "signature": signature,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "vocabulary_size": len(vocab), "label_names": names}
        bootstrap_frame = datasets[2].frame.iloc[evaluated["row_indices"]]
        group_field = "group_id" if "group_id" in bootstrap_frame else "prompt_id"
        result.update(grouped_bootstrap(evaluated["labels"], evaluated["predictions"],
            bootstrap_frame[group_field].astype(str).to_numpy(), num_classes,
            repetitions=int(cfg.get("bootstrap_reps", 200)),
            seed=int(cfg.get("bootstrap_seed", 100000)) + seed))
        if mode == "input_only":
            # Audit of the expected negative control, not a constraint imposed
            # on predictions: deterministic identical inputs must share a label.
            result["input_only_chance_accuracy"] = 1.0 / num_classes
            result["input_only_matches_chance"] = bool(np.isclose(evaluated["accuracy"], 1 / num_classes))
        _atomic_json(run_dir / "metrics.json", result)
        print(f"[{run_dir.name}] finished: test accuracy={result['accuracy']:.4f}; "
              f"macro-F1={result['macro_f1']:.4f}; best epoch={best_epoch}.", flush=True)
        return result
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError("GPU/CPU memory was exhausted. Set batch_size to 16 or 8 and use a "
                "NEW output root for the changed configuration, or resume this same configuration "
                "after freeing memory. Existing completed-epoch checkpoints are preserved. "
                "No experiment setting was changed automatically.") from error
        raise
