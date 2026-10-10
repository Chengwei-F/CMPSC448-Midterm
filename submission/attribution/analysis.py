"""Research analysis and report exports for the LLM attribution project.

The neural-network runner owns training. This module never selects an epoch or a
hyperparameter using test results. All descriptive text-feature comparisons use
the training split, and all learned preprocessing in the feature baseline is
fitted on the training split alone.
"""
from __future__ import annotations

import hashlib
import base64
import html
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Works from VS Code, Windows terminals, and headless jobs.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_SENTENCES = re.compile(r"[.!?]+(?:\s|$)")
_HEADINGS = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LISTS = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_NORMALIZE_WORD = re.compile(r"[^\w]", re.UNICODE)
_DIGITS = re.compile(r"\d")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")

FEATURE_COLUMNS = [
    "word_count", "char_count", "mean_word_chars", "mean_sentence_words",
    "paragraph_count", "heading_line_rate", "list_line_rate", "code_fence_count",
    "question_marks_per_1000_chars", "exclamations_per_1000_chars",
    "digit_char_rate", "punctuation_char_rate", "ttr_128_words",
]


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # BOM allows Excel on Windows to recognize the encoding correctly.
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _column(frame: pd.DataFrame, choices: list[str]) -> str:
    for name in choices:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing required column. Expected one of {choices}.")


def _label_column(frame: pd.DataFrame) -> str:
    return _column(frame, ["LLM_name", "label_name", "model_name", "label", "model"])


def _output_column(frame: pd.DataFrame) -> str:
    return _column(frame, ["LLM_output", "output", "response", "answer"])


def _get_labels(frame: pd.DataFrame, cfg: dict) -> list[str]:
    configured = cfg.get("label_names") or cfg.get("classes")
    if configured:
        return [str(x) for x in configured]
    return sorted(frame[_label_column(frame)].astype(str).unique())


def _slug(value: str) -> str:
    cleaned = _SAFE_FILENAME.sub("_", str(value)).strip("_.")
    if len(cleaned) > 125:
        cleaned = cleaned[:110] + "_" + hashlib.sha256(cleaned.encode()).hexdigest()[:10]
    return cleaned or "run"


def _features(text: str) -> dict[str, float]:
    """One whitespace tokenization per response; do not truncate for lengths.

    These are deliberately transparent heuristics, not linguistic ground truth.
    A sentence ends at . ! or ? followed by whitespace/end of string. TTR uses
    exactly the first 128 whitespace words, lowercased and stripped of punctuation;
    shorter responses get a missing TTR instead of a noncomparable short-window TTR.
    """
    text = str(text)
    words = text.split()
    n_words, n_chars = len(words), len(text)
    n_sentences = max(1, len(_SENTENCES.findall(text)))
    n_lines = max(1, text.count("\n") + 1)
    window = [_NORMALIZE_WORD.sub("", word.lower()) for word in words[:128]]
    window = [word for word in window if word]
    ttr = len(set(window)) / len(window) if n_words >= 128 and window else np.nan
    return {
        "word_count": float(n_words),
        "char_count": float(n_chars),
        "mean_word_chars": sum(map(len, words)) / max(1, n_words),
        "mean_sentence_words": n_words / n_sentences,
        "paragraph_count": float(len([p for p in re.split(r"\n\s*\n", text) if p.strip()])),
        "heading_line_rate": len(_HEADINGS.findall(text)) / n_lines,
        "list_line_rate": len(_LISTS.findall(text)) / n_lines,
        "code_fence_count": float(text.count("```")),
        "question_marks_per_1000_chars": 1000 * text.count("?") / max(1, n_chars),
        "exclamations_per_1000_chars": 1000 * text.count("!") / max(1, n_chars),
        "digit_char_rate": len(_DIGITS.findall(text)) / max(1, n_chars),
        "punctuation_char_rate": len(_PUNCTUATION.findall(text)) / max(1, n_chars),
        "ttr_128_words": ttr,
    }


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return identifiers, labels, and auditable response-level features."""
    label_col, output_col = _label_column(frame), _output_column(frame)
    metadata = [x for x in ["record_id", "prompt_id", "group_id", "task_domain", "benchmark"] if x in frame]
    out = frame[metadata].copy().reset_index(drop=True)
    out["LLM_name"] = frame[label_col].astype(str).to_numpy()
    computed = pd.DataFrame.from_records(_features(text) for text in frame[output_col].fillna(""))
    return pd.concat([out, computed], axis=1)


def _cached_features(frame: pd.DataFrame, directory: Path, split: str) -> pd.DataFrame:
    """Reuse a cache only for the exact response/identifier content, not row count."""
    directory.mkdir(parents=True, exist_ok=True)
    path, stamp_path = directory / f"{split}_features.csv", directory / f"{split}_features.sha256"
    relevant = [x for x in ["record_id", "prompt_id", "group_id", "task_domain", "benchmark",
                             _label_column(frame), _output_column(frame)] if x in frame]
    hasher = hashlib.sha256(Path(__file__).read_bytes())
    hasher.update(pd.util.hash_pandas_object(frame[relevant].astype(str), index=False).values.tobytes())
    digest = hasher.hexdigest()
    if path.exists() and stamp_path.exists() and stamp_path.read_text(encoding="utf-8") == digest:
        cached = pd.read_csv(path, dtype={x: str for x in ["record_id", "prompt_id", "group_id", "LLM_name"]})
        if len(cached) == len(frame) and set(FEATURE_COLUMNS).issubset(cached.columns):
            return cached
    result = extract_features(frame)
    _csv_write(result, path)
    stamp_path.write_text(digest, encoding="utf-8")
    return result


def describe_data(splits: dict[str, pd.DataFrame], out_dir: Path, cfg: dict) -> None:
    """Save split counts and TRAINING-ONLY linguistic summaries/figures."""
    out_dir = Path(out_dir)
    tables, figures = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    counts = []
    for split, frame in splits.items():
        label = _label_column(frame)
        group_cols = [label] + [x for x in ["task_domain", "benchmark"] if x in frame]
        summary = frame.groupby(group_cols, dropna=False).size().rename("n_responses").reset_index()
        summary = summary.rename(columns={label: "LLM_name"})
        summary.insert(0, "split", split)
        counts.append(summary)
    _csv_write(pd.concat(counts, ignore_index=True), tables / "sample_counts.csv")
    train = _cached_features(splits["train"], out_dir / "features", "train")
    labels = _get_labels(splits["train"], cfg)
    grouping = ["LLM_name"] + (["task_domain"] if "task_domain" in train else [])
    summary = train.groupby(grouping, dropna=False)[FEATURE_COLUMNS].agg(["count", "mean", "std", "median"])
    summary.columns = ["__".join(col) for col in summary.columns]
    _csv_write(summary.reset_index(), tables / "training_features_by_model_domain.csv")
    by_model = train.groupby("LLM_name")[FEATURE_COLUMNS].agg(["count", "mean", "std", "median"])
    by_model.columns = ["__".join(col) for col in by_model.columns]
    _csv_write(by_model.reset_index(), tables / "training_features_by_model.csv")

    # Fixed-window lexical comparison uses identical qualifying prompt sets.
    pair_col = "prompt_id" if "prompt_id" in train else "group_id"
    if pair_col not in train:
        raise ValueError("Feature comparisons require prompt_id or group_id for paired eligibility.")
    paired = train.groupby(pair_col).agg(n_models=("LLM_name", "nunique"), min_words=("word_count", "min"), n_rows=("LLM_name", "size"))
    eligible = paired.index[(paired.n_models == len(labels)) & (paired.n_rows == len(labels)) & (paired.min_words >= 128)]
    lexical = train[train[pair_col].isin(eligible)].copy()
    _csv_write(lexical[[pair_col, "LLM_name", "ttr_128_words"]], tables / "training_paired_ttr128.csv")
    lexical_summary = lexical.groupby("LLM_name").agg(n_responses=("ttr_128_words", "size"), mean=("ttr_128_words", "mean"), std=("ttr_128_words", "std"), median=("ttr_128_words", "median")).reset_index()
    _csv_write(lexical_summary, tables / "training_paired_ttr128_summary.csv")
    _json_write(tables / "descriptive_methods.json", {
        "descriptive_split": "train", "ttr_window": "first 128 whitespace words; punctuation-stripped lowercase types",
        "paired_ttr_prompt_count": len(eligible), "paired_ttr_rule": "all model responses have at least 128 whitespace words",
        "sentence_rule": ". ! or ? followed by whitespace/end; approximate, particularly for code/math",
        "feature_interpretation": "Associations under this dataset's generation settings; not causal effects or universal model signatures.",
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [train.loc[train.LLM_name == label, "word_count"].to_numpy() for label in labels]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("Response words (symlog scale)")
    ax.set_title("Training responses: length distribution (outliers hidden)")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.tight_layout(); fig.savefig(figures / "training_response_lengths.png", dpi=160); plt.close(fig)
    if len(lexical):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([lexical.loc[lexical.LLM_name == label, "ttr_128_words"].to_numpy() for label in labels], tick_labels=labels, showfliers=False)
        ax.set_ylabel("Type-token ratio, fixed 128-word window")
        ax.set_title(f"Training lexical diversity: {len(eligible):,} shared eligible prompts")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        fig.tight_layout(); fig.savefig(figures / "training_paired_lexical_diversity.png", dpi=160); plt.close(fig)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))}


def run_feature_baseline(splits: dict[str, pd.DataFrame], outdir: Path, cfg: dict) -> dict:
    """RQ4 interpretable baseline with fixed C=1, trained without test selection.

    This function uses each supplied split as-is. The caller must already verify
    prompt/group separation and complete model pairs. Coefficients are predictive
    associations, affected by correlated features, and cannot identify causation.
    """
    from .data import _original_span

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_budget = int(cfg.get("output_max_tokens", 512))
    if output_budget < 1:
        raise ValueError("Feature baseline requires a positive output_max_tokens.")
    features = {}
    for name, frame in splits.items():
        # Descriptive full-response features must never be reused here: the
        # baseline observes only the same original token span as the neural model.
        prefix_frame = frame.copy()
        output_column = _output_column(prefix_frame)
        prefix_frame[output_column] = prefix_frame[output_column].map(lambda text: _original_span(str(text), output_budget))
        features[name] = _cached_features(prefix_frame, outdir / "prefix_features", name)
    val_name = "validation" if "validation" in features else "val"
    train, val, test = features["train"], features[val_name], features["test"]
    labels = _get_labels(splits["train"], cfg)
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=int(cfg.get("seeds", [42])[0]))),
    ])
    pipeline.fit(train[FEATURE_COLUMNS], train.LLM_name)
    results: dict[str, Any] = {"architecture": "feature_logistic", "mode": "output_features", "condition": "fixed_interpretable_features",
                               "seed": int(cfg.get("seeds", [42])[0]),
                               "n_train": len(train), "n_val": len(val), "n_test": len(test), "runstatus": "completed"}
    for split, frame in [("validation", val), ("test", test)]:
        predictions = pipeline.predict(frame[FEATURE_COLUMNS])
        scores = _metrics(frame.LLM_name.to_numpy(), predictions, labels)
        results[split] = scores
        if split == "test":
            results.update(scores)
            metadata = [x for x in ["record_id", "prompt_id", "group_id", "task_domain", "benchmark"] if x in frame]
            pred_frame = frame[metadata].copy()
            pred_frame["true_label"] = frame.LLM_name
            pred_frame["pred_label"] = predictions
            probabilities = pipeline.predict_proba(frame[FEATURE_COLUMNS])
            estimator_labels = list(pipeline.named_steps["classifier"].classes_)
            for i, label in enumerate(labels):
                pred_frame[f"prob_{i}"] = probabilities[:, estimator_labels.index(label)]
            _csv_write(pred_frame, outdir / "test_predictions.csv")
            _csv_write(pd.DataFrame(classification_report(frame.LLM_name, predictions, labels=labels, output_dict=True, zero_division=0)).T.rename_axis("class").reset_index(), outdir / "classification_report.csv")
    names = pipeline.named_steps["impute"].get_feature_names_out(FEATURE_COLUMNS)
    classifier = pipeline.named_steps["classifier"]
    rows = [{"LLM_name": label, "feature": feature, "coefficient_per_training_sd": float(value)}
            for label, coefficients in zip(classifier.classes_, classifier.coef_) for feature, value in zip(names, coefficients)]
    _csv_write(pd.DataFrame(rows), outdir / "feature_coefficients.csv")
    _json_write(outdir / "metrics.json", results)
    _json_write(outdir / "feature_baseline_methods.json", {
        "features": FEATURE_COLUMNS, "label_order": labels, "C": 1.0, "solver": "lbfgs", "max_iter": 3000,
        "preprocessing_fit": "training split only", "missing_ttr": "training-median imputation plus missingness indicator",
        "information_scope": "All predictive features use the same original output prefix available to the neural baseline; separate full-response training descriptions are not classifier inputs.",
        "output_max_tokens": output_budget,
        "selection": "fixed a priori; validation scores reported without selecting C", "causal_interpretation": False,
    })
    try:
        import joblib
        joblib.dump(pipeline, outdir / "feature_baseline.joblib")
    except ImportError:
        pass
    return results


def _load_index(root: Path) -> list[dict]:
    path = root / "experiment_index.json"
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj.get("runs", obj.get("experiments", []))
    if not isinstance(obj, list):
        raise ValueError("experiment_index.json must be a list, or an object with a runs list.")
    return obj


def _run_path(root: Path, record: dict) -> Path:
    path = Path(record["run_dir"])
    return path if path.is_absolute() else root / path


def _read_predictions(path: Path) -> pd.DataFrame | None:
    file = path / "test_predictions.csv"
    if not file.exists():
        return None
    return pd.read_csv(file, dtype={x: str for x in ["record_id", "prompt_id", "group_id", "true_label", "pred_label"]})


def _score_confusion(matrix: np.ndarray) -> tuple[float, float]:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(len(matrix), dtype=float), where=denominator > 0)
    return float(np.trace(matrix) / max(1, matrix.sum())), float(f1.mean())


def paired_group_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, reps: int = 500, seed: int = 20261005) -> dict:
    """Candidate minus reference, resampling whole leakage groups together.

    Test record IDs must be identical. Using an intersection could silently change
    the estimand for an ablation with a shorter-response selection restriction.
    """
    required = {"record_id", "true_label", "pred_label"}
    if not required.issubset(reference) or not required.issubset(candidate):
        return {"comparison_status": "missing_prediction_columns"}
    if reference.record_id.duplicated().any() or candidate.record_id.duplicated().any():
        return {"comparison_status": "duplicate_record_ids"}
    if set(reference.record_id) != set(candidate.record_id):
        return {"comparison_status": "different_test_record_ids", "n_reference": len(reference), "n_candidate": len(candidate)}
    ref = reference.sort_values("record_id").reset_index(drop=True)
    cand = candidate.sort_values("record_id").reset_index(drop=True)
    if not ref.true_label.equals(cand.true_label):
        return {"comparison_status": "inconsistent_true_labels"}
    group_col = "group_id" if "group_id" in ref else "prompt_id"
    if group_col not in ref or ref[group_col].isna().any():
        return {"comparison_status": "missing_group_ids"}
    if group_col in cand and not ref[group_col].equals(cand[group_col]):
        return {"comparison_status": "inconsistent_group_ids"}
    labels = sorted(set(ref.true_label) | set(ref.pred_label) | set(cand.pred_label))
    label_map = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    group_codes, unique_groups = pd.factorize(ref[group_col], sort=True)
    n_groups = len(unique_groups)
    actual = ref.true_label.map(label_map).to_numpy()
    ref_pred, cand_pred = ref.pred_label.map(label_map).to_numpy(), cand.pred_label.map(label_map).to_numpy()
    ref_group = np.bincount(group_codes * k * k + actual * k + ref_pred, minlength=n_groups * k * k).reshape(n_groups, k * k)
    cand_group = np.bincount(group_codes * k * k + actual * k + cand_pred, minlength=n_groups * k * k).reshape(n_groups, k * k)
    ref_score = _score_confusion(ref_group.sum(axis=0).reshape(k, k))
    cand_score = _score_confusion(cand_group.sum(axis=0).reshape(k, k))
    result: dict[str, Any] = {"comparison_status": "ok", "n_test": len(ref), "n_groups": n_groups,
                              "accuracy_delta": cand_score[0] - ref_score[0], "macro_f1_delta": cand_score[1] - ref_score[1],
                              "reference_accuracy": ref_score[0], "candidate_accuracy": cand_score[0],
                              "reference_macro_f1": ref_score[1], "candidate_macro_f1": cand_score[1],
                              "bootstrap_reps": int(reps), "bootstrap_seed": int(seed), "bootstrap_unit": group_col}
    if n_groups < 2 or reps < 2:
        result["ci_status"] = "too_few_groups_or_replicates"
        return result
    rng = np.random.default_rng(seed)
    deltas = np.empty((reps, 2))
    for b in range(reps):
        weights = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
        r = _score_confusion((weights @ ref_group).reshape(k, k))
        c = _score_confusion((weights @ cand_group).reshape(k, k))
        deltas[b] = np.asarray(c) - np.asarray(r)
    for i, metric in enumerate(["accuracy", "macro_f1"]):
        low, high = np.quantile(deltas[:, i], [0.025, 0.975])
        result[f"{metric}_delta_ci_low"] = float(low)
        result[f"{metric}_delta_ci_high"] = float(high)
    result["ci_status"] = "percentile_95_conditional_on_this_fitted_seed"
    return result


def _resolve_reference(root: Path, record: dict, index: list[dict]) -> Path | None:
    if record.get("reference_run_dir"):
        path = Path(record["reference_run_dir"])
        return path if path.is_absolute() else root / path
    reference_condition = record.get("reference_condition")
    if not reference_condition:
        return None
    keys = ["rq", "architecture", "mode", "seed", "target"]
    matches = [entry for entry in index if entry.get("condition") == reference_condition
               and all(entry.get(key) == record.get(key) for key in keys)]
    if len(matches) == 1:
        return _run_path(root, matches[0])
    return None


def _confusion_plot(predictions: pd.DataFrame, destination: Path, title: str, label_names: list[str] | None = None) -> None:
    labels = sorted(set(predictions.true_label) | set(predictions.pred_label))
    display_labels = [label_names[int(label)] if label_names and str(label).isdigit() and int(label) < len(label_names) else label for label in labels]
    counts = confusion_matrix(predictions.true_label, predictions.pred_label, labels=labels)
    norm = np.divide(counts, counts.sum(axis=1, keepdims=True), out=np.zeros_like(counts, dtype=float), where=counts.sum(axis=1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > .55 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), display_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), display_labels, fontsize=8)
    ax.set_xlabel("Predicted model"); ax.set_ylabel("True model"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Fraction within true class")
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def _history_plot(path: Path, destination: Path, title: str) -> bool:
    history_file = path / "history.csv"
    if not history_file.exists():
        return False
    frame = pd.read_csv(history_file)
    if frame.empty:
        return False
    x = frame["epoch"] if "epoch" in frame else np.arange(1, len(frame) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for column in ["train_loss", "val_loss", "validation_loss"]:
        if column in frame:
            axes[0].plot(x, frame[column], label=column)
    for column in ["val_accuracy", "val_macro_f1", "validation_accuracy", "validation_macro_f1"]:
        if column in frame:
            axes[1].plot(x, frame[column], label=column)
    for ax in axes:
        ax.set_xlabel("Epoch")
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Cross-entropy loss"); axes[1].set_ylabel("Validation score")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)
    return True


def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    group_cols = [x for x in group_cols if x in frame]
    if frame.empty or not group_cols:
        return pd.DataFrame()
    measures = [x for x in ["accuracy", "macro_f1"] if x in frame]
    if not measures:
        return pd.DataFrame()
    agg = frame.groupby(group_cols, dropna=False)[measures].agg(["mean", "std", "count"])
    agg.columns = ["_".join(x) for x in agg.columns]
    return agg.reset_index()


def _html_table(frame: pd.DataFrame, limit: int = 80) -> str:
    if frame.empty:
        return '<p class="missing">Not run / no completed results available.</p>'
    visible = frame.head(limit)
    table = visible.to_html(index=False, border=0, escape=True, float_format=lambda v: f"{v:.4f}", na_rep="—")
    if len(frame) > limit:
        table += f"<p>Showing {limit} of {len(frame)} rows. The CSV file contains every row.</p>"
    return table


def _optional_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists() and path.stat().st_size > 3:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                pass
    return pd.DataFrame()


def _export_summary_pdf(root: Path, cfg: dict, main: pd.DataFrame, summary: pd.DataFrame,
                        comparisons: pd.DataFrame, stage_status: list[dict], figures: list[tuple[str, str]]) -> None:
    """Portable PDF export using matplotlib only; not an auto-written paper."""
    with PdfPages(root / "results_summary.pdf") as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        lines = ["Who Wrote It? LLM Attribution", "Measured Results Summary", "",
                 f"Execution profile: {cfg.get('profile', 'unspecified')}",
                 "SMOKE TEST: SOFTWARE CHECK ONLY" if str(cfg.get("profile", "")).lower() == "smoke" else "DRAFT: verify results and write your own interpretation.", ""]
        lines += [f"{item['research_question']}: {item['status']}" for item in stage_status]
        lines += ["", "Numbers in the CSV files are proportions: 0.60 means 60%.",
                  "Main error bars show SD across available training seeds; one seed has no SD.",
                  "Paired intervals resample complete test groups and condition on one fitted seed.",
                  "The feature summaries describe the training set; coefficients are associations.",
                  "Missing runs are not filled with invented values.", "",
                  "This file is a results appendix, not a completed final project report.",
                  "Use report_draft.html as the report scaffold and replace STUDENT placeholders."]
        fig.text(.07, .93, "\n".join(lines), va="top", fontsize=13, linespacing=1.6)
        plt.axis("off"); pdf.savefig(fig); plt.close(fig)

        def table_pages(frame: pd.DataFrame, title: str, columns: list[str]) -> None:
            if frame.empty:
                return
            selected = frame[[column for column in columns if column in frame]].copy()
            for column in selected:
                if pd.api.types.is_numeric_dtype(selected[column]):
                    selected[column] = selected[column].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
                else:
                    selected[column] = selected[column].fillna("").astype(str).map(lambda value: "\n".join(textwrap.wrap(value, 22)))
            for start in range(0, len(selected), 16):
                chunk = selected.iloc[start:start + 16]
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis("off"); ax.set_title(title + f" (rows {start + 1}-{start + len(chunk)})", pad=22, fontsize=13)
                table = ax.table(cellText=chunk.values, colLabels=[column.replace("_", "\n") for column in chunk.columns], cellLoc="left", loc="center")
                table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 2.4)
                for (row, _), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor("#eaf0f5"); cell.set_height(.095)
                    else:
                        cell.set_height(min(.052, .78 / max(1, len(chunk))))
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        table_pages(main, "RQ1/RQ2: test results", ["architecture", "mode", "condition", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "macro_f1_count"])
        table_pages(summary, "All completed conditions", ["rq", "architecture", "mode", "condition", "target", "accuracy_mean", "macro_f1_mean", "macro_f1_count"])
        table_pages(comparisons, "Paired comparisons: condition minus reference", ["rq", "architecture", "condition", "target", "seed", "macro_f1_delta", "macro_f1_delta_ci_low", "macro_f1_delta_ci_high", "comparison_status"])
        # Put overview figures first, then representative confusion/training plots.
        ordered = sorted(figures, key=lambda item: (item[0].startswith(("confusion_", "learning_")), item[0]))
        limit = int(cfg.get("pdf_max_figures", 24))
        for filename, caption in ordered[:limit]:
            source = root / "figures" / filename
            if not source.exists():
                continue
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(plt.imread(source)); ax.axis("off")
            fig.text(.05, .025, textwrap.fill(caption, 135), fontsize=8)
            fig.tight_layout(rect=[0, .045, 1, 1]); pdf.savefig(fig); plt.close(fig)


def generate_report(output_root: Path, cfg: dict) -> None:
    """Export real results available so far; explicitly mark missing stages.

    Expected index entries include rq, run_dir, architecture, mode, seed,
    condition, and optional target and reference_run_dir/reference_condition.
    RQ3/RQ4 paired differences require an explicit reference, never a guessed
    comparison between different test subsets.
    """
    root = Path(output_root)
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    index = _load_index(root)
    metrics_rows, domain_rows, comparison_rows, status_rows = [], [], [], []
    figure_entries: list[tuple[str, str]] = []
    seen_run_dirs: set[str] = set()
    representative_keys: set[tuple] = set()
    for record in index:
        if "run_dir" not in record:
            status_rows.append({**record, "report_status": "index_missing_run_dir"}); continue
        path = _run_path(root, record)
        metrics_file = path / "metrics.json"
        if not metrics_file.exists():
            status_rows.append({**record, "report_status": "metrics_missing"}); continue
        scores = json.loads(metrics_file.read_text(encoding="utf-8"))
        test_scores = scores.get("test", {})
        row = {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
        for key, value in scores.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
        for key in ["accuracy", "macro_f1"]:
            if key not in row and key in test_scores:
                row[key] = test_scores[key]
            if key not in row and f"test_{key}" in scores:
                row[key] = scores[f"test_{key}"]
        if not all(key in row and pd.notna(row[key]) for key in ["accuracy", "macro_f1"]):
            status_rows.append({**record, "report_status": "test_metrics_missing"}); continue
        # A shared RQ1/RQ2 run may be referenced twice; avoid double-counting seeds.
        unique_key = str(path.resolve())
        if unique_key in seen_run_dirs:
            continue
        seen_run_dirs.add(unique_key)
        row["run_dir"] = str(record["run_dir"])
        metrics_rows.append(row)
        status_rows.append({**record, "report_status": "completed"})
        predictions = _read_predictions(path)
        if predictions is None or not {"true_label", "pred_label"}.issubset(predictions):
            continue
        labels = sorted(predictions.true_label.unique())
        if "task_domain" in predictions:
            for domain, subset in predictions.groupby("task_domain", dropna=False):
                domain_rows.append({key: row.get(key) for key in ["rq", "architecture", "mode", "condition", "target", "seed"]}
                                   | {"task_domain": domain, "n_test": len(subset)}
                                   | _metrics(subset.true_label.to_numpy(), subset.pred_label.to_numpy(), labels))
        reference_path = _resolve_reference(root, record, index)
        if reference_path is not None:
            reference = _read_predictions(reference_path)
            metadata = {key: row.get(key) for key in ["rq", "architecture", "mode", "condition", "target", "seed"]}
            metadata["reference_run_dir"] = str(reference_path)
            if reference is None:
                comparison_rows.append(metadata | {"comparison_status": "reference_predictions_missing"})
            else:
                run_seed = int(row["seed"]) if row.get("seed") is not None else 42
                comparison_rows.append(metadata | paired_group_bootstrap(reference, predictions,
                    reps=int(cfg.get("bootstrap_reps", 500)), seed=int(cfg.get("bootstrap_seed", 20261005)) + run_seed))
        # One representative seed per condition; all seeds remain in tables.
        rep_key = tuple(str(row.get(key, "")) for key in ["rq", "architecture", "mode", "condition", "target"])
        if rep_key not in representative_keys:
            representative_keys.add(rep_key)
            name = _slug("_".join(rep_key) + "_seed_" + str(row.get("seed", "NA")))
            title = " | ".join(x for x in rep_key if x) + f" | seed {row.get('seed', 'NA')}"
            destination = figures / f"confusion_{name}.png"
            _confusion_plot(predictions, destination, title, cfg.get("label_names"))
            figure_entries.append((destination.name, title + " (row-normalized confusion matrix)"))
            destination = figures / f"learning_{name}.png"
            if _history_plot(path, destination, title):
                figure_entries.append((destination.name, title + " (training/validation only)"))

    metrics_frame = pd.DataFrame(metrics_rows)
    _csv_write(metrics_frame, tables / "all_run_metrics.csv")
    _csv_write(pd.DataFrame(status_rows), tables / "run_status.csv")
    domains = pd.DataFrame(domain_rows)
    _csv_write(domains, tables / "domain_metrics_by_run.csv")
    domain_summary = _aggregate(domains, ["rq", "architecture", "mode", "condition", "target", "task_domain"])
    _csv_write(domain_summary, tables / "domain_results.csv")
    summary = _aggregate(metrics_frame, ["rq", "architecture", "mode", "condition", "target"])
    _csv_write(summary, tables / "results_summary.csv")
    if not metrics_frame.empty:
        rq_text = metrics_frame.get("rq", pd.Series("", index=metrics_frame.index)).astype(str).str.upper()
        main_frame = metrics_frame[rq_text.str.contains(r"RQ[12]|MAIN", regex=True)]
    else:
        main_frame = pd.DataFrame()
    main_summary = _aggregate(main_frame, ["architecture", "mode", "condition"])
    _csv_write(main_summary, tables / "main_results.csv")
    comparisons = pd.DataFrame(comparison_rows)
    _csv_write(comparisons, tables / "paired_comparisons.csv")
    rq2_comparisons = pd.DataFrame()
    advanced_comparisons = pd.DataFrame()
    if not comparisons.empty:
        comparison_rq = comparisons.rq.astype(str).str.upper()
        rq2_comparisons = comparisons[comparison_rq.str.contains(r"MAIN|RQ2", regex=True)
                                     & comparisons["mode"].eq("input_and_output")]
        advanced_comparisons = comparisons[comparison_rq.str.contains(r"RQ3|RQ4", regex=True)]
    _csv_write(rq2_comparisons, tables / "rq2_paired_deltas.csv")
    for rq in ["RQ3", "RQ4"]:
        subset = comparisons[comparisons.rq.astype(str).str.upper().str.contains(rq)] if not comparisons.empty else pd.DataFrame()
        _csv_write(subset, tables / f"{rq.lower()}_paired_deltas.csv")
    if not comparisons.empty and "macro_f1_delta" in comparisons:
        valid = comparisons[comparisons.comparison_status == "ok"]
        columns = [x for x in ["rq", "architecture", "mode", "condition", "target"] if x in valid]
        delta_summary = valid.groupby(columns, dropna=False)[["accuracy_delta", "macro_f1_delta"]].agg(["mean", "std", "count"])
        delta_summary.columns = ["_".join(x) for x in delta_summary.columns]
        _csv_write(delta_summary.reset_index(), tables / "paired_delta_seed_summary.csv")

    if not main_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4.7))
        names = main_summary["architecture"].astype(str) + " / " + main_summary["mode"].astype(str)
        mean = main_summary.macro_f1_mean.to_numpy()
        error = main_summary.macro_f1_std.fillna(0).to_numpy()
        ax.bar(np.arange(len(names)), mean, yerr=error, capsize=4, color="#31688e")
        ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("Test Macro-F1")
        ax.set_title("Main experiments: mean and SD across available seeds")
        fig.tight_layout(); fig.savefig(figures / "main_macro_f1.png", dpi=160); plt.close(fig)
        figure_entries.insert(0, ("main_macro_f1.png", "Main experiments; SD is undefined with one seed and omitted."))
    if not domain_summary.empty:
        focus = domain_summary[domain_summary.get("rq", pd.Series("", index=domain_summary.index)).astype(str).str.upper().str.contains(r"RQ[12]|MAIN", regex=True)]
        if not focus.empty:
            pivot = focus.pivot_table(index="task_domain", columns=["architecture", "mode"], values="macro_f1_mean", aggfunc="mean")
            ax = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1), ylabel="Test Macro-F1", title="Main experiments by task domain")
            ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
            fig = ax.figure; fig.tight_layout(); fig.savefig(figures / "domain_comparison.png", dpi=160); plt.close(fig)
            figure_entries.append(("domain_comparison.png", "Within-domain test performance; this is separate from the held-out-domain RQ3 experiment."))
    if not comparisons.empty and "macro_f1_delta" in comparisons:
        valid = comparisons[comparisons.comparison_status == "ok"].copy()
        if len(valid):
            fig, ax = plt.subplots(figsize=(10, max(4, .33 * len(valid))))
            names = valid["condition"].astype(str) + " / " + valid["architecture"].astype(str)
            if "target" in valid:
                names += valid["target"].fillna("").astype(str).map(lambda value: f" / {value}" if value else "")
            names += " / seed " + valid["seed"].astype(str)
            ax.scatter(valid.macro_f1_delta, np.arange(len(valid)), color="#31688e")
            if {"macro_f1_delta_ci_low", "macro_f1_delta_ci_high"}.issubset(valid):
                ax.hlines(np.arange(len(valid)), valid.macro_f1_delta_ci_low, valid.macro_f1_delta_ci_high, color="#31688e")
            ax.axvline(0, color="gray", linestyle="--"); ax.set_yticks(np.arange(len(valid)), names, fontsize=7)
            ax.set_xlabel("Macro-F1 difference: condition minus explicit reference")
            ax.set_title("Paired comparisons: group-bootstrap 95% intervals (per fitted seed)")
            fig.tight_layout(); fig.savefig(figures / "paired_differences.png", dpi=160); plt.close(fig)
            figure_entries.append(("paired_differences.png", "Intervals resample whole test groups; they do not include between-training-seed uncertainty."))

    profile = str(cfg.get("profile", "unspecified"))
    is_smoke = profile.lower() in {"smoke", "debug", "test"}
    observed_rqs = " ".join(metrics_frame.get("rq", pd.Series(dtype=str)).astype(str)).upper()
    stage_status = [{"research_question": rq, "status": "results available" if rq in observed_rqs else "not run / not yet available"}
                    for rq in ["RQ1", "RQ2", "RQ3", "RQ4"]]
    # Distinguish availability of one dependency run from coverage of both
    # required architectures and all six core conditions.
    available_main = set(zip(main_frame.get("architecture", []), main_frame.get("mode", [])))
    required_rq1 = {(arch, "output_only") for arch in ["cnn", "bilstm"]}
    required_rq2 = {(arch, mode) for arch in ["cnn", "bilstm"]
                    for mode in ["input_only", "output_only", "input_and_output"]}
    for i, required in enumerate([required_rq1, required_rq2]):
        covered = len(available_main & required)
        stage_status[i]["status"] = (f"{'Complete' if covered == len(required) else 'Partial'} condition coverage: "
                                     f"{covered}/{len(required)}; completed seed counts shown below")
    _csv_write(pd.DataFrame(stage_status), tables / "research_question_status.csv")
    counts = _optional_csv([root / "descriptive" / "tables" / "sample_counts.csv", tables / "sample_counts.csv"])
    if not counts.empty:
        counts = counts.groupby(["split", "LLM_name"], dropna=False)["n_responses"].sum().reset_index()
    description = _optional_csv([root / "descriptive" / "tables" / "training_features_by_model.csv", tables / "training_features_by_model.csv"])
    display_columns = [x for x in ["LLM_name", "word_count__mean", "word_count__median", "mean_sentence_words__mean",
                                    "heading_line_rate__mean", "list_line_rate__mean", "code_fence_count__mean"] if x in description]
    if display_columns:
        description = description[display_columns]
    lexical = _optional_csv([root / "descriptive" / "tables" / "training_paired_ttr128_summary.csv", tables / "training_paired_ttr128_summary.csv"])
    coefficients = pd.DataFrame()
    coefficient_paths = [_run_path(root, record) / "feature_coefficients.csv" for record in index if "run_dir" in record]
    coefficients = _optional_csv(coefficient_paths + [root / "rq4_features" / "feature_coefficients.csv"])
    if not coefficients.empty:
        coefficients["absolute_coefficient"] = coefficients.coefficient_per_training_sd.abs()
        coefficients = coefficients.sort_values("absolute_coefficient", ascending=False).groupby("LLM_name", sort=False).head(5).drop(columns="absolute_coefficient")
        coefficients = coefficients.sort_values("LLM_name", kind="stable")
    esc = html.escape
    sections = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><title>LLM Attribution Results</title>",
        "<style>body{font:15px/1.55 Arial,sans-serif;max-width:1150px;margin:35px auto;padding:0 24px;color:#202832}h1,h2,h3{line-height:1.2}table{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}th,td{border:1px solid #ccd5dc;padding:6px 9px;text-align:left}th{background:#eaf0f5}img{max-width:100%;height:auto}.notice{padding:14px;background:#fff0d7;border-left:5px solid #d99b2b}.missing,.placeholder{color:#7a4512}.figure{break-inside:avoid;margin:25px 0}pre{white-space:pre-wrap;background:#f4f6f8;padding:14px} @media print{body{margin:0}table{font-size:9px}h2{break-after:avoid}}</style></head><body>",
        "<h1>Who Wrote It? Identifying LLMs from Their Responses</h1>",
        f"<p>Automatically assembled results draft. Profile: <strong>{esc(profile)}</strong>. Completed unique runs: {len(metrics_frame)}.</p>",
        "<p class='notice'>" + ("SMOKE TEST ONLY — these small-run results verify the software and must not be presented as the final experiment." if is_smoke else "This draft contains only available, saved results. Missing stages and single-seed uncertainty remain visible. Review the design and add your own interpretation before submission.") + "</p>",
        "<h2>Research questions and completion status</h2>", _html_table(pd.DataFrame(stage_status)),
        "<ol><li>RQ1: Can CNN and BiLSTM identify the generating model from its output?</li><li>RQ2: How do input-only, output-only, and input-plus-output attribution compare?</li><li>RQ3: How well does attribution transfer to a task domain absent from training?</li><li>RQ4: Which linguistic/structural features are predictive, and how do controlled ablations change accuracy?</li></ol>",
        "<h2>Data and experimental methods</h2>",
        "<p>Use the prepared prompt-paired data with one response per model for each retained prompt. Keep all responses to a prompt, and all linked near-duplicate prompts, in one split. Fit vocabulary, imputers, and scaling on training data only. Select neural checkpoints by validation Macro-F1; reserve the test split for final evaluation of each predeclared condition. Input-plus-output must preserve the same response-token budget as output-only.</p>",
        "<p>Input-only is a negative control: with complete balanced model pairs and identical model-visible inputs, deterministic predictions have accuracy exactly 1/K for K model classes. Macro-F1 need not equal 1/K.</p>",
        "<p class='placeholder'>STUDENT: Describe data provenance, exact model versions, retained sample counts, exclusions, tokenization, architecture, and training settings using the accompanying configuration and audit files. Cite the dataset authors and original task sources.</p>",
        "<h3>Response counts by split and model</h3>", _html_table(counts),
        "<h2>RQ1 and RQ2: main results</h2>", _html_table(main_summary),
        "<p>Reported SD is across completed random seeds; a blank SD means fewer than two seeds. RQ1 uses the output-only rows of this comparison.</p>",
        "<p class='placeholder'>STUDENT: Compare architectures and input conditions using the measured effect sizes. Explain whether the input-only negative control behaves as expected. Do not claim a difference is reliable from one seed.</p>",
        "<h3>RQ2: Does adding the prompt help?</h3>", _html_table(rq2_comparisons),
        "<p>These paired differences are input-plus-output minus output-only for the same architecture, seed and test records, with the same output-token budget. Positive values indicate better attribution after adding the prompt. Group-bootstrap intervals condition on each fitted pair of models; compare their pattern across training seeds as well.</p>",
        "<h2>All conditions</h2>", _html_table(summary),
        "<h2>RQ3 and RQ4: matched comparisons</h2>", _html_table(advanced_comparisons),
        "<p>Differences are condition minus the explicit reference on identical test record IDs. Confidence intervals resample whole group IDs, preserving paired model responses and linked prompts. Each interval conditions on one fitted seed; seed variability is separately exported. Mismatched test sets are flagged rather than silently intersected.</p>",
        "<p class='placeholder'>STUDENT: For RQ3, compare target-heldout training with the matched_mixed_domain baseline on identical target test records. State how validation data were restricted. For RQ4, compare raw versus formatting and fixed-token-length ablations on identical eligible samples. Select the token length using training data only. Truncating to a fixed token count also removes semantic content, so a decline cannot be attributed exclusively to response-length cues. A model/provider keyword mask does not imply every matched keyword was a genuine self-identification.</p>",
        "<h2>Training-only linguistic analysis and feature baseline</h2>",
        "<p>The tables folder contains training-only full-response length, sentence, layout and lexical-diversity descriptions. Fixed-window lexical diversity compares the same prompts for which every model has at least 128 words. Separately, the feature classifier computes its predictors only from the same original output-token prefix available to the neural baseline; it does not observe full-response lengths beyond that prefix. It uses training-fitted imputation/scaling and fixed-C logistic regression. Its coefficients are predictive associations, not causal effects or universal model fingerprints.</p>",
        "<h3>Training response features</h3>", _html_table(description),
        "<h3>Paired fixed-window lexical diversity</h3>", _html_table(lexical),
        "<h3>Feature baseline: largest absolute coefficients per model</h3>", _html_table(coefficients),
        "<p>Coefficients are per training-set standard deviation after imputation. Their signs are relative predictive associations in a joint multiclass model, and correlated features can change the coefficients.</p>",
        "<h2>Limitations and lessons</h2>",
        "<ul><li>Attribution is conditional on the selected model versions, prompts, generation parameters and collection pipeline.</li><li>Accuracy can reflect length, formatting, refusals, benchmark templates or generation settings in addition to linguistic style.</li><li>Approximate sentence/word rules work differently for code, mathematics and natural language.</li><li>Do not tune settings or choose successful ablations based on test performance. A new exploratory hypothesis requires an explicitly labeled analysis or fresh evaluation data.</li><li>Inference applies to this sampled prompt population; five model versions do not represent every model family or future release.</li></ul>",
        "<p class='placeholder'>STUDENT: Add concrete error examples, findings that surprised you, changes made during development, compute limits, and the assistance you used. Explain every submitted method and result in your own words.</p>",
        "<h2>Figures</h2>"]
    for filename, caption in figure_entries:
        encoded = base64.b64encode((figures / filename).read_bytes()).decode("ascii")
        sections.append(f"<div class='figure'><img src='data:image/png;base64,{encoded}' alt='{esc(caption)}'><p>{esc(caption)}</p></div>")
    for filename in ["training_response_lengths.png", "training_paired_lexical_diversity.png"]:
        locations = [root / "descriptive" / "figures" / filename, figures / filename]
        source = next((location for location in locations if location.exists()), None)
        if source is not None:
            encoded = base64.b64encode(source.read_bytes()).decode("ascii")
            sections.append(f"<div class='figure'><img src='data:image/png;base64,{encoded}' alt='{esc(filename)}'><p>Training data only: {esc(filename)}</p></div>")
    sections += ["<h2>Configuration</h2>", f"<pre>{esc(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))}</pre>",
                 "<p>For a PDF draft, open this HTML file locally in a browser and print to PDF. This is a draft export; verify tables and replace all STUDENT placeholders before submission.</p>", "</body></html>"]
    (root / "report_draft.html").write_text("\n".join(sections), encoding="utf-8")
    _export_summary_pdf(root, cfg, main_summary, summary, comparisons, stage_status, figure_entries)
    _json_write(root / "report_status.json", {"profile": profile, "smoke_only": is_smoke, "completed_runs": len(metrics_frame),
                 "research_questions": stage_status, "has_student_placeholders": True,
                 "report_status": "partial_or_draft_not_a_final_submission"})
