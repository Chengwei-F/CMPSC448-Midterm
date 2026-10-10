"""Recompute saved test Accuracy/Macro-F1 using the Python standard library.

No training, external data, network connection, or machine-learning package is
required. This verifies the numerical summaries against their saved predictions;
it does not prove that upstream data collection/training was correct.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LABELS = ["GPT", "Claude", "Gemini", "Qwen", "DeepSeek"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_run(root: Path, entry: dict) -> dict:
    directory = root / entry["run_dir"]
    metrics = load_json(directory / "metrics.json")
    with (directory / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No predictions: {directory}")
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    records = set()
    prompt_labels: dict[str, Counter] = {}
    prompt_predictions: dict[str, set] = {}
    for row in rows:
        identifier = row["record_id"]
        if identifier in records:
            raise ValueError(f"Duplicate record_id in {directory}")
        records.add(identifier)
        truth = LABELS.index(row["true_label"])
        predicted = LABELS.index(row["pred_label"])
        matrix[truth][predicted] += 1
        probabilities = [float(row[f"prob_{i}"]) for i in range(5)]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError(f"Invalid probabilities in {directory}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError(f"Probabilities do not sum to one in {directory}")
        # Tolerate tiny differences introduced by CSV decimal serialization.
        if probabilities[predicted] + 1e-7 < max(probabilities):
            raise ValueError(f"Predicted label disagrees with probabilities in {directory}")
        prompt_labels.setdefault(row["prompt_id"], Counter())[row["true_label"]] += 1
        prompt_predictions.setdefault(row["prompt_id"], set()).add(row["pred_label"])
    if any(counts != Counter(LABELS) for counts in prompt_labels.values()):
        raise ValueError(f"Incomplete five-model prompt group in {directory}")
    accuracy = sum(matrix[i][i] for i in range(5)) / len(rows)
    f1_values = []
    for label in range(5):
        actual = sum(matrix[label])
        predicted_count = sum(matrix[i][label] for i in range(5))
        denominator = actual + predicted_count
        f1_values.append(2 * matrix[label][label] / denominator if denominator else 0.0)
    macro_f1 = sum(f1_values) / 5
    if int(metrics["n_test"]) != len(rows):
        raise ValueError(f"n_test mismatch in {directory}")
    for name, value in (("accuracy", accuracy), ("macro_f1", macro_f1)):
        if not math.isclose(float(metrics[name]), value, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{name} mismatch in {directory}: saved={metrics[name]}, recomputed={value}")
    if entry.get("mode") == "input_only":
        if not math.isclose(accuracy, 0.2, abs_tol=1e-12):
            raise ValueError(f"Input-only negative control differs from 20% in {directory}")
        if any(len(values) != 1 for values in prompt_predictions.values()):
            raise ValueError(f"Identical prompts have different input-only predictions in {directory}")
    return {"run_id": entry["run_id"], "n_test": len(rows),
            "accuracy": accuracy, "macro_f1": macro_f1, "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results/full")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    index = load_json(args.results / "experiment_index.json")
    names = [entry["run_id"] for entry in index]
    if len(names) != len(set(names)):
        raise ValueError("The experiment index repeats run IDs")
    results = [verify_run(args.results, entry) for entry in index]
    payload = {"all_passed": True, "verified_runs": len(results),
               "verification": "Accuracy/Macro-F1 recomputed from saved test predictions; complete prompt groups, probability validity, row counts and input-only controls checked",
               "training_recomputed": False, "runs": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PASS: {len(results)} saved runs verified; no training or external data used.")


if __name__ == "__main__":
    main()
