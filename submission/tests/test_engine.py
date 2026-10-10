"""Meaningful correctness checks for padding, evaluation, leakage controls and resume."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from attribution import engine
from attribution.data import pad_batch


FAMILIES = ["GPT", "Claude", "Gemini", "Qwen", "DeepSeek"]


def tiny_frame(split: str, prompts: int = 3) -> pd.DataFrame:
    records = []
    for prompt in range(prompts):
        for label, family in enumerate(FAMILIES):
            records.append({"record_id": f"{split}_{prompt}_{label}",
                "prompt_id": f"{split}_{prompt}", "group_id": f"{split}_group_{prompt}",
                "label": label, "LLM_name": family, "model_id": family,
                "LLM_Input": f"Question in {split}: explain item {prompt}.",
                "LLM_output": f"{family} style answer. Item {prompt} has a small result.",
                "task_domain": "reasoning", "benchmark": "synthetic_test_fixture", "split": split})
    return pd.DataFrame(records)


def tiny_config() -> dict:
    return {"vocabulary_size": 200, "embedding_dim": 8, "cnn_filters": 4,
            "cnn_kernels": [2, 3, 5], "lstm_hidden": 6, "dropout": 0.2,
            "batch_size": 7, "epochs": 2, "patience": 3, "learning_rate": 0.01,
            "input_max_tokens": 20, "output_max_tokens": 24, "device": "cpu",
            "num_workers": 0, "threads": 1, "bootstrap_reps": 10, "deterministic": True}


class EngineChecks(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_cnn_ignores_extra_batch_padding_even_for_short_sequences(self):
        model = engine.TextCNN(30, embedding_dim=8, filters=4, kernels=[3, 4, 5], dropout=0).eval()
        short = torch.tensor([[4, 6]])
        long_batch = torch.tensor([[4, 6, 0, 0, 0, 0, 0, 0], [2, 3, 4, 5, 6, 7, 8, 9]])
        with torch.no_grad():
            standalone = model(short, torch.tensor([2]))[0]
            padded = model(long_batch, torch.tensor([2, 8]))[0]
        torch.testing.assert_close(standalone, padded, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(standalone).all())

    def test_bilstm_ignores_extra_batch_padding(self):
        model = engine.BiLSTM(30, embedding_dim=8, hidden_dim=6, dropout=0).eval()
        short = torch.tensor([[4]])
        long_batch = torch.tensor([[4, 0, 0, 0, 0, 0], [2, 3, 4, 5, 6, 7]])
        with torch.no_grad():
            standalone = model(short, torch.tensor([1]))[0]
            padded = model(long_batch, torch.tensor([1, 6]))[0]
        torch.testing.assert_close(standalone, padded, atol=1e-6, rtol=1e-6)

    def test_evaluation_loss_is_weighted_by_rows(self):
        class FixedLogits(nn.Module):
            def forward(self, ids, lengths):
                logits = torch.zeros((len(ids), 5))
                logits[:, 0] = ids[:, 0].float()
                return logits
        rows = [([1], 0, 0), ([2], 1, 1), ([3], 2, 2), ([4], 3, 3), ([5], 4, 4)]
        loader = DataLoader(rows, batch_size=3, collate_fn=pad_batch)
        result = engine.evaluate(FixedLogits(), loader, "cpu")
        ids, lengths, labels, _ = pad_batch(rows)
        expected = nn.functional.cross_entropy(FixedLogits()(ids, lengths), labels).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(result["row_indices"].tolist(), list(range(5)))

    def test_group_bootstrap_preserves_paired_chance_control(self):
        labels = np.tile(np.arange(5), 10)
        predicted = np.repeat(np.arange(10) % 5, 5)
        groups = np.repeat(np.arange(10), 5)
        result = engine.grouped_bootstrap(labels, predicted, groups, repetitions=40)
        self.assertAlmostEqual(result["accuracy_ci_low"], 0.2)
        self.assertAlmostEqual(result["accuracy_ci_high"], 0.2)

    def test_real_training_resume_and_stale_data_guard(self):
        frames = [tiny_frame(name) for name in ("train", "validation", "test")]
        config = tiny_config()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input_control"
            result = engine.run_experiment(*frames, config, path, mode="input_only")
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["accuracy"], 0.2)
            reused = engine.run_experiment(*frames, config, path, mode="input_only")
            self.assertEqual(reused["signature"], result["signature"])
            edited = frames[0].copy()
            edited.loc[0, "LLM_output"] += " changed"
            with self.assertRaisesRegex(ValueError, "different data"):
                engine.run_experiment(edited, frames[1], frames[2], config, path, mode="input_only")
            predictions = pd.read_csv(path / "test_predictions.csv")
            self.assertEqual(len(predictions), len(frames[2]))
            self.assertEqual(predictions["record_id"].tolist(), frames[2]["record_id"].tolist())
            self.assertTrue(np.allclose(predictions[[f"prob_{i}" for i in range(5)]].sum(axis=1), 1))

    def test_interrupted_epoch_resume_matches_uninterrupted_training(self):
        frames = [tiny_frame(name) for name in ("train", "validation", "test")]
        config = tiny_config()
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full"
            interrupted = Path(folder) / "interrupted"
            engine.run_experiment(*frames, config, full, architecture="bilstm", mode="output_only", seed=13)
            original_save = engine._atomic_checkpoint

            def interrupt_after_first_epoch(path, value):
                original_save(path, value)
                if path.name == "last.pt" and value["epoch"] == 1:
                    raise KeyboardInterrupt("Simulated interruption after an atomic epoch checkpoint")

            with patch.object(engine, "_atomic_checkpoint", side_effect=interrupt_after_first_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    engine.run_experiment(*frames, config, interrupted, architecture="bilstm", mode="output_only", seed=13)
            engine.run_experiment(*frames, config, interrupted, architecture="bilstm", mode="output_only", seed=13)
            uninterrupted = torch.load(full / "last.pt", weights_only=False, map_location="cpu")
            resumed = torch.load(interrupted / "last.pt", weights_only=False, map_location="cpu")
            for name, value in uninterrupted["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], atol=0, rtol=0)
            pd.testing.assert_frame_equal(pd.read_csv(full / "test_predictions.csv"),
                                          pd.read_csv(interrupted / "test_predictions.csv"))


if __name__ == "__main__":
    unittest.main()
