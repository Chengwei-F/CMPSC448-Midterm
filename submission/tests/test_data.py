"""Focused checks for leakage controls and fair experimental preprocessing."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from attribution.data import (
    EncodedDataset, SPECIAL_TOKENS, audit_splits, build_vocab, output_tokens,
    pad_batch, sample_groups, tokenize,
)


def make_frame(split, count=6):
    rows = []
    names = ["GPT", "Claude", "Gemini", "Qwen", "DeepSeek"]
    for prompt in range(count):
        for label, name in enumerate(names):
            rows.append({
                "record_id": f"{split}-{prompt}-{label}", "prompt_id": f"{split}-{prompt}",
                "group_id": f"{split}-{prompt}", "LLM_name": name, "model_id": f"model-{label}",
                "label": label, "LLM_Input": f"{split} question number {prompt}",
                "LLM_output": "### Hello\nThis is a model answer with extra words and punctuation!",
                "task_domain": "coding" if prompt % 2 else "math", "benchmark": "example", "split": split,
            })
    return pd.DataFrame(rows)


class DataControlsTests(unittest.TestCase):
    def test_unicode_and_newline_token_budget(self):
        self.assertEqual(tokenize("École\n中文 42! next", 5), ["école", "<nl>", "中文", "42", "!"])
        self.assertEqual(tokenize("something", 0), [])

    def test_format_removal_cannot_expose_later_answer(self):
        cfg = {"input_max_tokens": 4, "output_max_tokens": 6}
        text = "### First\nword LATER_SECRET"
        original = output_tokens(text, cfg)
        cleaned = output_tokens(text, cfg, "no_format")
        self.assertEqual(original, ["#", "#", "#", "first", "<nl>", "word"])
        self.assertEqual(cleaned, ["first", "word"])
        self.assertNotIn("later", cleaned)

    def test_input_plus_output_retains_same_answer_span(self):
        frame = make_frame("train", 1)
        cfg = {"input_max_tokens": 3, "output_max_tokens": 7, "vocabulary_size": 100}
        vocab = build_vocab(frame, cfg)
        answer_only = EncodedDataset(frame, vocab, "output_only", cfg)[0][0]
        both = EncodedDataset(frame, vocab, "input_and_output", cfg)[0][0]
        self.assertEqual(both[4:], answer_only)
        self.assertEqual(both[3], SPECIAL_TOKENS["<sep>"])

    def test_fixed_length_means_actual_tokens_and_rejects_short_answers(self):
        cfg = {"input_max_tokens": 10, "output_max_tokens": 8}
        self.assertEqual(len(output_tokens("word!!!\nmore", cfg, "fixed_length", 5)), 5)
        with self.assertRaises(ValueError):
            output_tokens("too short", cfg, "fixed_length", 5)

    def test_vocabulary_truncates_original_training_spans(self):
        frame = make_frame("train", 1)
        frame["LLM_output"] = "normal body zzzhidden"
        cfg = {"input_max_tokens": 2, "output_max_tokens": 2, "vocabulary_size": 100}
        vocab = build_vocab(frame, cfg)
        self.assertNotIn("zzzhidden", vocab)
        self.assertIn("normal", vocab)
        self.assertNotIn("validationexclusive", vocab)

    def test_masking_has_generic_token(self):
        cfg = {"input_max_tokens": 10, "output_max_tokens": 30}
        masked = output_tokens("I am Claude Sonnet 4 by Anthropic.", cfg, "mask_identity")
        self.assertIn("<identity>", masked)
        self.assertNotIn("claude", masked)
        self.assertNotIn("anthropic", masked)

    def test_sampling_retains_connected_groups_and_each_task(self):
        frame = make_frame("train", 8)
        frame.loc[frame.prompt_id.eq("train-1"), "group_id"] = "train-0"
        chosen = sample_groups(frame, 3, 99)
        self.assertEqual(chosen.group_id.nunique(), 3)
        self.assertEqual(set(chosen.task_domain), {"coding", "math"})
        for group in chosen.group_id.unique():
            self.assertEqual(len(chosen[chosen.group_id.eq(group)]), len(frame[frame.group_id.eq(group)]))
        pd.testing.assert_frame_equal(chosen, sample_groups(frame, 3, 99))

    def test_audit_detects_normalized_text_leakage(self):
        splits = {name: make_frame(name, 2) for name in ("train", "validation", "test")}
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(audit_splits(splits, folder)["passed"])
            leaked = "  TRAIN   question number 0 "
            splits["test"].loc[splits["test"].prompt_id.eq("test-0"), "LLM_Input"] = leaked
            with self.assertRaisesRegex(ValueError, "normalized exact input overlap"):
                audit_splits(splits, folder)
            self.assertTrue((Path(folder) / "data_audit.json").is_file())

    def test_collation_preserves_lengths_and_row_indices(self):
        ids, lengths, labels, rows = pad_batch([([4, 5], 1, 9), ([2], 0, 3)])
        self.assertEqual(ids.tolist(), [[4, 5], [2, 0]])
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(rows.tolist(), [9, 3])
        self.assertEqual(ids.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
