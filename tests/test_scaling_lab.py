from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import scaling_lab


ROOT = Path(__file__).resolve().parents[1]


class ScalingLabTests(unittest.TestCase):
    def test_parameter_count_increases_with_width(self) -> None:
        counts = [scaling_lab.parameter_count(scaling_lab.TinyTransformer(width)) for width in (4, 8, 16)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(counts), len(set(counts)))

    def test_rl_prompt_splits_are_disjoint_and_targets_copy_first_symbol(self) -> None:
        train_prompts, train_answers, eval_prompts, eval_answers = scaling_lab.copy_prompt_split(99, 40, 20)
        train_rows = {tuple(row.tolist()) for row in train_prompts}
        eval_rows = {tuple(row.tolist()) for row in eval_prompts}
        self.assertTrue(train_rows.isdisjoint(eval_rows))
        for prompts, answers in ((train_prompts, train_answers), (eval_prompts, eval_answers)):
            for prompt, answer in zip(prompts, answers):
                text = "".join(scaling_lab.VOCABULARY[token] for token in prompt.tolist())
                self.assertEqual(scaling_lab.RL_SYMBOLS[int(answer)], text[len("Copy:")])

    def test_attention_is_causal(self) -> None:
        torch.manual_seed(1)
        model = scaling_lab.TinyTransformer(8).eval()
        first = scaling_lab.encode("abcdefgh")[None, :]
        second = first.clone()
        second[0, -1] = scaling_lab.STOI["z"]
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        self.assertTrue(torch.allclose(first_logits[:, :-1], second_logits[:, :-1], atol=1e-6))

    def test_smoke_run_writes_image_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "smoke.png"
            receipt = Path(directory) / "smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scaling_lab.py"),
                    "--widths",
                    "4,6,8",
                    "--seeds",
                    "1",
                    "--pretrain-steps",
                    "2",
                    "--rl-steps",
                    "2",
                    "--batch-size",
                    "4",
                    "--image",
                    str(image),
                    "--receipt",
                    str(receipt),
                    "--no-open",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertGreater(image.stat().st_size, 10_000)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["summary_results"]), 3)
            self.assertEqual(payload["fixed_controls"]["replicate_seeds"], [1])


if __name__ == "__main__":
    unittest.main()
