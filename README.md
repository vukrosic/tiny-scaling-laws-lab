# Tiny LLM Scaling Laws Lab

Train four character-level transformers and produce two small scaling curves:

1. **Pretraining:** held-out next-character cross-entropy versus parameter count.
2. **RL post-training:** held-out reward gap versus parameter count.

The complete experiment runs on CPU and opens one PNG. No model is downloaded.
The first run installs CPU PyTorch, NumPy, and Pillow; the experiment itself took
about four seconds on the development Mac.

## Level 1 — Reproduce the result

macOS or Linux:

```bash
git clone https://github.com/vukrosic/tiny-scaling-laws-lab.git && cd tiny-scaling-laws-lab && ./first_win.sh
```

Windows PowerShell:

```powershell
git clone https://github.com/vukrosic/tiny-scaling-laws-lab.git; cd tiny-scaling-laws-lab; .\first_win.bat
```

The command trains 844-, 1,944-, 4,912-, and 8,904-parameter models over three
random seeds. It saves and opens `my_scaling_laws.png` and writes the complete
measurements to `my_scaling_laws.json`.

One seeded macOS reference run:

| Parameters | Pretraining loss ↓ | RL mean reward ↑ |
|---:|---:|---:|
| 844 | 3.386 | 0.142 |
| 1,944 | 3.012 | 0.391 |
| 4,912 | 2.580 | 0.879 |
| 8,904 | 2.197 | 0.929 |

Your prediction before running:

```text
I predict increasing model width will ______ because ______.
```

## What is trained?

Every model is a one-block, one-head, decoder-only transformer. It reads and
generates individual characters, not words or subword tokens.

Pretraining examples come from a deterministic synthetic mini-language:

```text
Ada follows the blue owl near the lake.
At the lab, Bo builds a green robot.
```

The model receives 24 characters and predicts the next character. The primary
metric is cross-entropy on a separately generated validation corpus, measured
in nats per character. Lower is better.

RL uses a one-step copy task:

```text
prompt: Copy:facbed=
action: one character from a, b, c, d, e, f, g, h
reward: 1 for f; 0 for every other character
```

There are 512 unique RL training prompts and 256 disjoint evaluation prompts.
The pretrained transformer body is retained, while the eight action rows start
from the same uniform policy at every width. Because there are only eight
actions, the code evaluates every reward and differentiates the exact expected
return:

```text
J(theta) = mean over prompts of sum_a pi_theta(a | prompt) * reward(prompt, a)
```

### What rule does RL learn?

The rule is: **output the first character after `Copy:`**. The reward function,
not the model, knows that `f` is correct for `Copy:facbed=` because `f` is the
first character after the colon. The model is not given this rule in words.

Its action probabilities begin uniformly at about `1/8` each. For this prompt,
the expected reward is exactly `P(f | Copy:facbed=)`, so gradient updates
increase the probability of `f`. Across many prompts with different first
characters, the model learns the general copying policy. Evaluation then tests
that policy on 256 prompts excluded from RL training.

This is exact policy optimization for a small contextual bandit. It is genuine
reward-based post-training, but it is not RLHF, PPO, or GRPO. The plotted RL
metric is the reward gap, `1 - mean reward`; lower is better.

## Level 2 — Change one number

Replace the largest width, `24`, with `32`:

```bash
./experiment.sh --widths 4,8,16,32
```

Windows uses `experiment.bat` with the same arguments. Predict whether both
curves will continue improving. Keep every other setting fixed. A larger model
can improve pretraining loss while becoming less reliable under a fixed RL
optimization budget; that is a valid negative result.

## Level 3 — Test the explanation

Test whether additional RL optimization reduces the larger model's reward gap:

```bash
./experiment.sh --widths 4,8,16,24,32 --rl-steps 120 \
  --image results/more_rl.png --receipt results/more_rl.json
```

Write down:

```text
Hypothesis:
Independent variable:
Fixed controls:
Primary metric:
Result:
What the result does not establish:
```

Do not compare the two runs as a pure model-size experiment: Level 3 changes
the RL budget. It tests a possible optimization-budget explanation.

## What is and is not a scaling law here

Only Transformer residual width changes in the capacity sweep. Every width uses
the same one-block, one-head architecture family, training and validation data,
context length, sampled batches within each seed, 30 pretraining updates, 50 RL
updates, batch size 32, learning rates, evaluator, and random seeds.

This is **update-matched, not compute-matched**. A wider model performs more
operations per update and may take longer, so total computation is not fixed.
The graph shows one-standard-deviation error bars and a descriptive log-log
slope.

This is a **mini empirical capacity sweep**, not a universal neural scaling law.
Four small model sizes and three seeds cannot establish an asymptotic power law.
The runs are also step-matched rather than compute-matched: larger models use
more operations per step. Large-lab scaling studies use many more scales,
larger datasets, tuned training budgets, and uncertainty analysis.

## Useful commands

```bash
./first_win.sh --no-open
./experiment.sh --widths 4,8,16,24
./experiment.sh --seeds 7,19,31,43,55
.venv/bin/python -m unittest discover -s tests -v
```

All generated PNG and JSON files are ignored by Git.

MIT License.
