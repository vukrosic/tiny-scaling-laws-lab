#!/usr/bin/env python3
"""Run tiny, controlled capacity sweeps for pretraining and RL post-training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_WIDTHS = (4, 8, 16, 24)
DEFAULT_SEEDS = (7, 19, 31)
CONTEXT_LENGTH = 24
VOCABULARY = "\n .,:;!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-="
STOI = {character: index for index, character in enumerate(VOCABULARY)}
RL_SYMBOLS = "abcdefgh"
RL_ACTION_IDS = torch.tensor([STOI[symbol] for symbol in RL_SYMBOLS], dtype=torch.long)


@dataclass
class RunResult:
    seed: int
    width: int
    parameters: int
    pretrain_validation_loss: float
    rl_exact_before: float
    rl_exact_after: float
    rl_reward_before: float
    rl_reward_after: float
    rl_reward_gap_after: float
    pretrain_seconds: float
    rl_seconds: float


@dataclass
class SummaryResult:
    width: int
    parameters: int
    pretrain_validation_loss: float
    pretrain_validation_loss_std: float
    rl_exact_before: float
    rl_exact_after: float
    rl_exact_after_std: float
    rl_reward_before: float
    rl_reward_after: float
    rl_reward_after_std: float
    rl_reward_gap_after: float
    total_training_seconds: float


class TinyTransformer(nn.Module):
    """One-block, one-head, decoder-only character transformer."""

    def __init__(self, width: int, context_length: int = CONTEXT_LENGTH) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(len(VOCABULARY), width)
        self.position_embedding = nn.Embedding(context_length, width)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.final_norm = nn.LayerNorm(width)
        self.language_head = nn.Linear(width, len(VOCABULARY), bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, length = token_ids.shape
        if length > self.context_length:
            raise ValueError(f"sequence length {length} exceeds {self.context_length}")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        normalized = self.norm1(hidden)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attended = F.softmax(scores, dim=-1) @ value
        hidden = hidden + self.attention_output(attended)
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return self.language_head(self.final_norm(hidden))


def encode(text: str) -> torch.Tensor:
    return torch.tensor([STOI[character] for character in text], dtype=torch.long)


def make_language_corpus(seed: int, lines: int) -> str:
    """Create an unlimited deterministic mini-language without a download."""
    generator = random.Random(seed)
    names = ("Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hana")
    colors = ("red", "blue", "green", "gold", "black", "white")
    nouns = ("fox", "owl", "robot", "comet", "boat", "tree", "moon", "book")
    verbs = ("sees", "finds", "follows", "helps", "builds", "moves")
    places = ("lake", "hill", "lab", "road", "field", "cave")
    templates = (
        "{name} {verb} the {color} {noun} near the {place}.\n",
        "At the {place}, {name} {verb} a {color} {noun}.\n",
        "The {color} {noun} {verb} {name} at the {place}.\n",
    )
    output: list[str] = []
    for _ in range(lines):
        output.append(
            generator.choice(templates).format(
                name=generator.choice(names),
                color=generator.choice(colors),
                noun=generator.choice(nouns),
                verb=generator.choice(verbs),
                place=generator.choice(places),
            )
        )
    return "".join(output)


def fixed_batches(data: torch.Tensor, *, steps: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(
        0,
        len(data) - CONTEXT_LENGTH - 1,
        (steps, batch_size),
        generator=generator,
    )
    offsets = torch.arange(CONTEXT_LENGTH + 1)
    return [data[batch_starts[:, None] + offsets] for batch_starts in starts]


def validation_loss(model: TinyTransformer, data: torch.Tensor, windows: int = 96) -> float:
    starts = torch.linspace(0, len(data) - CONTEXT_LENGTH - 1, windows).long()
    offsets = torch.arange(CONTEXT_LENGTH + 1)
    samples = data[starts[:, None] + offsets]
    inputs, targets = samples[:, :-1], samples[:, 1:]
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return float(loss)


def train_language_model(
    width: int,
    batches: Iterable[torch.Tensor],
    validation_data: torch.Tensor,
    *,
    seed: int,
    learning_rate: float,
) -> tuple[TinyTransformer, float, float]:
    torch.manual_seed(seed)
    model = TinyTransformer(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    model.train()
    for sample in batches:
        inputs, targets = sample[:, :-1], sample[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    return model, validation_loss(model, validation_data), elapsed


def copy_prompt_split(
    seed: int, training_count: int, evaluation_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make disjoint train/evaluation prompts for the copy-reward task."""
    generator = random.Random(seed)
    examples: list[tuple[str, int]] = []
    seen: set[str] = set()
    while len(examples) < training_count + evaluation_count:
        symbols = "".join(generator.choice(RL_SYMBOLS) for _ in range(6))
        if symbols in seen:
            continue
        seen.add(symbols)
        examples.append((f"Copy:{symbols}=", RL_SYMBOLS.index(symbols[0])))

    def tensors(items: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.stack([encode(text) for text, _ in items]),
            torch.tensor([answer for _, answer in items]),
        )

    training = tensors(examples[:training_count])
    evaluation = tensors(examples[training_count:])
    return *training, *evaluation


def reward_metrics(
    model: TinyTransformer, prompts: torch.Tensor, answers: torch.Tensor
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        action_logits = model(prompts)[:, -1, RL_ACTION_IDS]
        probabilities = F.softmax(action_logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        exact = float((predictions == answers).float().mean())
        expected_reward = float(probabilities.gather(1, answers[:, None]).mean())
    return exact, expected_reward


def rl_post_train(
    pretrained_model: TinyTransformer,
    training_prompts: torch.Tensor,
    training_answers: torch.Tensor,
    evaluation_prompts: torch.Tensor,
    evaluation_answers: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> tuple[float, float, float, float, float]:
    """Optimize exact expected reward by enumerating the eight possible actions."""
    model = copy.deepcopy(pretrained_model)
    # Use the same uniform policy initialization at every width. The pretrained
    # transformer body is retained; only the eight constrained action rows are
    # reset so an unrelated language-model prior cannot dominate this sweep.
    with torch.no_grad():
        model.language_head.weight[RL_ACTION_IDS].zero_()
    exact_before, reward_before = reward_metrics(model, evaluation_prompts, evaluation_answers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    prompt_indices = torch.randint(0, len(training_prompts), (steps, batch_size), generator=generator)
    started = time.perf_counter()

    model.train()
    for indices in prompt_indices:
        selected_prompts = training_prompts[indices]
        selected_answers = training_answers[indices]
        action_logits = model(selected_prompts)[:, -1, RL_ACTION_IDS]
        log_probabilities = F.log_softmax(action_logits, dim=-1)
        probabilities = log_probabilities.exp()
        rewards = F.one_hot(selected_answers, num_classes=len(RL_SYMBOLS)).float()
        expected_reward = (probabilities * rewards).sum(dim=-1).mean()
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
        loss = -expected_reward - 0.02 * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - started
    exact_after, reward_after = reward_metrics(model, evaluation_prompts, evaluation_answers)
    return exact_before, exact_after, reward_before, reward_after, elapsed


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parse_widths(raw: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from error
    if len(widths) < 3:
        raise argparse.ArgumentTypeError("use at least three widths for a scaling curve")
    if len(set(widths)) != len(widths) or any(width < 4 or width > 128 for width in widths):
        raise argparse.ArgumentTypeError("widths must be unique integers from 4 to 128")
    return tuple(sorted(widths))


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide one or more unique integer seeds")
    return seeds


def descriptive_power_slope(parameters: list[int], metrics: list[float]) -> tuple[float, float]:
    x = np.log(np.asarray(parameters, dtype=float))
    y = np.log(np.clip(np.asarray(metrics, dtype=float), 1e-6, None))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    return float(slope), r_squared


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, *, size: int, fill: str, bold: bool = False, anchor: str = "la") -> None:
    draw.text(xy, value, font=_font(size, bold), fill=fill, anchor=anchor)


def _plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    parameters: list[int],
    values: list[float],
    standard_deviations: list[float],
    *,
    title: str,
    subtitle: str,
    slope: float,
    y_label: str,
    color: str,
    value_format: str,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 105, top + 125, right - 35, bottom - 90
    _text(draw, (left + 8, top + 8), title, size=30, fill="#111827", bold=True)
    _text(draw, (left + 8, top + 49), subtitle, size=18, fill="#64748b")
    _text(
        draw,
        (left + 8, top + 76),
        f"descriptive log-log slope: {slope:+.2f}",
        size=16,
        fill="#64748b",
    )
    _text(draw, (plot_left, plot_top - 12), y_label, size=17, fill="#334155", bold=True, anchor="ls")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#94a3b8", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#94a3b8", width=2)

    log_x = np.log10(np.asarray(parameters, dtype=float))
    x_min, x_max = float(log_x.min()), float(log_x.max())
    y_min = min(values)
    y_max = max(values)
    padding = max((y_max - y_min) * 0.18, 0.015)
    lower = max(0.0, y_min - padding)
    upper = y_max + padding

    def x_pixel(value: float) -> float:
        fraction = (math.log10(value) - x_min) / max(x_max - x_min, 1e-9)
        return plot_left + fraction * (plot_right - plot_left)

    def y_pixel(value: float) -> float:
        fraction = (value - lower) / max(upper - lower, 1e-9)
        return plot_bottom - fraction * (plot_bottom - plot_top)

    for tick in np.linspace(lower, upper, 5):
        y = y_pixel(float(tick))
        draw.line((plot_left, y, plot_right, y), fill="#e2e8f0", width=1)
        _text(draw, (plot_left - 14, y), f"{tick:.2f}", size=17, fill="#475569", anchor="ra")

    points = [(x_pixel(parameter), y_pixel(value)) for parameter, value in zip(parameters, values)]
    for (x, _), value, standard_deviation in zip(points, values, standard_deviations):
        error_top = y_pixel(min(upper, value + standard_deviation))
        error_bottom = y_pixel(max(lower, value - standard_deviation))
        # White halo plus wide caps keeps uncertainty visible over grid and curve.
        draw.line((x, error_top, x, error_bottom), fill="#ffffff", width=11)
        draw.line((x - 16, error_top, x + 16, error_top), fill="#ffffff", width=11)
        draw.line((x - 16, error_bottom, x + 16, error_bottom), fill="#ffffff", width=11)
        draw.line((x, error_top, x, error_bottom), fill=color, width=6)
        draw.line((x - 14, error_top, x + 14, error_top), fill=color, width=6)
        draw.line((x - 14, error_bottom, x + 14, error_bottom), fill=color, width=6)

    draw.line(points, fill=color, width=5, joint="curve")
    for (x, y), parameter, value in zip(points, parameters, values):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#ffffff", outline=color, width=5)
        _text(draw, (x, y - 20), value_format.format(value), size=17, fill="#0f172a", bold=True, anchor="ms")
        _text(draw, (x, plot_bottom + 15), f"{parameter:,}", size=16, fill="#475569", anchor="ma")
    _text(draw, ((plot_left + plot_right) / 2, bottom - 38), "model parameters (log scale)", size=18, fill="#334155", anchor="mm")


def save_chart(results: list[SummaryResult], image_path: Path, pretrain_slope: float, rl_slope: float, seed_count: int) -> None:
    canvas = Image.new("RGB", (1600, 940), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    _text(draw, (70, 55), "Tiny scaling curves: pretraining + RL", size=48, fill="#0f172a", bold=True)
    _text(
        draw,
        (70, 120),
        "One character-level Transformer architecture. Only width changes; budgets and evaluation stay fixed.",
        size=23,
        fill="#475569",
    )
    parameters = [result.parameters for result in results]
    pretraining = [result.pretrain_validation_loss for result in results]
    pretraining_std = [result.pretrain_validation_loss_std for result in results]
    rl_errors = [result.rl_reward_gap_after for result in results]
    rl_error_std = [result.rl_reward_after_std for result in results]
    _plot_panel(
        draw,
        (55, 185, 800, 790),
        parameters,
        pretraining,
        pretraining_std,
        title="1. Pretraining capacity sweep",
        subtitle=f"mean ± 1 SD over {seed_count} seeds",
        slope=pretrain_slope,
        y_label="validation cross-entropy (nats/character) ↓",
        color="#2563eb",
        value_format="{:.3f}",
    )
    _plot_panel(
        draw,
        (800, 185, 1545, 790),
        parameters,
        rl_errors,
        rl_error_std,
        title="2. RL post-training capacity sweep",
        subtitle=f"mean ± 1 SD over {seed_count} seeds",
        slope=rl_slope,
        y_label="reward gap (1 - mean reward) ↓",
        color="#dc2626",
        value_format="{:.0%}",
    )
    draw.rounded_rectangle((70, 825, 1530, 900), radius=18, fill="#e2e8f0")
    _text(
        draw,
        (800, 862),
        "Vertical bars = ±1 SD across seeds. Evidence for this setup, not a universal scaling law.",
        size=23,
        fill="#334155",
        bold=True,
        anchor="mm",
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path)


def open_image(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, FileNotFoundError):
        print(f"Open this image manually: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=parse_widths, default=DEFAULT_WIDTHS)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--pretrain-steps", type=int, default=30)
    parser.add_argument("--rl-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-learning-rate", type=float, default=0.008)
    parser.add_argument("--rl-learning-rate", type=float, default=0.008)
    parser.add_argument("--image", type=Path, default=ROOT / "my_scaling_laws.png")
    parser.add_argument("--receipt", type=Path, default=ROOT / "my_scaling_laws.json")
    parser.add_argument("--no-open", action="store_true")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    for name in ("pretrain_steps", "rl_steps", "batch_size"):
        if getattr(arguments, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("pretrain_learning_rate", "rl_learning_rate"):
        value = getattr(arguments, name)
        if not 0 < value <= 0.2:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 0.2]")


def main() -> int:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    training_data = encode(make_language_corpus(2026, 900))
    validation_data = encode(make_language_corpus(2027, 180))
    training_prompts, training_answers, evaluation_prompts, evaluation_answers = copy_prompt_split(
        2028, 512, 256
    )

    runs: list[RunResult] = []
    print("\nTiny capacity sweep (lower loss/error is better)")
    print(f"Replicate seeds: {', '.join(map(str, arguments.seeds))}\n")
    for seed in arguments.seeds:
        batches = fixed_batches(
            training_data,
            steps=arguments.pretrain_steps,
            batch_size=arguments.batch_size,
            seed=seed,
        )
        for width in arguments.widths:
            model, pretrain_loss, pretrain_seconds = train_language_model(
                width,
                batches,
                validation_data,
                seed=seed,
                learning_rate=arguments.pretrain_learning_rate,
            )
            exact_before, exact_after, reward_before, reward_after, rl_seconds = rl_post_train(
                model,
                training_prompts,
                training_answers,
                evaluation_prompts,
                evaluation_answers,
                steps=arguments.rl_steps,
                batch_size=arguments.batch_size,
                seed=seed + 1000,
                learning_rate=arguments.rl_learning_rate,
            )
            runs.append(
                RunResult(
                    seed=seed,
                    width=width,
                    parameters=parameter_count(model),
                    pretrain_validation_loss=pretrain_loss,
                    rl_exact_before=exact_before,
                    rl_exact_after=exact_after,
                    rl_reward_before=reward_before,
                    rl_reward_after=reward_after,
                    rl_reward_gap_after=1.0 - reward_after,
                    pretrain_seconds=pretrain_seconds,
                    rl_seconds=rl_seconds,
                )
            )

    results: list[SummaryResult] = []
    print(f"{'width':>6} {'parameters':>11} {'pretrain loss':>15} {'RL reward':>16}")
    print("-" * 54)
    for width in arguments.widths:
        selected = [run for run in runs if run.width == width]
        pretraining = np.asarray([run.pretrain_validation_loss for run in selected])
        rl_before = np.asarray([run.rl_exact_before for run in selected])
        rl_after = np.asarray([run.rl_exact_after for run in selected])
        reward_before = np.asarray([run.rl_reward_before for run in selected])
        reward_after = np.asarray([run.rl_reward_after for run in selected])
        result = SummaryResult(
            width=width,
            parameters=selected[0].parameters,
            pretrain_validation_loss=float(pretraining.mean()),
            pretrain_validation_loss_std=float(pretraining.std()),
            rl_exact_before=float(rl_before.mean()),
            rl_exact_after=float(rl_after.mean()),
            rl_exact_after_std=float(rl_after.std()),
            rl_reward_before=float(reward_before.mean()),
            rl_reward_after=float(reward_after.mean()),
            rl_reward_after_std=float(reward_after.std()),
            rl_reward_gap_after=1.0 - float(reward_after.mean()),
            total_training_seconds=sum(run.pretrain_seconds + run.rl_seconds for run in selected),
        )
        results.append(result)
        print(
            f"{width:>6} {result.parameters:>11,} "
            f"{result.pretrain_validation_loss:>9.3f} +/- {result.pretrain_validation_loss_std:<5.3f} "
            f"{result.rl_reward_after:>7.0%} +/- {result.rl_reward_after_std:<5.0%}"
        )

    parameters = [result.parameters for result in results]
    pretrain_metrics = [result.pretrain_validation_loss for result in results]
    rl_metrics = [max(result.rl_reward_gap_after, 1e-6) for result in results]
    pretrain_slope, pretrain_r_squared = descriptive_power_slope(parameters, pretrain_metrics)
    rl_slope, rl_r_squared = descriptive_power_slope(parameters, rl_metrics)
    elapsed = time.perf_counter() - started

    image_path = arguments.image.resolve()
    receipt_path = arguments.receipt.resolve()
    save_chart(results, image_path, pretrain_slope, rl_slope, len(arguments.seeds))
    receipt = {
        "claim_scope": "small multi-seed educational capacity sweep; not a universal scaling law",
        "controlled_variable": "transformer residual width",
        "fixed_controls": {
            "architecture": "one-block, one-head, decoder-only character transformer",
            "vocabulary_size": len(VOCABULARY),
            "context_length": CONTEXT_LENGTH,
            "pretrain_steps": arguments.pretrain_steps,
            "rl_steps": arguments.rl_steps,
            "batch_size": arguments.batch_size,
            "pretrain_learning_rate": arguments.pretrain_learning_rate,
            "rl_learning_rate": arguments.rl_learning_rate,
            "data_seeds": {"pretraining_train": 2026, "pretraining_validation": 2027, "rl_split": 2028},
            "replicate_seeds": list(arguments.seeds),
        },
        "pretraining_metric": "held-out next-character cross-entropy in nats",
        "rl_metric": "reward gap (1 minus probability of sampling the rewarded character) on 256 frozen held-out copy prompts",
        "rl_algorithm": "exact expected-return policy gradient with entropy bonus and an eight-character constrained action space",
        "descriptive_log_log_fits": {
            "pretraining": {"slope": pretrain_slope, "r_squared": pretrain_r_squared},
            "rl_reward_gap": {"slope": rl_slope, "r_squared": rl_r_squared},
        },
        "summary_results": [asdict(result) for result in results],
        "per_seed_results": [asdict(run) for run in runs],
        "total_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(f"\nFinished in {elapsed:.2f} seconds")
    print(f"Image:   {image_path}")
    print(f"Receipt: {receipt_path}")
    if not arguments.no_open:
        open_image(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
