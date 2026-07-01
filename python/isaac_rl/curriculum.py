"""Curriculum stages (plan §8).

A stage is a small policy that decides:
  - which `stage N` to `Isaac.ExecuteCommand` on each reset
  - which seed pool to draw from
  - which reward config to use (some stages want more shaping)

The PPO trainer takes one stage at a time via its `reset_stage` config knob.
Advance stages by rerunning training with a new config file that resumes from
the previous stage's checkpoint (see docs/training.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .reward import RewardConfig


@dataclass
class Stage:
    name: str
    reset_stage: int | None       # what Lua receives as `stage N`
    seed_pool: Sequence[int]
    reward_config: RewardConfig
    max_episode_steps: int
    description: str


def stage1_single_room() -> Stage:
    """Basement-1 single-room combat. Teleport into the same handful of rooms."""
    return Stage(
        name="stage1_single_room",
        reset_stage=1,
        seed_pool=(101, 202, 303, 404, 505, 606, 707, 808),
        reward_config=RewardConfig(
            r_room_clear=2.0,   # amplify the training signal for room clears
            r_new_room=0.0,     # not relevant for single-room stage
        ),
        max_episode_steps=1800,   # ~2 min at 15 Hz
        description="Clear a single Basement-1 combat room. Reward on kill + room clear.",
    )


def stage2_floor_clear() -> Stage:
    return Stage(
        name="stage2_floor_clear",
        reset_stage=1,
        seed_pool=(
            10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008,
        ),
        reward_config=RewardConfig(),
        max_episode_steps=4500,   # ~5 min
        description="Clear all of Basement 1 (reach trapdoor into Basement 2).",
    )


def stage3_two_floors() -> Stage:
    return Stage(
        name="stage3_two_floors",
        reset_stage=1,
        seed_pool=tuple(range(100_000, 100_032)),
        reward_config=RewardConfig(),
        max_episode_steps=9000,   # ~10 min
        description="Basement 1 + Basement 2.",
    )


def stage4_full_run() -> Stage:
    return Stage(
        name="stage4_full_run",
        reset_stage=1,
        seed_pool=tuple(range(200_000, 200_128)),
        reward_config=RewardConfig(),
        max_episode_steps=27000,  # ~30 min
        description="Full six-floor run to Mom.",
    )


def stage5_generalization() -> Stage:
    """Same task as stage 4 but with a disjoint held-out seed pool."""
    return Stage(
        name="stage5_generalization",
        reset_stage=1,
        seed_pool=tuple(range(900_000, 900_064)),
        reward_config=RewardConfig(),
        max_episode_steps=27000,
        description="Eval-only stage on held-out seeds.",
    )


STAGES = {
    "stage1_single_room": stage1_single_room,
    "stage2_floor_clear": stage2_floor_clear,
    "stage3_two_floors": stage3_two_floors,
    "stage4_full_run": stage4_full_run,
    "stage5_generalization": stage5_generalization,
}


def get_stage(name: str) -> Stage:
    if name not in STAGES:
        raise KeyError(f"unknown curriculum stage: {name}. Available: {list(STAGES)}")
    return STAGES[name]()
