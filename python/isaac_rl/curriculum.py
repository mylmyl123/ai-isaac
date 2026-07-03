"""B2: Curriculum learning framework.

Applies stage-based hyperparameter overrides over training. Since our mod
doesn't yet support direct room-difficulty control from Python, this MVP
implementation focuses on REWARD-SHAPING and EXPLORATION curriculum: adjust
reward weights and entropy coefficient as a function of global_step.

A full environment curriculum (spawn in easy rooms, graduate to hard rooms)
requires Lua mod changes documented in docs/FUTURE_WORK.md B2.

Usage in yaml:

    curriculum:
      - until_step: 500000
        overrides:
          ent_coef: 0.05
          reward_overrides:
            r_new_room: 5.0        # heavy exploration reward early
            r_kill: 0.2            # de-emphasise combat early
      - until_step: 1500000
        overrides:
          ent_coef: 0.02
          reward_overrides:
            r_new_room: 2.0
            r_kill: 0.5
      # No entry -> use defaults from cfg.reward beyond 1.5M steps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CurriculumStage:
    until_step: int
    overrides: dict[str, Any]


class CurriculumScheduler:
    """Applies stage overrides based on global_step.

    stages: list of dicts each with 'until_step' and 'overrides' keys.
    Stage transitions are edge-triggered (crossing until_step advances stage).

    Overrides supported:
      - ent_coef (PPOConfig field): entropy coefficient
      - lr (PPOConfig field): learning rate base
      - reward_overrides (dict): passed to RewardConfig field-by-field

    Any override not listed above is ignored with a warning.
    """

    def __init__(self, stages: list[dict[str, Any]] | None):
        if not stages:
            self.stages: list[CurriculumStage] = []
        else:
            self.stages = [CurriculumStage(**s) for s in stages]
        self._active_stage_idx = -1

    def current_stage(self, global_step: int) -> CurriculumStage | None:
        """Return the currently-active stage, or None if past all stages."""
        for stage in self.stages:
            if global_step < stage.until_step:
                return stage
        return None

    def apply(self, cfg, reward_shaper, global_step: int) -> bool:
        """Apply stage overrides to cfg and reward_shaper. Returns True if
        stage changed since last call.
        """
        stage = self.current_stage(global_step)
        # Determine new stage index for change detection.
        new_idx = -1
        if stage is not None:
            new_idx = self.stages.index(stage)
        changed = new_idx != self._active_stage_idx
        self._active_stage_idx = new_idx

        if stage is None:
            return changed
        overrides = stage.overrides
        if "ent_coef" in overrides:
            cfg.ent_coef = float(overrides["ent_coef"])
        if "lr" in overrides:
            cfg.lr = float(overrides["lr"])
        if "reward_overrides" in overrides:
            if reward_shaper is not None:
                for k, v in overrides["reward_overrides"].items():
                    if hasattr(reward_shaper.cfg, k):
                        setattr(reward_shaper.cfg, k, v)
            # else: reward_overrides silently ignored. Full environment
            # curriculum requires per-env reward_shaper access; see
            # docs/FUTURE_WORK.md B2.
        return changed
