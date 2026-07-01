"""Utility: convert a numpy Dict-obs (from spaces.flatten_dict_obs) into a torch dict on device."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .spaces import flatten_dict_obs


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def obs_to_tensors(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Flatten and convert a single-env obs dict to a batched (B=1) torch dict."""
    flat = flatten_dict_obs(obs)
    return {k: _to_tensor(v, device).unsqueeze(0) for k, v in flat.items()}


def batch_obs_to_tensors(obs_batch: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack N env observations into a batched torch dict, shape [B, ...]."""
    flats = [flatten_dict_obs(o) for o in obs_batch]
    keys = flats[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([f[k] for f in flats], axis=0)
        out[k] = _to_tensor(arr, device)
    return out


def stack_time_batch(rollout: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack T dicts of [B, ...] into a single dict of [T, B, ...]."""
    keys = rollout[0].keys()
    return {k: torch.stack([r[k] for r in rollout], dim=0) for k in keys}
