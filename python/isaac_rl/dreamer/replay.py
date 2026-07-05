"""Sequence replay buffer for DreamerV3.

Design notes:
- Stores transitions in a flat ring buffer per env. Each entry is
  (obs, action_onehot, reward, is_first, is_terminal, is_last, discount).
- ``is_first`` is 1 on the first observation of each episode — RSSM resets
  its hidden state on ``is_first=True`` (see vendor/networks.py:174-193).
- ``is_terminal`` is 1 on the step where the env terminated (death, beat_mom).
  This is what the continue-head (``cont``) is trained to predict.
- ``is_last`` is 1 on the last step of an episode (either terminated or
  truncated) — used only to know episode boundaries when sampling.
- We store obs *after* the action was taken (i.e., ``o_{t+1}``). The very
  first entry of each episode is the reset obs with a zero action and
  reward, and is_first=True.

Sampling contract:
  ``sample(batch_size, seq_len) -> dict[str, Tensor]``
  Returns a dict with keys: everything in obs (flat schema), plus
  ``action`` [B, T, onehot_dim] float, ``reward`` [B, T] float,
  ``is_first`` [B, T] float (0/1), ``is_terminal`` [B, T] float.
  Sequences may cross episode boundaries; ``is_first`` marks the reset.

Episode-boundary handling: unlike Dreamer's canonical implementation which
samples fixed-length windows within completed episodes, we allow sequences
to include an episode boundary — the RSSM handles it via the ``is_first``
flag (that's exactly what the flag is for). This gives us cleaner sampling
without worrying about short episodes near the buffer's start.

Memory footprint at defaults:
  - 1M transitions * (obs ~7 KB + action 14 * 4 + scalars ~40) ≈ 7 GB RAM.
  - We store obs on CPU as float32 (Isaac's schema); move to GPU on sample.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..spaces import (
    ENEMY_FEATS,
    GLOBAL_DIM,
    MAX_ENEMIES,
    MAX_PICKUPS,
    MAX_PROJECTILES,
    PASSIVES_K,
    PICKUP_FEATS,
    PLAYER_DIM,
    PLAYER_HISTORY_DIM,
    PROJ_FEATS,
    ROOM_H,
    ROOM_W,
    SPATIAL_DIM,
    Z_DIM,
    flatten_dict_obs,
)


# The exact obs keys and per-key shapes/dtypes we store. Matches
# spaces.flatten_dict_obs output. Kept as a module constant so the buffer
# can preallocate storage.
OBS_SCHEMA: dict[str, tuple[tuple[int, ...], str]] = {
    "player":            ((PLAYER_DIM,), "float32"),
    "passives":          ((PASSIVES_K,), "float32"),
    "room_grid":         ((4, ROOM_H, ROOM_W), "float32"),
    "doors":             ((4, 6), "float32"),
    "global":            ((GLOBAL_DIM,), "float32"),
    "last_action":       ((2,), "float32"),
    "spatial":           ((SPATIAL_DIM,), "float32"),
    "player_history":    ((PLAYER_HISTORY_DIM,), "float32"),
    "z":                 ((Z_DIM,), "float32"),
    "enemies_feats":     ((MAX_ENEMIES, ENEMY_FEATS), "float32"),
    "enemies_mask":      ((MAX_ENEMIES,), "float32"),
    "projectiles_feats": ((MAX_PROJECTILES, PROJ_FEATS), "float32"),
    "projectiles_mask":  ((MAX_PROJECTILES,), "float32"),
    "pickups_feats":     ((MAX_PICKUPS, PICKUP_FEATS), "float32"),
    "pickups_mask":      ((MAX_PICKUPS,), "float32"),
}


class SequenceReplay:
    """Ring buffer over transitions, sampled as fixed-length sequences.

    Args:
      capacity: max number of transitions.
      onehot_dim: sum of MultiDiscrete factors (Isaac: 14).
      dtype_obs: numpy dtype for obs tensors. Defaults to float32.
    """

    def __init__(self, capacity: int, onehot_dim: int):
        self.capacity = int(capacity)
        self.onehot_dim = int(onehot_dim)

        # Allocate per-key obs buffers.
        self._obs: dict[str, np.ndarray] = {}
        for k, (shape, dtype) in OBS_SCHEMA.items():
            self._obs[k] = np.zeros((self.capacity,) + shape, dtype=np.dtype(dtype))

        self._action = np.zeros((self.capacity, self.onehot_dim), dtype=np.float32)
        self._reward = np.zeros(self.capacity, dtype=np.float32)
        self._is_first = np.zeros(self.capacity, dtype=np.float32)
        self._is_terminal = np.zeros(self.capacity, dtype=np.float32)
        self._is_last = np.zeros(self.capacity, dtype=np.float32)

        self._idx = 0                # next write position
        self._filled = 0             # number of valid entries (<= capacity)

    def __len__(self) -> int:
        return self._filled

    def add(
        self,
        obs: dict[str, np.ndarray],
        action_onehot: np.ndarray,
        reward: float,
        is_first: bool,
        is_terminal: bool,
        is_last: bool,
    ) -> None:
        """Add one transition. ``obs`` is the flat-schema dict from ``flatten_dict_obs``."""
        i = self._idx
        for k in OBS_SCHEMA:
            self._obs[k][i] = obs[k]
        self._action[i] = action_onehot
        self._reward[i] = reward
        self._is_first[i] = 1.0 if is_first else 0.0
        self._is_terminal[i] = 1.0 if is_terminal else 0.0
        self._is_last[i] = 1.0 if is_last else 0.0
        self._idx = (self._idx + 1) % self.capacity
        self._filled = min(self._filled + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """Sample ``batch_size`` sequences of length ``seq_len``.

        Returns dict with:
          - All obs keys, shape ``[B, T, ...]``.
          - ``action`` ``[B, T, onehot_dim]``, ``reward`` ``[B, T]``,
            ``is_first`` ``[B, T]``, ``is_terminal`` ``[B, T]``.

        Sampling is uniform over all valid start indices such that the whole
        window ``[start, start+seq_len)`` fits in the filled region. We exclude
        sequences that would wrap around the ring boundary during a partial fill.
        """
        if self._filled < seq_len:
            raise ValueError(f"buffer has {self._filled} < seq_len={seq_len}, cannot sample")
        if rng is None:
            rng = np.random.default_rng()

        # Valid start indices: any position s such that s + seq_len <= _filled
        # when the buffer isn't full yet; otherwise sample uniformly and use
        # modulo indexing (all positions are valid because the ring is full).
        if self._filled < self.capacity:
            max_start = self._filled - seq_len
            starts = rng.integers(0, max_start + 1, size=batch_size)
            # Absolute indices, no wrap.
            idx_grid = starts[:, None] + np.arange(seq_len)[None, :]
        else:
            # Ring is full. All starts are valid; do modulo indexing.
            # Skip starts that land on the write cursor (would splice old + new
            # transitions from different episodes into a fabricated sequence).
            # Simplest: exclude a small band around _idx.
            starts = rng.integers(0, self.capacity, size=batch_size)
            # If any window would cross _idx, resample. In practice this is rare
            # (band of width seq_len over capacity 1M is 0.006% of starts).
            # We just retry up to a few times.
            for _ in range(5):
                bad = ((starts <= self._idx) & (starts + seq_len > self._idx)) | \
                      ((starts + seq_len) % self.capacity < starts) & (self._idx < starts)
                if not bad.any():
                    break
                starts[bad] = rng.integers(0, self.capacity, size=int(bad.sum()))
            idx_grid = (starts[:, None] + np.arange(seq_len)[None, :]) % self.capacity

        out: dict[str, np.ndarray] = {}
        for k, arr in self._obs.items():
            out[k] = arr[idx_grid]                  # [B, T, ...]
        out["action"] = self._action[idx_grid]      # [B, T, onehot_dim]
        out["reward"] = self._reward[idx_grid]      # [B, T]
        out["is_first"] = self._is_first[idx_grid]
        out["is_terminal"] = self._is_terminal[idx_grid]
        return out


def encode_and_add(
    replay: SequenceReplay,
    obs: dict[str, Any],
    action_onehot: np.ndarray,
    reward: float,
    is_first: bool,
    is_terminal: bool,
    is_last: bool,
) -> None:
    """Convenience: flatten obs (from env's nested Dict form) and add to replay."""
    flat = flatten_dict_obs(obs)
    replay.add(flat, action_onehot, reward, is_first, is_terminal, is_last)
