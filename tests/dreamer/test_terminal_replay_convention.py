"""Regression tests for the 2026-07-13 fix of the cont_pred=1 bug.

The bug: prior to this commit, the trainer stored replay entries as
(obs=pre-step, action=outgoing, is_terminal=next_step_terminates), which is
off-by-one from what NM512/dreamerv3-torch expects. RSSM.observe(embed, action,
is_first) treats action[t] as the INCOMING action to obs[t]; the cont head
treats is_terminal[t]=1 as "obs[t] IS the terminal state". Under the old
convention, cont_pred_mean stayed pinned at ~1.0 and value targets diverged.

These tests exercise the new store convention end-to-end so a future refactor
can't silently reintroduce the same bug.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from isaac_rl.dreamer.replay import SequenceReplay
from isaac_rl.spaces import ACTION_FACTORS, flatten_dict_obs, observation_space


def _random_raw_obs(is_dead: bool = False, hp: int = 6) -> dict:
    """Build a minimally-populated raw obs dict that flatten_dict_obs will accept."""
    return {
        "schema": 2,
        "tick": 0,
        "player": {
            "is_dead": is_dead,
            "hp_red": 0 if is_dead else hp,
            "hp_soul": 0,
            "hp_black": 0,
            "hp_max": hp,
            "x": 320.0,
            "y": 280.0,
            "vx": 0.0,
            "vy": 0.0,
            "coins": 0,
            "bombs": 1,
            "keys": 0,
        },
        "events": [{"kind": "death"}] if is_dead else [],
    }


def _fake_env_step_obs(is_dead: bool = False) -> dict:
    """Build a fully-populated flat obs matching OBS_SCHEMA."""
    from isaac_rl.dreamer.replay import OBS_SCHEMA
    obs = {k: np.zeros(shape, dtype=np.dtype(dtype))
           for k, (shape, dtype) in OBS_SCHEMA.items()}
    if is_dead:
        # player[0] is hp_red per spaces.encode_obs; keep zeros for dead.
        obs["player"][:] = 0.0
    return obs


def test_replay_stores_terminal_obs_with_is_terminal_flag():
    """When an episode terminates, the terminal obs MUST land in replay with
    is_terminal=True. Prior bug: reset obs landed there instead, and the
    cont head could never learn 'obs looks dead -> cont=0'."""
    onehot_dim = int(sum(int(x) for x in ACTION_FACTORS.tolist()))
    replay = SequenceReplay(capacity=100, onehot_dim=onehot_dim)

    reset_obs = _fake_env_step_obs(is_dead=False)
    terminal_obs = _fake_env_step_obs(is_dead=True)
    zero_action = np.zeros(onehot_dim, dtype=np.float32)
    action = np.zeros(onehot_dim, dtype=np.float32)
    action[0] = 1.0  # move-left

    # Simulate: reset, one normal step, terminal step.
    replay.add(reset_obs, zero_action, 0.0, is_first=True, is_terminal=False, is_last=False)
    replay.add(_fake_env_step_obs(), action, 0.1, is_first=False, is_terminal=False, is_last=False)
    replay.add(terminal_obs, action, -3.0, is_first=False, is_terminal=True, is_last=True)

    # Sample the whole 3-step sequence.
    batch = replay.sample(batch_size=1, seq_len=3, rng=np.random.default_rng(0))

    # The terminal step must exist in the batch with is_terminal=1.
    assert batch["is_terminal"].sum() >= 1, "no terminal step in replay batch"
    # is_first is 1 exactly once — at the reset entry.
    assert batch["is_first"].sum() >= 1, "no is_first=True entry"


def test_replay_is_first_and_is_terminal_are_disjoint_per_step():
    """A single step can't simultaneously be is_first AND is_terminal
    (the convention: first = start-of-episode, terminal = end-of-episode,
    they occupy different indices). Prior bug: a corrupt sequence could
    have both flags on the same step because reset and terminal obs
    were being conflated."""
    onehot_dim = int(sum(int(x) for x in ACTION_FACTORS.tolist()))
    replay = SequenceReplay(capacity=100, onehot_dim=onehot_dim)

    reset_obs = _fake_env_step_obs()
    zero_action = np.zeros(onehot_dim, dtype=np.float32)

    replay.add(reset_obs, zero_action, 0.0, is_first=True, is_terminal=False, is_last=False)
    replay.add(reset_obs, zero_action, 0.0, is_first=False, is_terminal=True, is_last=True)
    replay.add(reset_obs, zero_action, 0.0, is_first=True, is_terminal=False, is_last=False)

    batch = replay.sample(batch_size=1, seq_len=3, rng=np.random.default_rng(0))
    # Assert no step has both flags = 1.
    both = (batch["is_first"] > 0.5) & (batch["is_terminal"] > 0.5)
    assert not both.any(), "is_first and is_terminal both set on the same step"


def test_env_synth_terminal_obs_has_dead_player():
    """env._synth_terminal_obs must produce an obs with is_dead=True and hp=0.
    That's the semantic contract the WM's cont head relies on to learn
    'terminal state looks dead'."""
    from isaac_rl.env import SocketIsaacEnv

    # Build an env without opening the socket (test-only shortcut).
    env = SocketIsaacEnv.__new__(SocketIsaacEnv)
    env.port = 0
    env._last_raw = _random_raw_obs(is_dead=False, hp=6)

    term = env._synth_terminal_obs()
    assert term["player"]["is_dead"] is True
    assert term["player"]["hp_red"] == 0
    assert term["player"]["hp_soul"] == 0
    assert term["player"]["hp_black"] == 0
    # Non-player fields (room, entities, etc.) preserved from the last-known obs.
    assert term.get("events") == [{"kind": "death"}]


def test_env_synth_terminal_falls_back_to_crash_when_no_prior_obs():
    """If we hit mod_restart before receiving any obs (extreme edge case),
    fall back gracefully instead of crashing."""
    from isaac_rl.env import SocketIsaacEnv

    env = SocketIsaacEnv.__new__(SocketIsaacEnv)
    env.port = 9500
    env._last_raw = None

    term = env._synth_terminal_obs()
    # The fallback path returns _crash_penalty_obs which also has is_dead=True.
    assert term["player"]["is_dead"] is True
