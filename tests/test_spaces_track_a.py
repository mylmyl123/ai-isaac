"""Track A (2026-07-12) schema regression tests.

Locks in the expanded action space, obs schema, and decoder behavior for
the BC-bootstrap pivot. Written after the gating experiment resolved
H_hard (see docs/analysis-2026-07-11/verdict.md) \u2014 we no longer resume
old v3 checkpoints, so breaking the [9, 5] action head is fine.

These tests guard against silent schema drift:
- `ACTION_FACTORS` == [9, 5, 2, 2, 2]
- `PASSIVES_K` == 733
- Doors have 18 features
- New obs keys exist and zero-fill when raw JSON is missing them
- Character one-hot maps player_type correctly (Isaac=0, Lilith=13,
  unknown->last slot)
- Active-item id normalized by 730; charge normalized by max_charge
"""
from __future__ import annotations

import numpy as np

from isaac_rl import spaces as sp


def test_action_factors_extended() -> None:
    assert sp.ACTION_FACTORS.tolist() == [9, 5, 2, 2, 2]
    assert sp.ACTION_KEYS == ("move", "shoot", "use_item", "drop_bomb", "use_pillcard")
    assert sum(sp.ACTION_FACTORS.tolist()) == 20


def test_passives_k_bumped() -> None:
    assert sp.PASSIVES_K == 733


def test_door_feats_18() -> None:
    assert sp.DOOR_FEATS == 18
    space = sp.observation_space()
    assert space["doors"].shape == (4, 18)


def test_new_obs_keys_in_space() -> None:
    space = sp.observation_space()
    for k in ("character", "active_items", "trinkets", "cards", "pills", "transformations"):
        assert k in space.spaces, f"missing obs key: {k}"
    assert space["character"].n == sp.CHARACTER_K == 35
    assert space["active_items"].shape == (2, 3)
    assert space["trinkets"].shape == (2, 2)
    assert space["cards"].shape == (4, 2)
    assert space["pills"].shape == (4, 2)
    assert space["transformations"].shape == (15,)


def test_zero_obs_covers_new_keys() -> None:
    z = sp.zero_obs()
    assert z["character"].shape == (35,)
    assert z["character"].dtype == np.int8    # MultiBinary
    assert z["active_items"].shape == (2, 3)
    assert z["transformations"].shape == (15,)


def test_encode_obs_missing_new_fields_backward_compat() -> None:
    """Raw JSON from before mod expansion still parses cleanly."""
    raw = {
        "schema": 1,
        "tick": 1,
        "player": {"x": 0, "y": 0, "hp_red": 3},   # no player_type / actives
        "global": {"stage": 1},
    }
    obs = sp.encode_obs(raw)
    assert obs["character"].sum() == 0
    assert obs["active_items"].sum() == 0
    assert obs["transformations"].sum() == 0
    assert obs["doors"].shape == (4, 18)


def test_encode_obs_populates_character_isaac() -> None:
    raw = {"player": {"player_type": 0}}
    obs = sp.encode_obs(raw)
    assert obs["character"][0] == 1
    assert obs["character"][1:].sum() == 0


def test_encode_obs_populates_character_lilith() -> None:
    raw = {"player": {"player_type": 13}}
    obs = sp.encode_obs(raw)
    assert obs["character"][13] == 1
    # All other slots zero.
    other = np.array(obs["character"]); other[13] = 0
    assert other.sum() == 0


def test_encode_obs_unknown_character_goes_to_last_slot() -> None:
    raw = {"player": {"player_type": 99}}   # tainted / DLC we don't classify
    obs = sp.encode_obs(raw)
    assert obs["character"][sp.CHARACTER_K - 1] == 1


def test_encode_obs_active_item_id_and_charge() -> None:
    raw = {
        "player": {
            "active_item_id": 105,        # D6 (Isaac's default)
            "active_charge": 3,
            "active_max_charge": 6,
        }
    }
    obs = sp.encode_obs(raw)
    # ID normalized by 730.
    assert abs(obs["active_items"][0][0] - 105.0 / 730.0) < 1e-6
    # Charge normalized by max (3/6 = 0.5).
    assert abs(obs["active_items"][0][1] - 0.5) < 1e-6
    # Has-flag == 1 (item present).
    assert obs["active_items"][0][2] == 1.0
    # Slot 1 empty.
    assert obs["active_items"][1][2] == 0.0


def test_encode_obs_transformations_clip_and_normalize() -> None:
    raw = {"player": {"transformations": [0, 3, 5, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}}
    obs = sp.encode_obs(raw)
    assert obs["transformations"][0] == 0.0
    assert abs(obs["transformations"][1] - 0.3) < 1e-6      # 3/10
    assert abs(obs["transformations"][2] - 0.5) < 1e-6      # 5/10
    assert obs["transformations"][3] == 1.0                  # 15 clipped to 10/10


def test_flatten_dict_obs_includes_new_keys() -> None:
    z = sp.zero_obs()
    flat = sp.flatten_dict_obs(z)
    for k in ("character", "active_items", "trinkets", "cards", "pills", "transformations"):
        assert k in flat, f"flatten_dict_obs missing key: {k}"
    # Character in the flat dict is float32 (zero_obs stored int8; flatten casts).
    assert flat["character"].dtype == np.float32


def test_encode_action_backward_compat_short_action() -> None:
    """Old callers passing [move, shoot] still get a valid 5-key dict."""
    d = sp.encode_action(np.array([3, 2]))
    assert d == {"move": 3, "shoot": 2, "use_item": 0, "drop_bomb": 0, "use_pillcard": 0}


def test_encode_action_full_length() -> None:
    d = sp.encode_action(np.array([1, 2, 1, 0, 1]))
    assert d == {"move": 1, "shoot": 2, "use_item": 1, "drop_bomb": 0, "use_pillcard": 1}
