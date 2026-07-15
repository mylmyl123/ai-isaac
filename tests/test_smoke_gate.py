"""Tests for tools/smoke_gate.py — the identity gate must PASS on the correct
enemy and FAIL on a substituted one (the wrong-enemy false-pass it exists to
catch). Uses synthetic log.txt + episodes.csv fixtures; no live Isaac.

Run:
    PYTHONPATH=python pytest tests/test_smoke_gate.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("smoke_gate", REPO / "tools" / "smoke_gate.py")
smoke_gate = importlib.util.module_from_spec(_spec)
sys.modules["smoke_gate"] = smoke_gate
_spec.loader.exec_module(smoke_gate)


HORF_LOG = """\
[INFO] - Lua Debug: [isaac-rl-bridge] curriculum stage=0 enemy_type=12
[INFO] - Lua Debug: [isaac-rl-bridge] STAGE=0: spawned type=12 at (100,80) dist=250
[INFO] - Lua Debug: [isaac-rl-bridge] kill npc_type=12 variant=0
[INFO] - Lua Debug: [isaac-rl-bridge] kill npc_type=12 variant=0
[INFO] - Lua Debug: [isaac-rl-bridge] handle_player_death firing (source=MC_POST_UPDATE)
"""

MAW_LOG = """\
[INFO] - Lua Debug: [isaac-rl-bridge] STAGE=0: spawned type=26 at (100,80) dist=250
[INFO] - Lua Debug: [isaac-rl-bridge] kill npc_type=26 variant=0
[INFO] - Lua Debug: [isaac-rl-bridge] handle_player_death firing (source=MC_POST_UPDATE)
"""

EPISODES_CSV = """\
step,env_idx,ep_r,ep_len,ep_kills,terminated,truncated
100,0,3.5,300,5,1,0
250,1,2.1,200,3,1,0
400,0,4.0,350,6,1,0
"""


def _write(tmp_path, log_text):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "episodes.csv").write_text(EPISODES_CSV, encoding="utf-8")
    log = tmp_path / "log.txt"
    log.write_text(log_text, encoding="utf-8")
    return tmp_path / "run", log


def _run_gate(run_dir, log, expected_type):
    argv = ["smoke_gate", "--run-dir", str(run_dir), "--isaac-log", str(log),
            "--expected-enemy-type", str(expected_type), "--min-episodes", "3"]
    old = sys.argv
    sys.argv = argv
    try:
        return smoke_gate.main()
    finally:
        sys.argv = old


def test_gate_passes_on_correct_horf(tmp_path):
    run_dir, log = _write(tmp_path, HORF_LOG)
    assert _run_gate(run_dir, log, 12) == 0


def test_gate_fails_on_maw_substituted_for_horf(tmp_path):
    # THE regression: spawning/killing Maw(26) while expecting Horf(12) MUST fail.
    run_dir, log = _write(tmp_path, MAW_LOG)
    assert _run_gate(run_dir, log, 12) == 1


def test_gate_fails_when_no_kills_logged(tmp_path):
    log_text = "[INFO] - Lua Debug: [isaac-rl-bridge] STAGE=0: spawned type=12 dist=250\n"
    run_dir, log = _write(tmp_path, log_text)
    assert _run_gate(run_dir, log, 12) == 1


def test_gate_fails_when_log_missing(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "episodes.csv").write_text(EPISODES_CSV, encoding="utf-8")
    assert _run_gate(tmp_path / "run", tmp_path / "nope.txt", 12) == 1


def test_parse_isaac_log_counts(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(HORF_LOG, encoding="utf-8")
    parsed = smoke_gate.parse_isaac_log(log)
    assert parsed["spawns"][12] == 1
    assert parsed["kills"][12] == 2
    assert parsed["deaths"] == 1
