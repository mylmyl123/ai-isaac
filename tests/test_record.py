"""Test the human-demo recorder's framing + JSONL output.

We can't launch a real Isaac binary from CI, so we simulate it with a
minimal socket client that sends length-prefixed JSON frames just like
mods/isaac-rl-bridge/net.lua does. Verifies:
  * record.py accepts our simulated client
  * Each frame becomes exactly one JSONL line in the output file
  * Human action + obs fields round-trip correctly
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path

import pytest

from isaac_rl.record import record_session


def _client_send_frames(port: int, frames: list[dict], connect_delay: float = 0.1) -> None:
    """Simulate the Lua mod: connect and send length-prefixed JSON frames."""
    time.sleep(connect_delay)  # give the server a moment to bind
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    for f in frames:
        payload = json.dumps(f).encode("utf-8")
        s.sendall(struct.pack(">I", len(payload)) + payload)
    time.sleep(0.05)          # flush before close
    s.close()


def test_record_session_writes_jsonl(tmp_path: Path) -> None:
    """One frame in -> one JSONL line out with the same content, including
    the extended human_action schema (move, shoot, use_item, drop_bomb,
    use_pillcard) and the new player-side identity/item fields."""
    port = 9503
    frames = [
        {
            "tick": 1, "schema": 2,
            "human_action": {
                "move": 3, "shoot": 0,
                "use_item": 0, "drop_bomb": 0, "use_pillcard": 0,
            },
            "player": {
                "hp_red": 6, "player_type": 0,
                "active_item_id": 105, "active_charge": 6, "active_max_charge": 6,
                "trinket_id_1": 0, "card_id_1": 0, "pill_id_1": 0,
                "transformations": [0]*15,
            },
        },
        {
            "tick": 2, "schema": 2,
            "human_action": {
                "move": 0, "shoot": 2,
                "use_item": 1, "drop_bomb": 0, "use_pillcard": 0,  # space pressed
            },
            "player": {
                "hp_red": 6, "player_type": 0,
                "active_item_id": 105, "active_charge": 6, "active_max_charge": 6,
                "transformations": [0]*15,
            },
        },
        {
            "tick": 3, "schema": 2,
            "human_action": {
                "move": 8, "shoot": 4,
                "use_item": 0, "drop_bomb": 1, "use_pillcard": 0,  # bomb dropped
            },
            "player": {
                "hp_red": 5, "player_type": 13,  # Lilith
                "active_item_id": 360, "active_charge": 2, "active_max_charge": 6,
                "transformations": [0]*15,
            },
        },
    ]

    t = threading.Thread(target=_client_send_frames, args=(port, frames), daemon=True)
    t.start()

    out_path = record_session(
        port=port,
        out_dir=tmp_path,
        isaac_binary=None,       # don't try to launch Isaac
        accept_timeout_s=5.0,
    )
    t.join(timeout=3.0)

    assert out_path is not None and out_path.exists()

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == len(frames), f"expected {len(frames)} lines, got {len(lines)}"

    for line, expected in zip(lines, frames):
        got = json.loads(line)
        assert got == expected


def test_record_session_timeout_when_no_client(tmp_path: Path) -> None:
    """If no Isaac connects, the recorder returns None cleanly."""
    port = 9504
    out_path = record_session(
        port=port,
        out_dir=tmp_path,
        isaac_binary=None,
        accept_timeout_s=0.5,   # short timeout
    )
    assert out_path is None
