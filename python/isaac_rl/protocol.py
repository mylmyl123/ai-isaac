"""Framed TCP protocol shared by the Lua bridge and Python trainer.

Wire format: 4-byte big-endian length prefix + JSON payload (UTF-8).
JSON is used through M1 for legibility; swap to MessagePack once the schema stabilizes.
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any


LEN_STRUCT = struct.Struct(">I")

# On Windows, socket.recv() blocks indefinitely and ignores Python signals
# (Ctrl-C only fires after the syscall returns). We use a short timeout so
# recv wakes up periodically, giving the interpreter a chance to raise
# KeyboardInterrupt. The trainer's step loop retries transparently.
_RECV_TIMEOUT_S = 1.0


def send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(LEN_STRUCT.pack(len(body)) + body)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes. Retries through timeout wake-ups so signals get processed."""
    sock.settimeout(_RECV_TIMEOUT_S)
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            # Wake up, let Python check signals, keep waiting.
            continue
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, 4)
    (n,) = LEN_STRUCT.unpack(header)
    body = recv_exact(sock, n) if n else b""
    return json.loads(body.decode("utf-8")) if body else {}
