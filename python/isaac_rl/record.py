"""Human demonstration recorder for BC-bootstrap training.

Launches Isaac in RECORD mode (via the ISAAC_RL_RECORD env var passed to the
child process) and passively listens on a socket. On every 15 Hz control tick,
the mod sends a JSON obs payload with an added ``human_action`` field
containing the player's current keyboard/gamepad state encoded as a
MultiDiscrete([9, 5]) tuple (same schema the RL policy emits).

We do NOT send actions back \u2014 the human is playing directly through Isaac's
normal input path. Our only job is to log the (obs, action) stream.

Output format: one JSON object per line (JSONL) at
``<out_dir>/session_<YYYYMMDD-HHMMSS>.jsonl``. Each object is the raw obs
payload sent by the mod, with these fields added Python-side by the mod:
  * ``human_action = { move: int in [0,9), shoot: int in [0,5) }``
  * Everything the mod would send in training mode (schema, tick, player,
    passives, room_grid, doors, enemies, projectiles, pickups, global,
    events, room_bounds).

Recording is schema-agnostic (we save the raw obs dict, not the encoded
numpy arrays), so if we bump the obs schema later we can re-encode existing
demos through the new encoder without re-recording. If a schema-breaking
change removes fields, the older demos degrade to missing values but stay
parseable.

CLI:
    python -m isaac_rl.record --isaac <path>              # launches Isaac
    python -m isaac_rl.record --port 9500 --out demos     # wait for external Isaac

Stop recording with Ctrl+C \u2014 saves final tick count and closes cleanly.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes. Returns None on EOF/timeout."""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (socket.timeout, ConnectionError):
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv_frame(sock: socket.socket) -> bytes | None:
    """Read one length-prefixed frame. Returns raw payload bytes or None."""
    header = _recv_exact(sock, 4)
    if not header:
        return None
    length = int.from_bytes(header, "big")
    # Sanity clamp: reject frames > 4 MB (any real obs is ~5-30 KB).
    if length <= 0 or length > 4 * 1024 * 1024:
        log.error("frame length out of range: %d bytes", length)
        return None
    return _recv_exact(sock, length)


def record_session(
    port: int = 9500,
    out_dir: Path = Path("demos"),
    isaac_binary: str | None = None,
    accept_timeout_s: float = 300.0,
) -> Path | None:
    """Record one session. Returns the output JSONL path, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"session_{session_id}.jsonl"

    proc: subprocess.Popen | None = None
    if isaac_binary:
        env = os.environ.copy()
        env["ISAAC_RL_PORT"] = str(port)
        env["ISAAC_RL_RECORD"] = "1"
        cmd = [isaac_binary, "--luadebug"]
        log.info("launching Isaac in RECORD mode: %s (port=%d)", " ".join(cmd), port)
        proc = subprocess.Popen(cmd, env=env)
    else:
        log.info("no --isaac path passed; waiting for external Isaac on port %d", port)
        log.info("(launch Isaac yourself with ISAAC_RL_RECORD=1 and ISAAC_RL_PORT=%d set)", port)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    server.settimeout(accept_timeout_s)

    log.info("listening on port %d (up to %.0fs for Isaac)...", port, accept_timeout_s)
    try:
        client, addr = server.accept()
    except socket.timeout:
        log.error("Isaac did not connect within %.0fs \u2014 giving up", accept_timeout_s)
        if proc:
            proc.terminate()
        server.close()
        return None
    client.settimeout(30.0)
    log.info("connected: %s", addr)
    log.info("writing to: %s", out_path)
    log.info("Play Isaac normally. Ctrl+C in THIS window to stop.")

    tick_count = 0
    t_start = time.time()
    try:
        with open(out_path, "w") as f:
            while True:
                frame = _recv_frame(client)
                if frame is None:
                    log.warning("connection lost / EOF after %d ticks", tick_count)
                    break
                # Raw JSON payload \u2014 write one per line.
                f.write(frame.decode("utf-8", errors="replace"))
                f.write("\n")
                tick_count += 1
                if tick_count % 100 == 0:
                    dt = max(1e-6, time.time() - t_start)
                    hz = tick_count / dt
                    print(f"\rrecorded {tick_count} ticks ({hz:.1f} Hz, {dt:.0f}s elapsed)", end="", flush=True)
    except KeyboardInterrupt:
        print()
        log.info("Ctrl+C received \u2014 stopping recording")
    finally:
        try:
            client.close()
        except Exception:
            pass
        server.close()
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    dt = max(1e-6, time.time() - t_start)
    log.info("session complete: %d ticks in %.1fs (%.1f Hz avg) \u2192 %s",
             tick_count, dt, tick_count / dt, out_path)
    if tick_count < 100:
        log.warning("very few ticks recorded \u2014 check that RECORD_MODE actually took effect")
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Isaac RL human demo recorder")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--out", type=Path, default=Path("demos"),
                    help="Output directory (created if missing). Default: demos/")
    ap.add_argument("--isaac", type=str, default="",
                    help="Path to isaac-ng.exe. Empty = don't launch, wait for external Isaac.")
    ap.add_argument("--accept-timeout-s", type=float, default=300.0,
                    help="Seconds to wait for Isaac to connect. Default: 300.")
    args = ap.parse_args()
    isaac = args.isaac if args.isaac else None
    record_session(
        port=args.port,
        out_dir=args.out,
        isaac_binary=isaac,
        accept_timeout_s=args.accept_timeout_s,
    )


if __name__ == "__main__":
    main()
