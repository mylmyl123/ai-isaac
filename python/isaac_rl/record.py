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
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Global stop flag. Set by SIGINT/SIGBREAK/SIGTERM handlers OR by the presence
# of a demos/STOP file. The recording loop polls this on every timeout tick
# so we can stop from any of: Ctrl+C in PowerShell, Ctrl+Break, `taskkill`,
# or `New-Item demos\STOP`.
_stop_flag = threading.Event()


def _install_stop_handlers() -> None:
    """Register OS signal handlers that set _stop_flag.

    On Windows, PowerShell's Ctrl+C forwarding to a running python subprocess
    is notoriously unreliable when the child is blocked in a native call
    (like ``socket.recv``). Registering an explicit ``signal.signal`` handler
    for SIGINT + SIGBREAK works around this: the handler runs in Python's
    signal thread and sets the flag; the main loop checks the flag on every
    1-second socket-timeout tick and exits cleanly within ~1s.
    """
    def _handler(signum, _frame):
        # Best-effort log, but don't rely on it (stdout may be redirected).
        log.info("stop signal %d received; finalizing session", signum)
        _stop_flag.set()

    # SIGINT = Ctrl+C. Universal.
    signal.signal(signal.SIGINT, _handler)
    # SIGBREAK = Ctrl+Break on Windows only. Fallback if Ctrl+C is swallowed
    # by some terminal / IDE combo (e.g. VS Code's integrated terminal).
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)
    # SIGTERM = graceful `taskkill /pid <pid>` and similar. Handle so we
    # still flush the JSONL properly instead of hard-killing mid-write.
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            pass  # not raisable on some Windows configs


def _stop_file_path(out_dir: Path) -> Path:
    return out_dir / "STOP"


def _check_stop(out_dir: Path) -> bool:
    """Return True if we should stop (signal received OR STOP file exists)."""
    if _stop_flag.is_set():
        return True
    if _stop_file_path(out_dir).exists():
        log.info("STOP file found at %s; ending session", _stop_file_path(out_dir))
        _stop_flag.set()
        return True
    return False


def _recv_exact(sock: socket.socket, n: int) -> tuple[bytes | None, str]:
    """Read exactly ``n`` bytes.

    Returns ``(bytes, 'ok')`` on success, ``(None, 'timeout')`` if the socket
    timed out with zero bytes read, or ``(None, 'eof')`` if the peer closed
    the connection or a socket error occurred. Callers should treat 'timeout'
    as 'keep waiting' (Isaac paused, on game-over screen, etc.) and only
    give up on 'eof' — conflating the two would mean disconnecting every
    time the player dies for a few seconds while the mod's reset_cooldown
    suppresses obs frames, which is exactly the bug we're fixing.
    """
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            return None, "timeout"
        except (ConnectionError, OSError):
            return None, "eof"
        if not chunk:
            return None, "eof"
        buf += chunk
    return buf, "ok"


def _recv_frame(sock: socket.socket) -> tuple[bytes | None, str]:
    """Read one length-prefixed frame. Returns (payload, status).

    Status is one of 'ok' (payload valid), 'timeout' (no data ready but
    connection alive), or 'eof' (real disconnect). Payload is None unless
    status is 'ok'.
    """
    header, status = _recv_exact(sock, 4)
    if status != "ok":
        return None, status
    length = int.from_bytes(header, "big")
    # Sanity clamp: reject frames > 4 MB (any real obs is ~5-30 KB).
    if length <= 0 or length > 4 * 1024 * 1024:
        log.error("frame length out of range: %d bytes", length)
        return None, "eof"
    payload, status = _recv_exact(sock, length)
    return payload, status


def record_session(
    port: int = 9500,
    out_dir: Path = Path("demos"),
    isaac_binary: str | None = None,
    accept_timeout_s: float = 300.0,
    min_ticks: int = 150,   # ~10s @ 15 Hz; below this we prompt to discard
) -> Path | None:
    """Record one session. Returns the output JSONL path, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"session_{session_id}.jsonl"

    _install_stop_handlers()
    # Clear any stale STOP file left from a previous session.
    stop_file = _stop_file_path(out_dir)
    if stop_file.exists():
        stop_file.unlink()
    _stop_flag.clear()
    proc: subprocess.Popen | None = None
    if isaac_binary:
        env = os.environ.copy()
        env["ISAAC_RL_PORT"] = str(port)
        env["ISAAC_RL_RECORD"] = "1"
        cmd = [isaac_binary, "--luadebug"]
        # CRITICAL: Isaac loads resources/scripts/enums.lua, resources/packed/*.a,
        # and all game assets via paths relative to its own CWD. When we
        # subprocess.Popen(cmd) without cwd=, the child inherits our repo dir,
        # Isaac can't find resources/, and dies before the mod even loads with
        # "ERR: cannot open resources/scripts/enums.lua". Steam's -applaunch
        # sets CWD internally which is why the training launcher works; here
        # we launch the binary directly so we have to set it ourselves.
        isaac_dir = os.path.dirname(isaac_binary) or "."
        log.info("launching Isaac in RECORD mode: %s (port=%d, cwd=%s)",
                 " ".join(cmd), port, isaac_dir)
        proc = subprocess.Popen(cmd, env=env, cwd=isaac_dir)
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
    client.settimeout(1.0)  # short poll so Ctrl+C is responsive on Windows PowerShell
    log.info("connected: %s", addr)
    log.info("writing to: %s", out_path)
    log.info("Play Isaac normally. Ctrl+C in THIS window to stop.")
    log.info("  fallback: `New-Item %s` from another shell also stops.", stop_file)

    tick_count = 0
    t_start = time.time()
    try:
        with open(out_path, "w") as f:
            while True:
                if _check_stop(out_dir):
                    break
                # 1s socket timeout — gives Ctrl+C a chance. NEVER treat a
                # timeout as disconnect: Isaac's death sequence, game-over
                # screen, pause menu, and main-menu-return all produce
                # multi-second gaps where the mod sends no obs but is very
                # much alive. Only 'eof' (Isaac's socket actually closed)
                # or a user stop signal ends the loop.
                frame, status = _recv_frame(client)
                if status == "eof":
                    log.warning("socket closed by Isaac after %d ticks (real disconnect)", tick_count)
                    break
                if status == "timeout":
                    continue
                # status == 'ok': write the payload.
                # Flush every tick so Ctrl+C mid-gameplay never loses data.
                f.write(frame.decode("utf-8", errors="replace"))
                f.write("\n")
                f.flush()
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
        # DO NOT terminate Isaac on session end. The Ctrl+C / STOP path is
        # user-initiated and they might want to keep playing (rerun record.ps1
        # to start a fresh session against the same Isaac window). If Isaac's
        # socket EOF'd naturally, Isaac has already exited anyway, so
        # proc.terminate() would be a no-op. Killing here was the bug that
        # made 'die -> game exits' happen: the recorder's own 60s idle
        # timeout was interpreted as 'session over', then proc.terminate()
        # closed Isaac's window.
        if proc is not None and proc.poll() is None:
            log.info("session ended; Isaac (pid=%d) left running — close it manually if desired", proc.pid)

    dt = max(1e-6, time.time() - t_start)
    log.info("session complete: %d ticks in %.1fs (%.1f Hz avg) \u2192 %s",
             tick_count, dt, tick_count / dt, out_path)
    if tick_count < 100:
        log.warning("very few ticks recorded \u2014 check that RECORD_MODE actually took effect")

    # Auto-discard trivially short sessions. These are almost always
    # 'launched then immediately restarted / quit' — they clutter the BC
    # corpus with zero signal. Ask the user before deleting so they can
    # override (Enter=discard, 'n'=keep).
    if tick_count < min_ticks:
        try:
            resp = input(
                f"\nsession is only {tick_count} ticks ({dt:.0f}s) — discard? [Y/n]: "
            ).strip().lower()
        except EOFError:
            resp = "y"  # non-interactive shells: default discard
        if resp in ("", "y", "yes"):
            try:
                out_path.unlink()
                log.info("discarded: %s", out_path)
            except OSError as e:
                log.error("failed to delete %s: %s", out_path, e)
            return None
        log.info("kept: %s", out_path)

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
    ap.add_argument("--min-ticks", type=int, default=150,
                    help="Sessions shorter than this ask to discard on exit. "
                         "Default: 150 (~10s @ 15 Hz). Pass 0 to keep all sessions.")
    args = ap.parse_args()
    isaac = args.isaac if args.isaac else None
    record_session(
        port=args.port,
        out_dir=args.out,
        isaac_binary=isaac,
        accept_timeout_s=args.accept_timeout_s,
        min_ticks=args.min_ticks,
    )


if __name__ == "__main__":
    main()
