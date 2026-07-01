"""One-shot training launcher.

What this does:
  1. Reads a PPO config (YAML) to figure out how many Isaac instances to run.
  2. Spawns each Isaac process with --luadebug and ISAAC_RL_PORT set.
  3. Waits until every Isaac has connected its socket to the trainer
     (the trainer opens the server; each Isaac's MC_POST_GAME_STARTED connects in).
  4. Runs training in this same process. TensorBoard events land under `runs/`.
  5. On Ctrl-C or on training completion, kills every child Isaac cleanly.

Usage (PowerShell):

    .\.venv\Scripts\Activate.ps1
    python train.py --config python\isaac_rl\configs\stage1_single_room.yaml `
                    --isaac "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"

If --isaac is omitted the script tries the config's `isaac_binary`, then a
handful of default Steam install paths.

Notes:
  - You still have to click "New Run" (or wait for the auto-continue) in each
    Isaac window on first launch, so it fires MC_POST_GAME_STARTED and
    connects the socket. After that the trainer drives resets automatically.
  - Set --tensorboard to also start TensorBoard in the background at :6006.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make sure the trainer package is importable no matter where you run this from.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))

from isaac_rl.ppo import PPOConfig, _cfg_from_yaml, train  # noqa: E402


log = logging.getLogger("train")


DEFAULT_ISAAC_BINARIES_WINDOWS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"C:\Program Files\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
]
DEFAULT_ISAAC_BINARIES_LINUX = [
    str(Path.home() / ".steam/steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
    str(Path.home() / ".local/share/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
]
DEFAULT_ISAAC_BINARIES_DARWIN = [
    str(Path.home() / "Library/Application Support/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
]


def resolve_isaac_binary(explicit: str | None, from_cfg: str | None) -> str | None:
    """Try the CLI arg, then the YAML value, then a list of platform defaults."""
    for candidate in (explicit, from_cfg):
        if candidate:
            p = Path(candidate)
            if p.exists():
                return str(p)
            log.warning("configured isaac_binary does not exist: %s", candidate)
    system = platform.system()
    lst = {
        "Windows": DEFAULT_ISAAC_BINARIES_WINDOWS,
        "Linux": DEFAULT_ISAAC_BINARIES_LINUX,
        "Darwin": DEFAULT_ISAAC_BINARIES_DARWIN,
    }.get(system, [])
    for candidate in lst:
        if Path(candidate).exists():
            log.info("using auto-detected isaac binary: %s", candidate)
            return candidate
    return None


class IsaacFleet:
    """A collection of Isaac child processes, one per port. Cleans itself up on exit."""

    def __init__(
        self,
        binary: str,
        base_port: int,
        n_envs: int,
        extra_args: list[str] | None = None,
        auto_start_stage: int | None = 1,
    ):
        self.binary = binary
        self.base_port = base_port
        self.n_envs = n_envs
        self.extra_args = extra_args or []
        self.auto_start_stage = auto_start_stage
        self.procs: list[subprocess.Popen] = []

    def _instance_workdir(self, i: int) -> Path:
        """Isolated per-instance working directory.

        Steam / Isaac otherwise fight over the same log.txt, save files, and
        lock files, and Steam's DRM will collapse two launches into one window.
        Giving each Isaac its own cwd makes them independent.

        We also drop a steam_appid.txt into this cwd — that file tells the
        Steamworks DRM stub "yes, we were legitimately launched, don't relaunch
        under Steam." Without it, Isaac exits within a couple seconds.
        """
        d = Path.cwd() / ".isaac-instances" / f"port_{self.base_port + i}"
        d.mkdir(parents=True, exist_ok=True)
        appid_file = d / "steam_appid.txt"
        if not appid_file.exists():
            appid_file.write_text("250900\n", encoding="utf-8")
        return d

    def spawn(self, stagger_s: float = 3.0) -> None:
        for i in range(self.n_envs):
            port = self.base_port + i
            env = os.environ.copy()
            env["ISAAC_RL_PORT"] = str(port)
            # NOTE: do NOT set SteamAppId / SteamGameId here — Repentance's
            # DRM stub reads them and refuses if they don't match a real
            # Steam context. The steam_appid.txt in the cwd is the right
            # mechanism (created in _instance_workdir).

            cmd = [self.binary, "--luadebug"]
            if self.auto_start_stage is not None:
                # Boot straight into a run at this stage — no title-screen click.
                cmd += ["--set-stage", str(self.auto_start_stage)]
            cmd += self.extra_args

            workdir = self._instance_workdir(i)

            log.info("[isaac %d/%d] port=%d cwd=%s cmd=%s",
                     i + 1, self.n_envs, port, workdir, " ".join(cmd))

            creationflags = 0
            if platform.system() == "Windows":
                # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so each child is
                # fully independent of this Python process's console. Without
                # NEW_PROCESS_GROUP, Ctrl-C in this terminal would also kill Isaac
                # BEFORE our signal handler can shut them down gracefully.
                creationflags = (
                    getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )

            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(workdir),
                creationflags=creationflags,
            )
            self.procs.append(proc)
            # Stagger so the OS doesn't spike CPU on N simultaneous cold starts.
            time.sleep(stagger_s)

    def shutdown(self) -> None:
        for i, p in enumerate(self.procs):
            if p.poll() is None:
                log.info("[isaac %d] terminating pid %d", i, p.pid)
                try:
                    p.terminate()
                except Exception as e:
                    log.warning("terminate failed for pid %d: %s", p.pid, e)
        # Give them a grace period, then kill.
        deadline = time.time() + 10
        for p in self.procs:
            remaining = max(0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log.warning("[isaac] force-killing pid %d", p.pid)
                try:
                    p.kill()
                except Exception:
                    pass


def maybe_start_tensorboard(logdir: Path, port: int = 6006) -> subprocess.Popen | None:
    exe = "tensorboard"
    try:
        proc = subprocess.Popen(
            [exe, "--logdir", str(logdir), "--port", str(port), "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("tensorboard started on http://localhost:%d (logdir=%s)", port, logdir)
        return proc
    except FileNotFoundError:
        log.warning("tensorboard not on PATH; skipping (pip install tensorboard).")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to a YAML config under python/isaac_rl/configs/")
    ap.add_argument("--isaac", default=None, help="Absolute path to isaac-ng.exe (overrides config + auto-detect)")
    ap.add_argument("--n-envs", type=int, default=None, help="Override n_envs from the config")
    ap.add_argument("--base-port", type=int, default=None, help="Override base_port from the config")
    ap.add_argument("--no-launch-isaac", action="store_true",
                    help="Don't spawn Isaac processes; expect them to be started manually.")
    ap.add_argument("--no-auto-start", action="store_true",
                    help="Don't auto-boot into a run. You'll have to click 'New Run' in each window.")
    ap.add_argument("--auto-start-stage", type=int, default=1,
                    help="Stage passed to --set-stage on first launch (default: 1). Only affects the first run; "
                         "subsequent resets are driven by the trainer via Isaac.ExecuteCommand.")
    ap.add_argument("--tensorboard", action="store_true", help="Also start TensorBoard in the background at :6006")
    ap.add_argument("--override", nargs="*", default=[], help="Extra config overrides: key=value")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load and reconcile config.
    cfg = _cfg_from_yaml(args.config)
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.base_port is not None:
        cfg.base_port = args.base_port
    for kv in args.override:
        k, _, v = kv.partition("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        setattr(cfg, k, v)

    # Decide whether we're launching Isaac ourselves. This unified launcher
    # ALWAYS launches — that's the whole point. Force the flag off inside the
    # trainer's own vec_env config so it doesn't try to double-spawn.
    launch = not args.no_launch_isaac
    cfg.launch_isaac = False  # trainer must NOT spawn; we do.
    cfg.isaac_binary = None

    log.info("config: %s", asdict(cfg))

    fleet: IsaacFleet | None = None
    tb_proc: subprocess.Popen | None = None

    def cleanup():
        if fleet is not None:
            fleet.shutdown()
        if tb_proc is not None and tb_proc.poll() is None:
            tb_proc.terminate()

    atexit.register(cleanup)

    # Signal handling notes for Windows:
    #   - Ctrl-C in the console raises KeyboardInterrupt in Python code at the
    #     next opportunity. It does NOT reach our SIGINT handler while a
    #     blocking syscall is running.
    #   - We work around that by setting short timeouts on socket recv/accept
    #     (see protocol.py and env.py) so the interpreter regularly gets a
    #     chance to raise the exception.
    #   - The Isaac children were spawned with CREATE_NEW_PROCESS_GROUP so
    #     Ctrl-C in our console doesn't reach them — cleanup() terminates
    #     them explicitly.
    def handle_signal(signum, frame):
        log.info("caught signal %d, shutting down", signum)
        cleanup()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    if args.tensorboard:
        tb_proc = maybe_start_tensorboard(Path(cfg.checkpoint_dir))

    if launch:
        binary = resolve_isaac_binary(args.isaac, None)
        if not binary:
            log.error(
                "could not find Isaac binary. Pass --isaac <path-to-isaac-ng.exe>, "
                "or use --no-launch-isaac and start Isaac yourself."
            )
            return 2
        fleet = IsaacFleet(
            binary=binary,
            base_port=cfg.base_port,
            n_envs=cfg.n_envs,
            auto_start_stage=None if args.no_auto_start else args.auto_start_stage,
        )
        fleet.spawn()

    if args.no_auto_start:
        log.info(
            "waiting for %d Isaac(s) to connect on ports %d..%d — click 'New Run' in each window.",
            cfg.n_envs, cfg.base_port, cfg.base_port + cfg.n_envs - 1,
        )
    else:
        log.info(
            "waiting for %d Isaac(s) to boot into stage %d and connect on ports %d..%d",
            cfg.n_envs, args.auto_start_stage, cfg.base_port, cfg.base_port + cfg.n_envs - 1,
        )

    # Hand off to the trainer. Its build_vec_env() will open the server sockets
    # and block until each Isaac connects. Ctrl-C during accept() surfaces as
    # KeyboardInterrupt which our signal handler catches.
    try:
        train(cfg)
    except KeyboardInterrupt:
        log.info("training interrupted by user")
    except Exception as e:
        log.exception("training failed: %s", e)
        return 1
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
