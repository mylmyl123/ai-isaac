"""Isaac RL entry point (post 2026-07-13 reset).

Launches the Isaac fleet + CleanRL PPO trainer with one command.

Usage:
    python train.py --config configs/curriculum.yaml
    python train.py --config configs/curriculum.yaml --isaac "C:\\path\\to\\isaac-ng.exe"
    python train.py --config configs/curriculum.yaml --override stage=B run_name=cleanrl_ppo_stageB

Design:
    * Reads a YAML into a PPOConfig dataclass (via cfg_from_yaml).
    * Auto-detects Isaac binary if --isaac not given.
    * Writes steam_appid.txt next to isaac-ng.exe (needed for external launch).
    * Spawns N Isaac processes with the right cwd, ISAAC_RL_PORT, and
      ISAAC_RL_STAGE=<A|B|C|D|E>.
    * Passes --set-stage=1 so Isaac boots directly into Basement 1 (skips
      menu).
    * Registers a per-port respawn callback so a crashed Isaac auto-restarts.
    * Optionally starts TensorBoard at :6006.
    * Calls the CleanRL PPO trainer.
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

# Ensure the local isaac_rl package is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from isaac_rl.cleanrl_ppo import PPOConfig, train as ppo_train   # noqa: E402
from isaac_rl.env import register_on_crash                       # noqa: E402
from isaac_rl.vec_env import build_vec_env                       # noqa: E402
from isaac_rl.reward import RewardConfig                         # noqa: E402


log = logging.getLogger("train")


# ------------------------------------------------------------------- config


def cfg_from_yaml(path: str) -> PPOConfig:
    """Load YAML and merge into PPOConfig defaults."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = PPOConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            log.warning("config key ignored (unknown to PPOConfig): %s", k)
    return cfg


# ------------------------------------------------------------------- Isaac binary


DEFAULT_ISAAC_BINARIES_WINDOWS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"E:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
]
DEFAULT_ISAAC_BINARIES_LINUX = [
    str(Path.home() / ".steam/steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
]
DEFAULT_ISAAC_BINARIES_DARWIN = [
    "/Applications/The Binding of Isaac Rebirth.app/Contents/MacOS/isaac-ng",
]


def resolve_isaac_binary(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    system = platform.system()
    lst = {
        "Windows": DEFAULT_ISAAC_BINARIES_WINDOWS,
        "Linux": DEFAULT_ISAAC_BINARIES_LINUX,
        "Darwin": DEFAULT_ISAAC_BINARIES_DARWIN,
    }.get(system, [])
    for candidate in lst:
        if Path(candidate).exists():
            log.info("auto-detected isaac binary: %s", candidate)
            return candidate
    return None


# ------------------------------------------------------------------- Isaac fleet


class IsaacFleet:
    """Spawns + supervises N Isaac child processes.

    - cwd = binary's directory (Isaac reads resources/ relative to cwd)
    - --set-stage=1 -> direct-boot into Basement 1 (skips menu)
    - ISAAC_RL_PORT=<port> per child
    - ISAAC_RL_STAGE=<stage letter> for the mod's curriculum
    - CREATE_NEW_CONSOLE on Windows so Ctrl+C in the parent doesn't hit them
    - Auto-respawn on crash via register_on_crash callbacks
    """

    def __init__(self, binary: str, base_port: int, n_envs: int, stage: str):
        self.binary = binary
        self.base_port = base_port
        self.n_envs = n_envs
        self.stage = stage.upper()
        self.procs: list[subprocess.Popen] = []

    def _install_dir(self) -> Path:
        return Path(self.binary).resolve().parent

    def _ensure_steam_appid(self) -> None:
        appid = self._install_dir() / "steam_appid.txt"
        if not appid.exists():
            try:
                appid.write_text("250900\n", encoding="utf-8")
                log.info("wrote %s (required for external launch)", appid)
            except OSError as e:
                log.warning(
                    "could not write %s (%s). If Isaac exits immediately, "
                    "run as Administrator once or create the file manually "
                    "with the line: 250900", appid, e,
                )

    def spawn(self) -> None:
        self._ensure_steam_appid()
        stagger = min(6.0, 3.0 + max(0, self.n_envs - 4))
        cpu_count = os.cpu_count() or 4
        safe_max = max(1, cpu_count - 1)
        if self.n_envs > safe_max:
            log.warning(
                "n_envs=%d > safe_max=%d (cpu_count-1). Isaac may fail to "
                "reach socket accept within timeout — consider n_envs=%d.",
                self.n_envs, safe_max, safe_max,
            )
        log.info("spawning %d Isaacs with %.1fs stagger (stage=%s)",
                 self.n_envs, stagger, self.stage)
        for i in range(self.n_envs):
            self._launch_one(i)
            time.sleep(stagger)

    def _launch_one(self, i: int) -> None:
        install_dir = self._install_dir()
        port = self.base_port + i
        env = os.environ.copy()
        env["ISAAC_RL_PORT"] = str(port)
        env["ISAAC_RL_STAGE"] = self.stage

        cmd = [self.binary, "--luadebug", "--set-stage=1"]
        log.info("[isaac %d/%d] port=%d stage=%s cwd=%s",
                 i + 1, self.n_envs, port, self.stage, install_dir)

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0)
            )

        proc = subprocess.Popen(cmd, env=env, cwd=str(install_dir), creationflags=creationflags)
        while len(self.procs) <= i:
            self.procs.append(proc)
        self.procs[i] = proc

    def respawn(self, port: int) -> None:
        i = port - self.base_port
        if not (0 <= i < self.n_envs):
            log.warning("respawn called with unknown port %d — skipping", port)
            return
        old = self.procs[i]
        if old.poll() is None:
            try:
                old.terminate()
                old.wait(timeout=5)
            except Exception:
                try:
                    old.kill()
                except Exception:
                    pass
        log.info("respawning isaac on port %d", port)
        self._launch_one(i)

    def shutdown(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# ------------------------------------------------------------------- TensorBoard


def maybe_start_tensorboard(logdir: Path, port: int = 6006) -> subprocess.Popen | None:
    try:
        proc = subprocess.Popen(
            ["tensorboard", "--logdir", str(logdir), "--port", str(port), "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("tensorboard: http://localhost:%d  (logdir=%s)", port, logdir)
        return proc
    except FileNotFoundError:
        log.warning("tensorboard not on PATH; skipping. `pip install tensorboard` to enable.")
        return None


# ------------------------------------------------------------------- main


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--isaac", default=None, help="Absolute path to isaac-ng.exe")
    ap.add_argument("--no-launch-isaac", action="store_true", help="Don't spawn Isaac; expect it manually")
    ap.add_argument("--tensorboard", action="store_true", help="Start TB at :6006")
    ap.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    args = ap.parse_args()

    cfg = cfg_from_yaml(args.config)
    for kv in args.override:
        k, _, v = kv.partition("=")
        try:
            v_typed: object = int(v)
        except ValueError:
            try:
                v_typed = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v_typed = (v.lower() == "true")
                else:
                    v_typed = v
        if hasattr(cfg, k):
            setattr(cfg, k, v_typed)
        else:
            log.warning("override key ignored (unknown to PPOConfig): %s", k)

    log.info("config: %s", asdict(cfg))

    launch = not args.no_launch_isaac
    fleet: IsaacFleet | None = None
    tb_proc: subprocess.Popen | None = None

    def cleanup():
        if fleet is not None:
            fleet.shutdown()
        if tb_proc is not None and tb_proc.poll() is None:
            tb_proc.terminate()

    atexit.register(cleanup)

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
        binary = resolve_isaac_binary(args.isaac)
        if not binary:
            log.error("could not find Isaac binary. Pass --isaac <path> or use --no-launch-isaac.")
            return 2
        fleet = IsaacFleet(binary=binary, base_port=cfg.base_port, n_envs=cfg.n_envs, stage=cfg.stage)
        for i in range(cfg.n_envs):
            register_on_crash(cfg.base_port + i, fleet.respawn)
        fleet.spawn()

    log.info("waiting for %d Isaac(s) to connect on ports %d..%d",
             cfg.n_envs, cfg.base_port, cfg.base_port + cfg.n_envs - 1)

    # ---- Build the vec env (accepts sockets, does handshake) ----
    env = build_vec_env(
        n_envs=cfg.n_envs,
        base_port=cfg.base_port,
        reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps,
        launch_isaac=False,               # WE own the fleet, not vec_env
        reward_config=RewardConfig(),     # 3-term reward
    )

    try:
        ppo_train(cfg, env)
    finally:
        env.close()
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
