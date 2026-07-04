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
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make sure the trainer package is importable no matter where you run this from.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))

from isaac_rl.env import register_on_crash  # noqa: E402
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

    def _install_dir(self) -> Path:
        return Path(self.binary).resolve().parent

    def _ensure_steam_appid(self) -> None:
        """Drop steam_appid.txt next to isaac-ng.exe.

        Without this file, Repentance's DRM stub tries to relaunch under Steam
        and both launches die with exit code 53. The file's presence is the
        standard 'external launch' signal for Steamworks games.
        """
        appid = self._install_dir() / "steam_appid.txt"
        if not appid.exists():
            try:
                appid.write_text("250900\n", encoding="utf-8")
                log.info("wrote %s (required for external launch)", appid)
            except OSError as e:
                log.warning(
                    "could not write %s (%s). Isaac may exit immediately. "
                    "Run this shell as Administrator once, OR create the file manually with the single line: 250900",
                    appid, e,
                )

    def spawn(self, stagger_s: float | None = None) -> None:
        """Spawn Isaac processes with stagger.

        stagger_s defaults to 3s for small fleets (<=4 envs) and scales up to
        6s for larger fleets. Isaac's mod compilation + D3D init is CPU-heavy
        during startup; too-fast staggers cause CPU saturation and some
        instances fail to reach the RL-bridge init code within the trainer's
        accept timeout.
        """
        if stagger_s is None:
            # Auto-scale stagger: 3s for <=4 envs, +1s per additional env, cap at 8s.
            stagger_s = min(8.0, 3.0 + max(0, self.n_envs - 4))
        self._ensure_steam_appid()
        # CPU-count sanity check: warn if user is over-subscribing cores.
        cpu_count = os.cpu_count() or 4
        safe_max = max(1, cpu_count - 1)
        if self.n_envs > safe_max:
            log.warning(
                "n_envs=%d exceeds recommended max (%d = cpu_count - 1 = %d - 1). "
                "Isaac processes may fail to start within accept_timeout. "
                "If you see 'Isaac did not connect on port ... within 300s' errors, "
                "reduce --override n_envs to %d or lower.",
                self.n_envs, safe_max, cpu_count, safe_max,
            )
        log.info("spawning %d Isaac processes with %.1fs stagger (cpu_count=%d)",
                 self.n_envs, stagger_s, cpu_count)
        for i in range(self.n_envs):
            self._launch_one(i)
            time.sleep(stagger_s)

    def _launch_one(self, i: int) -> None:
        """Spawn Isaac child at index i. Used by both initial spawn and respawn."""
        install_dir = self._install_dir()
        port = self.base_port + i
        env = os.environ.copy()
        env["ISAAC_RL_PORT"] = str(port)

        cmd = [self.binary, "--luadebug"]
        if self.auto_start_stage is not None:
            cmd += [f"--set-stage={self.auto_start_stage}"]
        cmd += self.extra_args

        log.info("[isaac %d/%d] port=%d cwd=%s cmd=%s",
                 i + 1, self.n_envs, port, install_dir, " ".join(cmd))

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0)
            )

        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(install_dir),
            creationflags=creationflags,
        )
        # Keep self.procs indexed by slot i. Grow the list on first spawn,
        # replace in place on respawn.
        while len(self.procs) <= i:
            self.procs.append(proc)
        self.procs[i] = proc

    def _capture_isaac_log(self, port: int) -> None:
        """Copy Isaac's log.txt to a timestamped file on the user's Desktop.

        Isaac truncates log.txt on every launch, which means the moment we
        respawn a fresh Isaac we lose all evidence of what the crashed one
        was doing. Capture the log BEFORE respawn so its available for
        post-mortem analysis. Best-effort — never raises.

        Written to: %USERPROFILE%/Desktop/isaac_crash_<port>_<timestamp>.txt
        """
        try:
            home = os.path.expanduser("~")
            src = os.path.join(
                home, "Documents", "My Games",
                "Binding of Isaac Repentance", "log.txt",
            )
            if not os.path.exists(src):
                # Try Rebirth path as fallback (non-Repentance installs).
                src = os.path.join(
                    home, "Documents", "My Games",
                    "Binding of Isaac Rebirth", "log.txt",
                )
                if not os.path.exists(src):
                    return
            desktop = os.path.join(home, "Desktop")
            os.makedirs(desktop, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(desktop, f"isaac_crash_{port}_{ts}.txt")
            shutil.copyfile(src, dst)
            log.info("respawn(port=%d): saved crashed Isaac's log to %s", port, dst)
        except Exception as e:
            log.warning("respawn(port=%d): failed to capture Isaac log: %s", port, e)

    def respawn(self, port: int) -> None:
        """Kill the Isaac child owning `port` (if any) and launch a fresh one.

        Registered as the SocketIsaacEnv on-crash callback so a dying Isaac
        gets automatically replaced without human intervention.
        """
        i = port - self.base_port
        if i < 0 or i >= self.n_envs:
            log.error("respawn(port=%d): port not owned by this fleet (base=%d, n=%d)",
                      port, self.base_port, self.n_envs)
            return

        # Capture the crashed Isaacs log BEFORE anything overwrites it.
        self._capture_isaac_log(port)

        # Kill the current occupant (may already be dead if it crashed on its own).
        if i < len(self.procs):
            old = self.procs[i]
            if old.poll() is None:
                log.info("respawn(port=%d): terminating existing pid %d", port, old.pid)
                try:
                    old.terminate()
                    try:
                        old.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        old.kill()
                        old.wait(timeout=5)
                except Exception as e:
                    log.warning("respawn(port=%d): terminate failed: %s", port, e)
            else:
                log.info("respawn(port=%d): existing child already exited (rc=%s)", port, old.returncode)

        # Isaac holds a singleton lock file for a few seconds after exit. If we
        # relaunch too quickly the new process either fails silently or connects
        # its socket and then instantly RSTs during boot (WinError 10054 on the
        # trainer side). Sleep past that window before spawning.
        #
        # 3 seconds turned out to be insufficient on some Windows / Steam
        # combinations — Steam's DRM stub can hold state 5-10s after the child
        # process exits, and a --set-stage=N fast-path launch during that
        # window makes the fresh Isaac exit cleanly during boot (before mods
        # are loaded; log.txt ends at the version banner). Bumped to 10s to
        # give the OS + Steam plenty of time to release everything.
        time.sleep(10.0)

        log.info("respawn(port=%d): launching replacement Isaac", port)
        self._launch_one(i)

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
                    help="Stage passed to --set-stage on first launch (default: 1 = boot straight into Basement 1). "
                         "Set to 0 to skip the flag entirely (slower boot, uses the mod's menu-auto-start).")
    ap.add_argument("--tensorboard", action="store_true", help="Also start TensorBoard in the background at :6006")
    ap.add_argument("--resume", type=str, default=None, metavar="CKPT.pt",
                    help="Resume training from a checkpoint. Point at a .pt file (e.g. runs/<run>/<timestamp>/latest.pt "
                         "or runs/<run>/<timestamp>/ckpts/step_1000000.pt). Loads policy, RND, optimizer, and step count.")
    ap.add_argument("--collect-demos", type=int, default=None, metavar="N",
                    help="Run the heuristic policy for N steps (across all envs) and save the (obs, action) "
                         "trajectories to runs/demos/<timestamp>.npz. Exits after collection unless --bc-epochs "
                         "is also provided (in which case BC pretraining runs on the fresh demos, then PPO starts).")
    ap.add_argument("--bc-pretrain-file", type=str, default=None, metavar="NPZ",
                    help="Path to a demos .npz file. Runs supervised BC pretraining on the policy network before "
                         "starting PPO. Compatible with --resume (BC applies to the resumed weights).")
    ap.add_argument("--bc-epochs", type=int, default=10,
                    help="Number of BC pretraining epochs (default 10). Ignored if no demos are involved.")
    ap.add_argument("--bc-batch-size", type=int, default=256,
                    help="BC minibatch size (default 256).")
    ap.add_argument("--bc-lr", type=float, default=3.0e-4,
                    help="BC learning rate (default 3e-4).")
    ap.add_argument("--override", nargs="*", default=[], help="Extra config overrides: key=value")
    ap.add_argument(
        "--human-override", action="store_true",
        help="Enable keyboard override to manually steer the bot during training. "
             "Movement: WASD (diagonals via combos). Shoot: IJKL. "
             "F1=toggle enable, F2=pause bot, F3=save corrections, ESC=disable. "
             "Requires: pip install pynput. Human corrections saved to "
             "runs/<name>/human_corrections.npz for later DAgger retraining."
    )
    ap.add_argument(
        "--debug-heuristic", action="store_true",
        help="Record every heuristic decision to a JSONL file for post-hoc "
             "analysis. Output: runs/<name>/heuristic_debug.jsonl. "
             "Equivalent to ISAAC_HEURISTIC_DEBUG=1 env var."
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load and reconcile config.
    cfg = _cfg_from_yaml(args.config)
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.base_port is not None:
        cfg.base_port = args.base_port
    if args.resume is not None:
        cfg.resume_from = args.resume
        log.info("will resume training from checkpoint: %s", args.resume)

    # Wire BC / demo-collection args onto the config so ppo.py can act on them.
    cfg.collect_demos_n = args.collect_demos
    cfg.bc_pretrain_file = args.bc_pretrain_file
    cfg.bc_epochs = args.bc_epochs
    cfg.bc_batch_size = args.bc_batch_size
    cfg.bc_lr = args.bc_lr
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
        # Wire the fleet's respawn method into each env's crash hook BEFORE we
        # call train() (which builds the SocketIsaacEnvs). When any Isaac dies
        # mid-training its env will invoke this callback, we'll kill the zombie
        # process and launch a fresh Isaac on the same port, the env re-accepts
        # the new socket, and training resumes with terminated=True on that env.
        for i in range(cfg.n_envs):
            register_on_crash(cfg.base_port + i, fleet.respawn)
        fleet.spawn()

    if args.no_auto_start:
        log.info(
            "waiting for %d Isaac(s) to connect on ports %d..%d — click 'New Run' in each window.",
            cfg.n_envs, cfg.base_port, cfg.base_port + cfg.n_envs - 1,
        )
    elif args.auto_start_stage:
        log.info(
            "waiting for %d Isaac(s) to boot into stage %d and connect on ports %d..%d",
            cfg.n_envs, args.auto_start_stage, cfg.base_port, cfg.base_port + cfg.n_envs - 1,
        )
    else:
        log.info(
            "waiting for %d Isaac(s) to boot (menu auto-start via mod) and connect on ports %d..%d",
            cfg.n_envs, cfg.base_port, cfg.base_port + cfg.n_envs - 1,
        )

    # Hand off to the trainer. Its build_vec_env() will open the server sockets
    # and block until each Isaac connects. Ctrl-C during accept() surfaces as
    # KeyboardInterrupt which our signal handler catches.

    # Set up human keyboard override if requested. Its listener runs in a
    # background thread; env.py's step() will consult it via the singleton
    # accessor. If pynput isn't installed, it silently becomes a no-op.
    human_override = None
    if args.human_override:
        from isaac_rl.human_override import HumanOverride, set_instance
        # Save corrections to the run directory.
        run_dir = getattr(cfg, "run_dir", None) or "runs"
        save_path = os.path.join(run_dir, "human_corrections.npz")
        human_override = HumanOverride(save_path=save_path)
        human_override.start()
        set_instance(human_override)

    # Diagnostic recorder: if ISAAC_HEURISTIC_DEBUG=1 OR --debug-heuristic
    # flag, log every heuristic decision to a JSONL file for post-hoc
    # analysis. Non-invasive; off by default. See
    # python/isaac_rl/debug_recorder.py.
    debug_recorder = None
    debug_enabled = args.debug_heuristic or bool(
        os.environ.get("ISAAC_HEURISTIC_DEBUG", "").strip()
    )
    if debug_enabled:
        from isaac_rl.debug_recorder import DebugRecorder
        # Use a stable path under checkpoint_dir/run_name so user can find it
        # without knowing the timestamped subfolder. Timestamp appended so
        # multiple runs don't overwrite each other.
        ts = time.strftime("%Y%m%d-%H%M%S")
        ckpt_dir = getattr(cfg, "checkpoint_dir", None) or "runs"
        run_name = getattr(cfg, "run_name", None) or "default"
        debug_path = os.path.abspath(
            os.path.join(ckpt_dir, run_name, f"heuristic_debug_{ts}.jsonl")
        )
        debug_recorder = DebugRecorder(save_path=debug_path, enabled=True, flush_every=100)
        DebugRecorder.set_instance(debug_recorder)
        log.info("=" * 72)
        log.info("HEURISTIC DEBUG RECORDER ACTIVE")
        log.info("  Path: %s", debug_path)
        log.info("  Format: JSONL, one line per heuristic decision")
        log.info("  Flush: every 100 ticks (~7s at 15Hz)")
        log.info("=" * 72)

    try:
        train(cfg)
    except KeyboardInterrupt:
        log.info("training interrupted by user")
    except Exception as e:
        log.exception("training failed: %s", e)
        return 1
    finally:
        if human_override is not None:
            human_override.stop()
        if debug_recorder is not None:
            debug_recorder.close()
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
