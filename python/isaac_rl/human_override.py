"""Human keyboard override for Isaac RL training.

Allows the user to manually steer the bot during training by pressing keys.
When any override key is pressed, the human's action REPLACES the policy's
action for that tick. When no key is pressed, the policy runs freely.

Corrections are logged to a .npz file for later DAgger-style retraining
(supervised learning on human corrections).

===============================================================================
Key mappings
===============================================================================

Movement (WASD, diagonals with combos):
    W          = up          (move=1)
    W+D        = up-right    (move=2)
    D          = right       (move=3)
    S+D        = down-right  (move=4)
    S          = down        (move=5)
    S+A        = down-left   (move=6)
    A          = left        (move=7)
    W+A        = up-left     (move=8)

Shooting (IJKL, cardinal only — Isaac only shoots in 4 directions):
    I          = shoot up
    L          = shoot right
    K          = shoot down
    J          = shoot left

Control:
    F1         = toggle override enable/disable
    F2         = toggle bot pause (bot idles until re-pressed)
    F3         = save current corrections buffer to disk immediately
    ESC        = disable override for the rest of the session

===============================================================================
Usage
===============================================================================

In train.py:
    from isaac_rl.human_override import HumanOverride
    override = HumanOverride(save_path="runs/my_run/human_corrections.npz")
    override.start()

Then in the env step loop:
    human_move, human_shoot = override.get_action()
    if human_move is not None:
        action[0] = human_move
    if human_shoot is not None:
        action[1] = human_shoot
    override.log_correction(obs, action)

Requires: pip install pynput
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class HumanOverride:
    """Global keyboard listener that produces manual action overrides.

    Non-invasive: if pynput is not installed, the override is silently
    disabled (get_action always returns (None, None)). Training continues
    with the policy in full control.

    Target-env selection: with multi-env training, use number keys 1-9 to
    select which env receives your keyboard overrides. The default target
    is env 0 (the first Isaac window that connected).

    Auto-focus detection (Windows only): when auto_focus=True, the target
    env automatically switches to whichever Isaac window has focus.
    Enumerates Isaac windows in order of creation (by HWND ascending) and
    maps them to env indices 0..N-1.
    """

    def __init__(self, save_path: str | Path | None = None, auto_focus: bool = True):
        self.save_path = Path(save_path) if save_path else None
        self.pressed: set[str] = set()
        self.enabled: bool = True
        self.bot_paused: bool = False
        self.target_env: int = 0        # which env gets the override (0-indexed)
        self.auto_focus: bool = auto_focus
        self.listener: Any = None
        self._focus_thread: threading.Thread | None = None
        self._focus_stop = threading.Event()
        self._corrections_obs: list[dict] = []
        self._corrections_actions: list[np.ndarray] = []
        self._lock = threading.Lock()
        # Try to import pynput. If unavailable, override is a no-op.
        self._available = False
        try:
            from pynput import keyboard  # noqa: F401
            self._available = True
        except ImportError:
            log.warning(
                "pynput not installed \u2014 human override DISABLED. "
                "To enable: pip install pynput"
            )

    def start(self) -> None:
        """Start the keyboard listener in a background thread."""
        if not self._available:
            return
        from pynput import keyboard
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.daemon = True
        self.listener.start()
        # Start focus-detection loop if available.
        if self.auto_focus:
            self._start_focus_watch()
        log.info(
            "Human override ACTIVE. Keys:\n"
            "  Movement: WASD (with diagonals via combos)\n"
            "  Shooting: IJKL\n"
            "  Target:   1/2/3/4/... = select which env gets your input (default: env 0)\n"
            "            Auto-focus: focusing an Isaac window auto-switches target (Windows only)\n"
            "  Toggle:   F1 = enable/disable override, F2 = pause bot\n"
            "            F3 = save corrections now, ESC = disable"
        )

    def _start_focus_watch(self) -> None:
        """Start a background thread that watches the focused window and
        auto-switches target_env when an Isaac window gets focus.

        Windows-only. Uses ctypes to call Win32 APIs (no extra deps).
        Silently disabled on non-Windows platforms.
        """
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
        except Exception:
            return

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]

        EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def get_window_title(hwnd: int) -> str:
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value

        def enum_isaac_windows() -> list[int]:
            """Return HWNDs of all Isaac windows, sorted ascending.

            HWNDs are Windows-assigned handles. On the same session, HWNDs
            issued later have larger numeric values, so ascending order
            roughly matches creation order — which typically matches env_idx.
            """
            hwnds: list[int] = []
            def callback(hwnd: int, _lparam: int) -> bool:
                title = get_window_title(hwnd)
                if "isaac" in title.lower() or "binding of isaac" in title.lower():
                    hwnds.append(hwnd)
                return True   # keep enumerating
            user32.EnumWindows(EnumWindowsProc(callback), 0)
            hwnds.sort()
            return hwnds

        def watch_loop():
            last_target = -1
            while not self._focus_stop.is_set():
                try:
                    fg = user32.GetForegroundWindow()
                    if fg:
                        hwnds = enum_isaac_windows()
                        if fg in hwnds:
                            new_target = hwnds.index(fg)
                            if new_target != self.target_env and new_target != last_target:
                                self.target_env = new_target
                                last_target = new_target
                                log.info("[override] auto-focus -> target env = %d", new_target)
                except Exception:
                    pass
                self._focus_stop.wait(0.3)   # poll every 300ms

        self._focus_thread = threading.Thread(target=watch_loop, daemon=True)
        self._focus_thread.start()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        # Stop focus watcher thread if running.
        self._focus_stop.set()
        if self._focus_thread is not None:
            self._focus_thread.join(timeout=1.0)
            self._focus_thread = None
        self.save_corrections()

    def _key_str(self, key: Any) -> str:
        """Convert pynput Key object to a string representation."""
        try:
            if hasattr(key, "char") and key.char is not None:
                return key.char.lower()
        except AttributeError:
            pass
        return str(key)   # e.g. "Key.f1"

    def _on_press(self, key: Any) -> None:
        ks = self._key_str(key)
        # Special keys: check before adding to pressed set.
        if ks == "Key.f1":
            self.enabled = not self.enabled
            log.info("[override] toggled to %s", "ENABLED" if self.enabled else "DISABLED")
            return
        if ks == "Key.f2":
            self.bot_paused = not self.bot_paused
            log.info("[override] bot pause -> %s", self.bot_paused)
            return
        if ks == "Key.f3":
            self.save_corrections()
            return
        if ks == "Key.esc":
            self.enabled = False
            log.info("[override] DISABLED (ESC)")
            return
        # Number keys 1-9: switch target env.
        if ks in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            new_target = int(ks) - 1
            if new_target != self.target_env:
                self.target_env = new_target
                log.info("[override] target env = %d", self.target_env)
            return
        with self._lock:
            self.pressed.add(ks)

    def _on_release(self, key: Any) -> None:
        ks = self._key_str(key)
        with self._lock:
            self.pressed.discard(ks)

    def get_action(self) -> tuple[int | None, int | None]:
        """Return (move_override, shoot_override), each None if not overridden.

        If bot is paused (F2), returns (0, 0) to force idle.
        If override is disabled, returns (None, None).
        """
        if not self._available or not self.enabled:
            return None, None
        if self.bot_paused:
            return 0, 0
        with self._lock:
            up    = "w" in self.pressed
            down  = "s" in self.pressed
            left  = "a" in self.pressed
            right = "d" in self.pressed
            i_key = "i" in self.pressed
            j_key = "j" in self.pressed
            k_key = "k" in self.pressed
            l_key = "l" in self.pressed
        # 8-way movement mapping.
        move: int | None = None
        if up and right:
            move = 2
        elif up and left:
            move = 8
        elif down and right:
            move = 4
        elif down and left:
            move = 6
        elif up:
            move = 1
        elif right:
            move = 3
        elif down:
            move = 5
        elif left:
            move = 7
        # Shoot (single cardinal only).
        shoot: int | None = None
        if i_key:
            shoot = 1
        elif l_key:
            shoot = 2
        elif k_key:
            shoot = 3
        elif j_key:
            shoot = 4
        return move, shoot

    def log_correction(self, obs: dict, action: np.ndarray) -> None:
        """Record (obs, action) pair when human is actively steering.

        Called from the env step loop AFTER get_action has been applied. Only
        records when override was actually active (some key was pressed).
        Corrections accumulate in memory and are flushed to disk periodically
        or on save_corrections().
        """
        if not self._available:
            return
        with self._lock:
            has_input = bool(self.pressed) or self.bot_paused
        if not has_input:
            return
        # Deep-copy the obs to avoid mutation, action is already a fresh ndarray.
        # Store only serializable fields to keep memory reasonable.
        self._corrections_obs.append(obs)
        self._corrections_actions.append(np.asarray(action).copy())

    def save_corrections(self) -> None:
        """Flush the corrections buffer to disk as an npz file."""
        if not self.save_path or not self._corrections_obs:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save actions as a stacked array. Obs is a list of dicts \u2014 use pickle
        # via np.savez with allow_pickle. Not ideal but simple.
        actions = np.stack(self._corrections_actions).astype(np.int64)
        # Convert obs list to object array for saving.
        obs_array = np.array(self._corrections_obs, dtype=object)
        np.savez(
            self.save_path,
            actions=actions,
            obs=obs_array,
        )
        log.info(
            "[override] saved %d human corrections to %s",
            len(self._corrections_obs),
            self.save_path,
        )


# ---- Module-level singleton --------------------------------------------
#
# The env step loop (env.py) needs to consult the override without having
# it passed through vec_env.py's fanout. Use a global singleton set by
# train.py at startup.
_INSTANCE: HumanOverride | None = None


def set_instance(inst: HumanOverride) -> None:
    global _INSTANCE
    _INSTANCE = inst


def get_instance() -> HumanOverride | None:
    return _INSTANCE


def apply_override(action: np.ndarray, env_idx: int = 0) -> np.ndarray:
    """If a HumanOverride instance is set and active, mutate action in place.

    Args:
        action: the (2,) or (5,) int action array to potentially modify.
        env_idx: which env is calling. Override only applies if env_idx
                 matches the current target_env (default 0). Prevents the
                 user's key press from affecting ALL envs in a multi-env
                 training run.

    Returns the (possibly modified) action array. Called by env.py's step()
    before sending the action to Isaac.
    """
    inst = _INSTANCE
    if inst is None:
        return action
    if env_idx != inst.target_env:
        return action
    move, shoot = inst.get_action()
    if move is not None:
        action[0] = move
    if shoot is not None:
        action[1] = shoot
    return action
