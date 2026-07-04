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
    """

    def __init__(self, save_path: str | Path | None = None):
        self.save_path = Path(save_path) if save_path else None
        self.pressed: set[str] = set()
        self.enabled: bool = True
        self.bot_paused: bool = False
        self.target_env: int = 0        # which env gets the override (0-indexed)
        self.listener: Any = None
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
        log.info(
            "Human override ACTIVE. Keys:\n"
            "  Movement: WASD (with diagonals via combos)\n"
            "  Shooting: IJKL\n"
            "  Target:   1/2/3/4/... = select which env gets your input (default: env 0)\n"
            "  Toggle:   F1 = enable/disable override, F2 = pause bot\n"
            "            F3 = save corrections now, ESC = disable"
        )

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
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
