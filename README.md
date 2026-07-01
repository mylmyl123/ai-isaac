# Isaac RL

Reinforcement learning agent that learns to play *The Binding of Isaac: Repentance* well enough to beat Mom (end of Depths 2). Full design is in `/Users/I048254/.claude/plans/glittery-foraging-goose.md`.

## Layout

```
mods/isaac-rl-bridge/     # Lua mod: extracts state, sends over TCP, injects action
  main.lua                # callback registration, action injection
  obs.lua                 # per-tick observation builder
  net.lua                 # LuaSocket wrapper (length-prefixed frames)
  metadata.xml            # mod manifest

python/isaac_rl/          # Python trainer
  env.py                  # SocketIsaacEnv(gymnasium.Env)
  spaces.py               # Dict observation / MultiDiscrete action space
  protocol.py             # length-prefixed JSON frame helpers

tools/launch_isaac.py     # Cross-platform Isaac launcher (adds --luadebug)
tests/                    # Pure-Python tests (no live Isaac needed)
```

## M1 verification recipe

Goal: prove the Lua ↔ Python socket loop is stable and actions actually reach the game.

1. **One-time setup.**
   - Install *The Binding of Isaac: Repentance* via Steam.
   - Confirm LuaSocket is available in the Isaac Lua sandbox by launching Isaac with `--luadebug` and running a hello-world mod that does `print(require("socket"))`. Repentance ships with `socket` on Windows/Linux native; on Mac you may need to drop `socket.so` alongside the binary.
   - Copy `mods/isaac-rl-bridge/` into your Isaac mods directory. Locations:
     - Windows: `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\mods\`
     - Linux: `~/.local/share/binding of isaac repentance/mods/`
     - macOS: `~/Library/Application Support/Binding of Isaac Repentance/mods/`
   - Enable the mod inside the game's Mods menu.

2. **Python side.**

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   PYTHONPATH=python pytest tests/      # offline sanity — no Isaac needed
   ```

3. **End-to-end smoke test.**

   Terminal A — start the trainer's server first:

   ```bash
   PYTHONPATH=python python -m isaac_rl.env --port 9500 --steps 1000
   ```

   Terminal B — launch Isaac with `--luadebug` and the port env var:

   ```bash
   python tools/launch_isaac.py --port 9500
   ```

   In Isaac, start any run. You should see:
   - Terminal A: `Isaac connected from ...`, `handshake: {'hello': True, 'seed': ...}`, then step logs at ~15 Hz.
   - Isaac window: character moves/shoots randomly.

## Configuration knobs

- `ISAAC_RL_PORT` — port the mod connects to (default 9500).
- Frame skip (control rate) — top of `mods/isaac-rl-bridge/main.lua`. `FRAME_SKIP = 2` gives 15 Hz; `1` gives 30 Hz.

## What's *not* here yet

M1 covers only the socket loop and a minimal player/global observation. Milestones from the plan:

- M2 (single-room combat): entities, projectiles, room grid in obs; reward shaping; PPO trainer.
- M3+ (floor clear onward): curriculum, RND, recurrent policy, vec-env launcher.

Each of these lands as its own PR-sized change.
