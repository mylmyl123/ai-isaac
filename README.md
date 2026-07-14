# Isaac RL

Reinforcement learning agent that learns to play *The Binding of Isaac: Repentance*.

**Status (2026-07-13)**: nuclear reset. The previous training layer (custom PPO + vendored DreamerV3 + 51-term reward function + audit-doc backlog) has been removed and replaced with a minimal, verifiable stack:

- **3-term reward**: `r_kill=+1`, `r_death=-1`, `r_step=-0.001`. That's it.
- **CleanRL-style PPO**: single-file, ~400 lines, MultiDiscrete-aware, no framework abstractions.
- **Curriculum via mod env-var**: stages A → E, one YAML config, one flag.

The Isaac mod (Lua) + Python env wrapper + obs/action schema + socket protocol are **unchanged**. They're the well-tested foundation this project has been built on.

## Repo layout

```
mods/isaac-rl-bridge/       Lua mod running inside Isaac (unchanged)
  main.lua                  callbacks, action injection, stage curriculum
  obs.lua                   per-tick observation builder
  reward.lua                event stream from damage/pickup callbacks
  net.lua                   LuaSocket wrapper (length-prefix frames)
  tables.lua                dense-index tables (collectibles, NPC types)
  metadata.xml              mod manifest

python/isaac_rl/
  cleanrl_ppo.py            NEW: CleanRL-style PPO trainer + policy net
  reward.py                 3-term reward shaper (was 51 terms, deleted)
  env.py                    SocketIsaacEnv(gymnasium.Env) — unchanged
  vec_env.py                SyncVecEnv + Isaac process launcher
  spaces.py                 Dict obs + MultiDiscrete action space
  protocol.py               framed JSON helpers
  heuristic.py              scripted-policy baseline
  bc.py                     Behavior cloning (BC pretraining from heuristic demos)
  eval.py                   Deterministic evaluation harness
  human_override.py         Keyboard teleop for DAgger corrections
  record.py                 Human-play recording mode
  debug_recorder.py         Post-hoc debug logging

train.py                    Repo-root entry point (spawns Isaac fleet + PPO)
configs/curriculum.yaml     Single training config
tools/                      Auxiliary scripts
scripts/                    Run helpers (push_data.ps1, etc.)
tests/                      pytest suite (50 tests, all green)
```

## Prerequisites

- Windows 10/11 with **Repentance DLC** installed
- Python 3.10+ with a virtual environment
- Steam running (or set `steam_appid.txt` — the launcher does this automatically)

Install:

```powershell
cd C:\path\to\isaac-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Copy the mod into Isaac's user mods folder:
$src = ".\mods\isaac-rl-bridge"
$dst = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\mods\isaac-rl-bridge"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Copy-Item -Recurse $src $dst
```

Then launch Isaac once via Steam → Mods → confirm `isaac-rl-bridge` has a green check.

## Training

**One command:**

```powershell
python train.py --config configs\curriculum.yaml --tensorboard
```

That's it. The trainer:

1. Auto-detects your Isaac binary
2. Writes `steam_appid.txt` next to `isaac-ng.exe`
3. Spawns 2 Isaac processes with the correct cwd, `--set-stage=1` (direct boot), `ISAAC_RL_PORT`, and `ISAAC_RL_STAGE`
4. Registers per-port crash-respawn callbacks
5. Starts TensorBoard at `http://localhost:6006`
6. Runs the PPO training loop

## Curriculum stages

Set `stage: A|B|C|D|E` in the YAML config, or override on CLI:

```powershell
python train.py --config configs\curriculum.yaml --override stage=B run_name=cleanrl_ppo_stageB
```

| Stage | Env                                     | Compute (2 envs, 3060 Ti) |
|-------|------------------------------------------|---------------------------|
| A     | Sealed room, 1 attack fly, respawn on kill | ~1 hr |
| B     | Sealed room, 3 attack flies, respawn on room-clear | ~2 hr |
| C     | Normal starting room, unsealed, 1 fly (mod wipes rest) | ~4 hr |
| D     | Normal Basement 1 room, vanilla enemies (mod passive) | ~8 hr |
| E     | Full Basement 1 run, no restrictions (mod fully passive) | ~16 hr |

Each stage warm-starts from the previous stage's checkpoint (via `--resume runs/.../latest.pt` — planned, not yet wired).

Success criterion for each stage: `charts/kills_mean` on TensorBoard climbs above the random baseline within the budget, on 3+ seeds.

## Baselines for the paper

Run these in order (each ~1-2 hours on Stage A):

1. **Random policy**: pass a bogus config with `lr=0.0` and no updates — TB records the random baseline.
2. **Heuristic**: `python -m isaac_rl.eval --policy heuristic --stage A` (scripted-policy floor).
3. **CleanRL PPO**: the default `train.py` run above.

Compare `charts/kills_mean` and `charts/ep_r_mean` across the three. If PPO doesn't clearly beat both, the pipeline still has a bug.

## Tests

```powershell
$env:PYTHONPATH = "python"
python -m pytest tests/ -q
```

Fifty tests, all green. Covers obs round-trip, heuristic policy, reward shaper, PPO network shapes, GAE computation.

## Data collection

After a training run finishes (or you Ctrl+C):

```powershell
.\scripts\push_data.ps1
```

Exports the TB scalars to a small JSON, copies `latest.pt`, commits, and pushes.

## Design notes

**Why 3 rewards?** Because 51 rewards was un-debuggable. Every training failure had 51 plausible root causes. With 3 rewards the signal-to-noise is trivially interpretable.

**Why CleanRL PPO?** Because reviewers know it, it's verified on many envs, and if it doesn't learn Isaac the bug is provably not in the RL algorithm. The vendored DreamerV3 port didn't have that reputation.

**Why curriculum?** Because starting on the full game with a random policy is a needle-in-a-haystack exploration problem. Curriculum lets each stage bootstrap the next.

**Why keep the mod?** Because it's 900+ lines of hard-won engineering — Isaac's Lua API has undocumented callbacks, C++ pointer safety issues on death, socket protocol quirks. Rewriting it would take months and just rediscover the same edge cases.
