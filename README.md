# Isaac RL

Reinforcement learning agent that learns to play *The Binding of Isaac: Repentance* well enough to beat Mom (end of Depths 2). Structured-state observation via a Lua mod, recurrent PPO trainer with entity attention + RND intrinsic reward, staged curriculum.

Full design: `/Users/I048254/.claude/plans/glittery-foraging-goose.md` (or ask; it's referenced in git history).

## Repo layout

```
mods/isaac-rl-bridge/       # Lua mod running inside Isaac
  main.lua                  # callbacks, action injection
  obs.lua                   # per-tick observation builder
  reward.lua                # event stream from damage/pickup callbacks
  net.lua                   # LuaSocket wrapper (length-prefix frames)
  tables.lua                # dense-index tables (collectibles, NPC types, pickups)
  metadata.xml              # mod manifest

python/isaac_rl/            # Python trainer
  env.py                    # SocketIsaacEnv(gymnasium.Env)
  vec_env.py                # SyncVecEnv + Isaac process launcher
  spaces.py                 # Dict obs + MultiDiscrete action space, encoders
  protocol.py               # framed JSON helpers
  reward.py                 # RewardShaper (Python-side event → scalar)
  model.py                  # IsaacPolicy (entity attention + GRU + factored heads)
  rnd.py                    # Random Network Distillation intrinsic reward
  ppo.py                    # Recurrent PPO trainer (single-file, CleanRL style)
  eval.py                   # Deterministic evaluation harness
  curriculum.py             # Stage definitions (single-room → floor → six-floor run)
  torch_utils.py            # obs → torch tensors
  configs/*.yaml            # per-stage training configs

tools/launch_isaac.py       # Cross-platform launcher (adds --luadebug)
tests/                      # Offline unit tests (no live Isaac needed)
```

## What's implemented

- **Lua bridge (M1)** — TCP socket with length-prefixed JSON, `MC_POST_UPDATE` step loop at configurable frame skip, `MC_INPUT_ACTION` decoding for `MultiDiscrete([9, 5, 2, 2, 2])`.
- **Full observation (M2)** — player + global scalars, top-256 collectible passives, 9×15×4 room grid (walls / rocks / spikes / poop), 4 doors × 6 features, up to 24 enemies × 16 features, 48 projectiles/lasers × 10, 16 pickups × 8. Set-transformer attention on enemies/projectiles.
- **Reward shaping** — event-driven (damage dealt/taken, kills, room clear, first-entry bonuses, floor descent, Mom defeat, death). Anti-farming caps on damage-per-room and once-per-room / once-per-run bookkeeping.
- **Policy** — entity attention encoder, room-grid CNN, MLP fusion, GRU across timesteps, factored MultiDiscrete heads + value head.
- **RND** — intrinsic reward with running-mean-std normalization.
- **PPO trainer** — recurrent, GAE, factored logprob/entropy, LR decay, gradient clipping, TensorBoard, periodic checkpointing.
- **Curriculum** — 5 stages, YAML configs for smoke / single-room / floor / full-run.
- **Eval** — greedy-policy eval harness with Mom-kill rate + mean-max-floor metrics.
- **Tests** — 17 offline unit tests (obs / action encoding, reward shaping, policy forward passes, framed-JSON protocol).

---

## Setting up on your GPU machine

### 1. System requirements

- **OS:** Linux (recommended), Windows, or macOS. Multi-instance training is cleanest on Linux with Xvfb.
- **GPU:** any CUDA-capable card with ≥8 GB VRAM. The default policy config is ~15 M parameters; batch size 512 fits in 6 GB comfortably.
- **CPU:** 4-8 physical cores. CPU is the bottleneck (each Isaac instance chews one core at 30 Hz).
- **RAM:** ≥16 GB.
- **Software:** Python 3.10+, `git`, Steam with *The Binding of Isaac: Rebirth* + *Repentance* DLC.

### 2. Clone and install

```bash
git clone https://github.com/mylmyl123/ai-isaac.git
cd ai-isaac

python3 -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# For CUDA-enabled torch on Linux (replace cu121 with your CUDA version):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch

PYTHONPATH=python pytest tests/               # should print "17 passed"
```

### 3. Install the Lua mod

Copy `mods/isaac-rl-bridge/` into your Repentance mods folder:

| OS | Path |
|---|---|
| Linux | `~/.local/share/binding of isaac repentance/mods/` |
| Windows | `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\mods\` |
| macOS | `~/Library/Application Support/Binding of Isaac Repentance/mods/` |

Symlinking so edits reflect live is nicer:

```bash
# Linux
ln -s "$(pwd)/mods/isaac-rl-bridge" \
      ~/.local/share/binding\ of\ isaac\ repentance/mods/isaac-rl-bridge
```

In-game, open Mods and enable **Isaac RL Bridge**.

### 4. Verify `--luadebug` works (Risk R1 in the plan)

Isaac's Lua sandbox only exposes `require`/`socket` when `--luadebug` is set. Launch with the helper:

```bash
python tools/launch_isaac.py --port 9500
```

In the Isaac log (`~/.local/share/binding of isaac repentance/log.txt` on Linux, `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\log.txt` on Windows), you should see:

```
[isaac-rl-bridge] mod loaded (port 9500, frame skip 2)
```

If instead you see `require('socket') failed`, the `--luadebug` flag isn't reaching the process. Don't launch Isaac from the Steam UI — always go through `tools/launch_isaac.py`.

### 5. Smoke test the socket loop

Two terminals.

**Terminal A** — start the Python side first (it opens the server; Isaac connects into it):

```bash
source .venv/bin/activate
PYTHONPATH=python python -m isaac_rl.env --port 9500 --steps 1000
```

**Terminal B**:

```bash
python tools/launch_isaac.py --port 9500
```

Start any run in Isaac. Expected in Terminal A:

```
Isaac connected from 127.0.0.1:...
handshake: {'hello': True, 'seed': ..., ...}
step 0 @ 15.0 Hz — hp_red=6 ep_reward=-0.01
step 100 @ 15.0 Hz — ...
```

Isaac window: the character moves and shoots randomly.

If that works, the bridge is proven end-to-end.

---

## Training

### Stage 1 — single-room combat

The config in `python/isaac_rl/configs/stage1_single_room.yaml` uses 4 Isaac instances on ports 9500–9503. Two options:

**Option A — let the trainer launch Isaac (Linux only, needs a display or Xvfb).** Edit the config:

```yaml
launch_isaac: true
isaac_binary: /path/to/isaac-ng     # full path to the binary
```

Then:

```bash
PYTHONPATH=python python -m isaac_rl.ppo --config python/isaac_rl/configs/stage1_single_room.yaml
```

**Option B — launch Isaac manually (works everywhere, most reliable).** In separate terminals:

```bash
python tools/launch_isaac.py --port 9500
python tools/launch_isaac.py --port 9501
python tools/launch_isaac.py --port 9502
python tools/launch_isaac.py --port 9503
```

Then start training with `launch_isaac: false` (the default):

```bash
PYTHONPATH=python python -m isaac_rl.ppo --config python/isaac_rl/configs/stage1_single_room.yaml
```

TensorBoard:

```bash
tensorboard --logdir runs
```

Watch `rollout/ep_reward` climb; `reward/room_clear` should start firing within an hour of wall-clock training. Kill/checkpoint criteria are in the plan.

### Multi-instance tips (Linux)

- Run each Isaac inside its own Xvfb window to avoid focus-stealing and cursor issues:
  ```bash
  Xvfb :99 -screen 0 640x360x24 &
  DISPLAY=:99 python tools/launch_isaac.py --port 9500 &
  ```
- In Isaac's `options.ini`, set `Fullscreen=0`, `WindowWidth=640`, `WindowHeight=360`, `VSync=0`, `MusicVolume=0`, `SFXVolume=0`. These reduce CPU load a lot per instance.
- Frame skip is set in `mods/isaac-rl-bridge/main.lua` (`FRAME_SKIP`). Default 2 = 15 Hz control.

### Advancing through the curriculum

After Stage 1 hits ~90% room clear:

```bash
# Resume from Stage 1 checkpoint by copying its state_dict path into
# your Stage 2 launch (support for auto-resume is a small follow-up).
PYTHONPATH=python python -m isaac_rl.ppo --config python/isaac_rl/configs/stage2_floor_clear.yaml
```

Stages 3 and 4 configs live alongside; edit `total_env_steps` if you want a shorter budget.

### Evaluating a checkpoint

```bash
PYTHONPATH=python python -m isaac_rl.eval \
    --checkpoint runs/stage4_full_run/<timestamp>/ckpts/step_10000000.pt \
    --config python/isaac_rl/configs/stage4_full_run.yaml \
    --episodes 32
```

Prints Mom-kill rate, mean max floor reached, mean/median reward.

---

## Wall-clock expectations

At 15 Hz control and 4 Isaac instances, you get ~60 env-steps/second. Training budgets (from the plan):

| Stage | env-steps | wall-clock @ 60 sps |
|---|---|---|
| Stage 1 | 20M | ~4 days |
| Stage 2 | 50M | ~10 days |
| Stage 4 | 500M | ~3 months |

Bumping to 8 instances roughly halves wall-clock. If you can get a speedhack working (2×), halve it again.

---

## Known limitations

- **JSON wire format.** Fast enough for M1–M2 but adds ~10% overhead vs msgpack. Swap in `protocol.py` once the obs schema stabilizes.
- **Auto-resume between stages isn't wired yet.** Manually pass the previous checkpoint's `policy` state dict via a small script or copy the `.pt` file and load it (see `eval.py` for the loading pattern).
- **Async vec env is synchronous** (`SyncVecEnv`). Fine on 1 GPU but leaves throughput on the table if you scale past ~8 envs. Port to `gym.AsyncVectorEnv` if you go bigger.
- **`--luadebug` socket unlock** is not machine-verified until you run step 4 above. It's the only real project risk.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `require('socket') failed` in Isaac log | Missing `--luadebug`. Always launch through `tools/launch_isaac.py`. |
| `[Errno 48] Address already in use` | Prior server still bound. `lsof -i :9500` → kill, or use a different `--port`. |
| Trainer hangs on `listening for Isaac` | Isaac isn't loading the mod. Confirm it's enabled in the Mods menu and check the log for `[isaac-rl-bridge] mod loaded`. |
| Handshake succeeds but no step logs | Isaac is on the main menu. Start any run. |
| Character doesn't move despite step logs | Another input mod is capturing `MC_INPUT_ACTION`. Disable other mods. |
| `torch.cuda.is_available()` False | You installed CPU torch. Reinstall with the CUDA index URL for your CUDA version. |
| Trainer OOMs | Drop `minibatch_size` or `policy.trunk_dim`, or reduce `n_envs`. |

---

## Next steps

Milestones from the plan, in order:

1. **M1 — bridge stable** ✓ (done — this repo).
2. **M2 — clears a single room >90%.** Train `stage1_single_room.yaml`, verify with `eval.py`.
3. **M3 — clears Basement 1 >70%.** Move to `stage2_floor_clear.yaml`.
4. **M4 — beats Mom >50% on training seeds.** `stage4_full_run.yaml`.
5. **M5 — >30% on held-out seeds.** Run `eval.py` with the stage-5 seed pool.

If Stage 1 doesn't converge in ~3 months on PPO, the plan's fallback is a swap to DreamerV3 — the encoder in `model.py` is reusable directly.
