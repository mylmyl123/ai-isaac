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

> **All shell examples are PowerShell** (Windows). Linux / macOS equivalents are in the collapsed sections at the end of each step.

### 1. System requirements

- **OS:** Windows 10/11 (primary), Linux, or macOS. Multi-instance training is cleanest on Linux with Xvfb; on Windows each instance opens a real window.
- **GPU:** any CUDA-capable card with ≥8 GB VRAM. Default policy is ~15 M parameters; batch 512 fits in 6 GB.
- **CPU:** 4–8 physical cores. Each Isaac instance chews one core at 30 Hz — CPU is the throughput bottleneck.
- **RAM:** ≥16 GB.
- **Software:** Python 3.10+, `git`, Steam with *The Binding of Isaac: Rebirth* + *Repentance* DLC.

### 2. Clone and install

```powershell
git clone https://github.com/mylmyl123/ai-isaac.git
cd ai-isaac

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# CUDA-enabled torch (replace cu121 with your CUDA version — check `nvidia-smi`):
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Confirm CUDA is visible:
python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Run offline tests (should print '17 passed'):
$env:PYTHONPATH = "python"; pytest tests/
```

If PowerShell blocks `Activate.ps1`, run once as admin:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

<details><summary>bash equivalent</summary>

```bash
git clone https://github.com/mylmyl123/ai-isaac.git
cd ai-isaac
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu121 torch
PYTHONPATH=python pytest tests/
```
</details>

### 3. Install the Lua mod

Copy `mods\isaac-rl-bridge\` into your Repentance mods folder:

| OS | Path |
|---|---|
| Windows | `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\mods\` |
| Linux | `~/.local/share/binding of isaac repentance/mods/` |
| macOS | `~/Library/Application Support/Binding of Isaac Repentance/mods/` |

A symlink so edits reflect live (**run PowerShell as Administrator** or Windows will refuse):

```powershell
$src = (Resolve-Path .\mods\isaac-rl-bridge).Path
$dst = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\mods\isaac-rl-bridge"
New-Item -ItemType SymbolicLink -Path $dst -Target $src
```

Or just plain copy (no admin needed):

```powershell
Copy-Item -Recurse .\mods\isaac-rl-bridge "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\mods\"
```

In-game: open Mods and enable **Isaac RL Bridge**.

### 4. Verify `--luadebug` works (Risk R1 in the plan)

Isaac's Lua sandbox only exposes `require`/`socket` when `--luadebug` is set. Launch with the helper:

```powershell
python tools\launch_isaac.py --port 9500
```

Check the Isaac log — on Windows it lives at:

```powershell
Get-Content "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\log.txt" -Tail 50 -Wait
```

You should see `[isaac-rl-bridge] mod loaded (port 9500, frame skip 2)`. If instead it says `require('socket') failed`, the `--luadebug` flag isn't reaching the process. **Don't launch Isaac from the Steam UI** — always go through `tools\launch_isaac.py`.

### 5. Smoke test the socket loop

Two PowerShell windows.

**Terminal A** — start the Python side first (it opens the server; Isaac connects into it):

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "python"
python -m isaac_rl.env --port 9500 --steps 1000
```

**Terminal B**:

```powershell
python tools\launch_isaac.py --port 9500
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

### Option 0 — one-line unified launcher (recommended)

`train.py` at the repo root does everything: reads the config, launches Isaac instances, waits for them to connect, runs training, and cleans up on Ctrl-C.

```powershell
.\.venv\Scripts\Activate.ps1
python train.py --config python\isaac_rl\configs\stage1_single_room.yaml `
                --isaac "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe" `
                --tensorboard
```

What happens:

1. The trainer opens 2 server sockets on ports 9500–9501 (from the config's `n_envs`). Bump `n_envs` in the YAML for more instances.
2. Two Isaac windows open — each with `--luadebug` and `ISAAC_RL_PORT` set to its port.
3. **Click "New Run" (or press Start) in each Isaac window.** The mod fires `MC_POST_GAME_STARTED` and connects the socket. After this, resets happen automatically for the rest of training.
4. Training runs in the same terminal. `Ctrl-C` gracefully kills every Isaac child before exiting.
5. TensorBoard opens at http://localhost:6006 (if `--tensorboard`).

Useful overrides without editing the YAML:

```powershell
python train.py --config python\isaac_rl\configs\stage1_single_room.yaml `
                --n-envs 2 --base-port 9600 `
                --override total_env_steps=1000000 ent_coef=0.03
```

If auto-detect finds your Steam Isaac install (it looks in the usual places on Windows), you can omit `--isaac` entirely.

### Manual multi-window setup (fallback)

If `train.py` doesn't fit your workflow, you can run the pieces separately. Two options:

**Option A — let the trainer launch Isaac itself** (works on Windows, Linux). Edit the YAML:

```yaml
launch_isaac: true
isaac_binary: "C:\\Program Files (x86)\\Steam\\steamapps\\common\\The Binding of Isaac Rebirth\\isaac-ng.exe"
```

Note the double-backslashes — YAML strings need them. Then in **one** PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "python"
python -m isaac_rl.ppo --config python\isaac_rl\configs\stage1_single_room.yaml
```

**Option B — launch Isaac manually** (most reliable — easier to watch/kill individual instances). One PowerShell per instance:

```powershell
python tools\launch_isaac.py --port 9500
```

Repeat for `9501` in one more PowerShell (or more, matching the `n_envs` in your config). Or spawn N from one shell:

```powershell
9500..9501 | ForEach-Object {
    Start-Process powershell -ArgumentList "-NoExit","-Command","python tools\launch_isaac.py --port $_"
}
```

Then start training (keep `launch_isaac: false`):

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "python"
python -m isaac_rl.ppo --config python\isaac_rl\configs\stage1_single_room.yaml
```

TensorBoard in yet another PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir runs
```

Watch `rollout/ep_reward` climb; `reward/room_clear` should start firing within an hour of wall-clock training. Kill/checkpoint criteria are in the plan.

<details><summary>bash equivalents</summary>

```bash
# Option B, one env per shell:
python tools/launch_isaac.py --port 9500
python tools/launch_isaac.py --port 9501
python tools/launch_isaac.py --port 9502
python tools/launch_isaac.py --port 9503

# Trainer:
PYTHONPATH=python python -m isaac_rl.ppo --config python/isaac_rl/configs/stage1_single_room.yaml

# TensorBoard:
tensorboard --logdir runs
```
</details>

### Multi-instance tips (Windows)

- Each Isaac instance opens its own window. Give the trainer window focus so it stays responsive; the Isaac windows can be minimized (background rendering still runs, but at a lower priority).
- In Isaac's `options.ini` (same folder as `log.txt`), set: `Fullscreen=0`, `WindowWidth=640`, `WindowHeight=360`, `VSync=0`, `MusicVolume=0`, `SFXVolume=0`. Cuts per-instance CPU meaningfully.
- Frame skip is set at the top of `mods\isaac-rl-bridge\main.lua` (`FRAME_SKIP`). Default 2 = 15 Hz control.
- Windows Defender occasionally slows the first mod load. If `require("socket")` succeeds but the first exchange stalls, add the Isaac install folder to Defender's exclusions.

### Advancing through the curriculum

After Stage 1 hits ~90% room clear:

```powershell
$env:PYTHONPATH = "python"
python -m isaac_rl.ppo --config python\isaac_rl\configs\stage2_floor_clear.yaml
```

Stages 3 and 4 configs live alongside; edit `total_env_steps` in the YAML if you want a shorter budget.

> **Auto-resume between stages isn't wired yet.** For continuity, load the previous stage's `.pt` in the trainer before it starts (small edit in `ppo.py` — the eval script shows the loading pattern).

### Evaluating a checkpoint

```powershell
$env:PYTHONPATH = "python"
python -m isaac_rl.eval `
    --checkpoint runs\stage4_full_run\<timestamp>\ckpts\step_10000000.pt `
    --config python\isaac_rl\configs\stage4_full_run.yaml `
    --episodes 32
```

Note: PowerShell continues lines with a backtick (`` ` ``), not backslash. Prints Mom-kill rate, mean max floor reached, mean/median reward.

---

## Wall-clock expectations

At 15 Hz control, each Isaac instance produces ~15 env-steps/second. The default config uses 2 instances (~30 sps).

| Stage | env-steps | @ 2 envs (30 sps) | @ 4 envs (60 sps) | @ 8 envs (120 sps) |
|---|---|---|---|---|
| Stage 1 | 20M | ~8 days | ~4 days | ~2 days |
| Stage 2 | 50M | ~20 days | ~10 days | ~5 days |
| Stage 4 | 500M | ~6 months | ~3 months | ~6 weeks |

Bump `n_envs` in the YAML for more instances. Each extra instance costs ~1 CPU core and ~500 MB RAM. If you can get a speedhack working (2×), halve wall-clock again.

---

## Known limitations

- **JSON wire format.** Fast enough for M1–M2 but adds ~10% overhead vs msgpack. Swap in `protocol.py` once the obs schema stabilizes.
- **Auto-resume between stages isn't wired yet.** Manually pass the previous checkpoint's `policy` state dict via a small script or copy the `.pt` file and load it (see `eval.py` for the loading pattern).
- **Async vec env is synchronous** (`SyncVecEnv`). Fine on 1 GPU but leaves throughput on the table if you scale past ~8 envs. Port to `gym.AsyncVectorEnv` if you go bigger.
- **`--luadebug` socket unlock** is not machine-verified until you run step 4 above. It's the only real project risk.

---

## Before training: disable other mods

The RL bridge assumes vanilla Repentance. Other mods will fight it for `MC_INPUT_ACTION`, spawn modded enemies the tables don't know, and reward-hack the agent. Toggle them off with the helper:

```powershell
python tools\manage_mods.py list             # see what's on
python tools\manage_mods.py disable-others   # disable everything except isaac-rl-bridge
# ... train ...
python tools\manage_mods.py enable-all       # restore everything after training
```

The disable is reversible (writes `disable.it` markers instead of deleting mods). Launch Isaac once through Steam after running `disable-others` so the change takes effect, then close.

## Troubleshooting

### Isaac opens then immediately closes

Almost always one of:

1. **`steam_appid.txt` missing next to `isaac-ng.exe`.** `train.py` writes it automatically now, but if you're launching Isaac another way, you need this file in the same folder as the binary containing just `250900`. Without it, Repentance's DRM stub tries to relaunch under Steam, both launches collide, exit code 53.
2. **Wrong working directory.** Isaac loads `resources/`, `shaders/`, `packed/` etc. via **paths relative to its cwd**. If cwd isn't the Isaac install directory, most assets fail to load and the game crashes (0xC0000005). `train.py` sets cwd to the install directory automatically.
3. **A conflicting mod is crashing on startup.** Run `python tools\manage_mods.py disable-others` and try again.

For a systematic diagnosis, run the launch tester — it tries 5 different launch strategies and prints the tail of every `log.txt` it can find:

```powershell
python tools\test_launch.py --isaac "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"
```

Before running the diagnostic, make sure:

- **Steam is running** (tray icon present): `Get-Process steam -ErrorAction SilentlyContinue`
- **Isaac has been launched through Steam at least once** on this account — do this by double-clicking Isaac in your Steam library, wait for the title screen, close it. This caches the Repentance DLC ownership.
- **Verify game files** in Steam → Right-click Isaac → Properties → Installed Files → Verify integrity, if you suspect a broken install.
- **Repentogon users:** tell the diagnostic which flag your version needs (or launch through Repentogon's launcher first once). Some Repentogon builds refuse `--luadebug` unless combined with `-repentogon`.

The diagnostic's summary line tells you which launch strategy works. Once one succeeds, that's what `train.py` should be doing. Paste the output if all 5 fail — Isaac's own log will name the failure (missing DLC, DRM refusal, asset load error, etc.).

### Other issues

| Symptom | Fix |
|---|---|
| `require('socket') failed` in Isaac log | Missing `--luadebug`. Always launch through `tools/launch_isaac.py` or `train.py`. |
| `[Errno 48] Address already in use` (or `[WinError 10048]`) | Prior server still bound. PowerShell: `Get-NetTCPConnection -LocalPort 9500` then `Stop-Process -Id <pid>`. Or just use a different `--port`. |
| Trainer hangs on `listening for Isaac` | Isaac isn't loading the mod, or another mod is crashing before ours loads. Check `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\log.txt` for `[isaac-rl-bridge] mod loaded`. Also run `python tools\manage_mods.py disable-others`. |
| Handshake succeeds but no step logs | Isaac is on the main menu. Use `--auto-start-stage 1` (the default) or click New Run. |
| Character doesn't move despite step logs | Another input mod is capturing `MC_INPUT_ACTION`. Disable all mods except Isaac RL Bridge. |
| Ctrl-C doesn't stop the trainer | Fixed on `main` — `pull` if you haven't recently. Socket recv now uses 1s timeouts so Python can process signals. |
| `torch.cuda.is_available()` False | You installed CPU torch. `pip uninstall torch -y` then reinstall with the CUDA index URL for your CUDA version. |
| Trainer OOMs (CUDA) | Drop `minibatch_size`, `policy.trunk_dim`, or `n_envs` in the config YAML. |
| CPU pegged at 100%, Isaac windows stuttering | Too many instances for your box. Drop `n_envs` (default is 2). |

---

## Next steps

Milestones from the plan, in order:

1. **M1 — bridge stable** ✓ (done — this repo).
2. **M2 — clears a single room >90%.** Train `stage1_single_room.yaml`, verify with `eval.py`.
3. **M3 — clears Basement 1 >70%.** Move to `stage2_floor_clear.yaml`.
4. **M4 — beats Mom >50% on training seeds.** `stage4_full_run.yaml`.
5. **M5 — >30% on held-out seeds.** Run `eval.py` with the stage-5 seed pool.

If Stage 1 doesn't converge in ~3 months on PPO, the plan's fallback is a swap to DreamerV3 — the encoder in `model.py` is reusable directly.
