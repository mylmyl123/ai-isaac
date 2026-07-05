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

Copy `mods\isaac-rl-bridge\` into your Repentance mods folder. Isaac reads mods from **two** locations, either works:

| Location | Path (Windows) |
|---|---|
| Next to the game binary (used by Steam Workshop) | `C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\mods\` |
| Documents folder (used for hand-installed mods) | `%USERPROFILE%\Documents\My Games\Binding of Isaac Repentance\mods\` |

Linux uses `~/.local/share/binding of isaac repentance/mods/` for user-installed and the equivalent `steamapps/.../mods/` for workshop.  macOS uses `~/Library/Application Support/Binding of Isaac Repentance/mods/`.

You can put `isaac-rl-bridge` in either — Isaac loads mods from both dirs at startup. The workshop-side location works even if you don't have write access to Documents.

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

## Training with DreamerV3 (recommended)

The repo also ships a full **DreamerV3** trainer alongside PPO. Dreamer is a world-model / imagination-based algorithm that's ~10–20× more sample-efficient than PPO on sparse-reward procedural games (DreamerV3 solved Minecraft-diamond from scratch, no demos, no curriculum — Hafner et al., *Nature* 2025). On this project's compute budget (1 GPU, ~30 env-steps/sec), Dreamer is the algorithm that makes "beat Mom in weeks" realistic; PPO was projected to take months.

The PPO stack is preserved for ablation baselines — pick between them with `--algo {ppo,dreamer}` on `train.py` and `eval.py`.

### What lives where

```
python/isaac_rl/dreamer/
├── vendor/           NM512/dreamerv3-torch @ 6ef8646 (MIT — RSSM, twohot, KL utils)
├── encoder.py        IsaacObsEncoder (entity-attn architecture from PPO's model.py)
├── decoder.py        IsaacObsDecoder (per-stream reconstruction heads)
├── action.py         MultiDiscreteActionHead (factored [9, 5] one-hot concat)
├── replay.py         SequenceReplay (episode-aware, terminal-obs correct)
├── isaac_models.py   IsaacWorldModel + IsaacImagBehavior
├── train.py          main training loop
├── config.py         DreamerConfig dataclass (paper hyperparameters)
└── configs/
    ├── stage1_single_room.yaml
    ├── stage2_floor_clear.yaml
    └── stage4_full_run.yaml
```

No new dependencies beyond what PPO already needs — same `requirements.txt`.

### 1. Verify the port compiles

```powershell
$env:PYTHONPATH = "python"; pytest tests/dreamer/
```

Expect `24 passed` in a few seconds. All offline (no live Isaac needed). Also run `pytest tests/` to confirm the 94 PPO tests still pass — the port doesn't touch the PPO code.

### 2. The 3-script workflow (this is all you need day-to-day)

```powershell
.\scripts\run.ps1           # start training. Ctrl-C to stop cleanly.
.\scripts\push_data.ps1     # export TB summary + checkpoint, commit, push.
.\scripts\clear_data.ps1    # nuke runs\ before a fresh experiment.
```

That's it. The wrappers pick sane defaults (stage 1, n_envs from the YAML, TensorBoard on, Isaac binary auto-detected from Steam), catch Ctrl-C so the trainer flushes a final checkpoint and TB events before exiting, and know where to look for output files.

**`scripts\run.ps1`** — start training. Common flags:

```powershell
.\scripts\run.ps1                              # stage 1 default (5M steps)
.\scripts\run.ps1 -Smoke                       # M1 smoke: 100k steps, n_envs=2 (~1 hour)
.\scripts\run.ps1 -Stage 2                     # stage 2 config
.\scripts\run.ps1 -Stage 4                     # stage 4 config
.\scripts\run.ps1 -NEnvs 4                     # override n_envs
.\scripts\run.ps1 -Isaac "C:\Path\isaac-ng.exe"  # override binary path
.\scripts\run.ps1 -NoTensorboard               # skip TB (saves a bit of I/O)
```

**Ctrl-C behavior:** the trainer traps SIGINT. First Ctrl-C finishes the current rollout, saves a checkpoint tagged `interrupted`, flushes TB, and exits cleanly. Second Ctrl-C force-quits (progress since the last checkpoint is lost — don't hit it twice unless you have to).

**`scripts\push_data.ps1`** — after Ctrl-C, run this to ship the run's artifacts back to the repo. Auto-picks the most recent run under `runs\`:

```powershell
.\scripts\push_data.ps1                        # most recent run
.\scripts\push_data.ps1 -RunGlob "dreamer_stage2_*"      # most recent stage 2
.\scripts\push_data.ps1 -RunDir "runs\dreamer_stage1_single_room\20260704-231512"  # specific run
.\scripts\push_data.ps1 -NoCheckpoint          # skip the .pt (JSON only)
```

What gets committed:

- `tb_dreamer_<stage>_<timestamp>.json` — compact scalar summary (~50 KB). Send me this for analysis.
- `ckpts\latest_<stage>_<timestamp>.pt` — copy of the run's `latest.pt` (~150 MB). Skip via `-NoCheckpoint` if GitHub rejects the size (>100 MB) — in that case upload the raw `.pt` via bucket/scp/email.

**`scripts\clear_data.ps1`** — wipe local data before a fresh run:

```powershell
.\scripts\clear_data.ps1                       # deletes runs\, prompts first
.\scripts\clear_data.ps1 -Yes                  # no prompt
.\scripts\clear_data.ps1 -Stage 1              # only stage 1's runs
.\scripts\clear_data.ps1 -All -Yes             # runs\ + ckpts\ + tb_*.json, no prompt
```

### 3. What to expect on TensorBoard (M1 smoke success criteria)

Run `.\scripts\run.ps1 -Smoke`, wait ~1 hour, open http://localhost:6006:

- `loss/total` (world model) trending down over 100k steps
- `loss/kl`, `loss/kl_dyn`, `loss/kl_rep` finite and roughly stable (0.1 – 10 range)
- `loss/actor` finite (starts near 0, mildly negative)
- `loss/critic` finite, gently trending
- `rollout/ep_reward` above the random-policy baseline
- No NaN anywhere; run completes without crash

If any of these blow up (NaN, exploding KL), `.\scripts\push_data.ps1` and share the JSON — that's a real bug and I need the numbers to diagnose. Do NOT proceed to full stage 1 training.

### 4. Full stage 1 → stage 2 → stage 4 chain

After M1 smoke looks clean:

```powershell
.\scripts\run.ps1                              # stage 1, 5M steps, 1-2 weeks
# ... Ctrl-C when ready, or let it complete ...
.\scripts\push_data.ps1                        # ship the data

# When stage 1 room-clear rate is >=90%, resume into stage 2:
.\scripts\run.ps1 -Stage 2                     # 20M steps, 2-3 weeks
# ... Ctrl-C or complete ...
.\scripts\push_data.ps1 -RunGlob "dreamer_stage2_*"

# Then stage 4:
.\scripts\run.ps1 -Stage 4                     # 100M steps, 4-8 weeks
.\scripts\push_data.ps1 -RunGlob "dreamer_stage4_*"
```

(To resume stage 2 from a stage 1 checkpoint, pass `--override resume_from=runs\dreamer_stage1_single_room\<timestamp>\latest.pt` to `run.ps1` via its `-NEnvs` isn't the right mechanism — use `python train.py` directly for that one-off, or open an issue and I'll wire a `-Resume` flag.)

### 5. Evaluating a Dreamer checkpoint

Same eval harness as PPO, just add `--algo dreamer`:

```powershell
python -m isaac_rl.eval `
    --algo dreamer `
    --checkpoint runs\dreamer_stage1_single_room\<timestamp>\latest.pt `
    --config python\isaac_rl\dreamer\configs\stage1_single_room.yaml `
    --episodes 32
```

Prints `n_episodes`, `mean_reward`, `median_reward`, `mom_kills`, `mom_kill_rate`, `mean_max_stage` — same output format as PPO eval, so you can compare directly.

### Ablation: Dreamer vs PPO on the same task

The PPO stack is untouched. Running the same experiment with `--algo ppo` on `python\isaac_rl\configs\stage1_single_room.yaml` gives you the ablation baseline. Both write to `runs/`; TensorBoard reads them all at once:

```powershell
tensorboard --logdir runs
```

The `rollout/ep_reward_best` and `reward/room_clear` curves side-by-side are the main ablation figure for a writeup.

### Hyperparameter notes

The YAML configs match DreamerV3 paper defaults (Hafner et al., *Nature* 2025) as reproduced in NM512/dreamerv3-torch. **Don't tune the world-model hyperparameters first** — they're validated across 150+ tasks in the paper. If M1 looks bad, check reward shaping (`reward:` block in the YAML) and `n_envs` before touching RSSM knobs.

Key knobs if you need to fit tighter memory:

- `replay_capacity` — 1M steps ≈ 7 GB RAM. Drop to 500K if RAM is tight.
- `batch_size` × `seq_len` = 16 × 64 by default. Fits ~6 GB VRAM at Isaac's obs size. Halve if you OOM.
- `train_ratio` — WM gradient steps per env-step. Default 16. Drop to 8 if GPU is the bottleneck; bump to 32 if GPU is idle.

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

## Before training: disable other mods & configure Isaac

The RL bridge assumes vanilla Repentance. Other mods will fight it for `MC_INPUT_ACTION`, spawn modded enemies the tables don't know, and reward-hack the agent. Toggle them off with the helper:

```powershell
python tools\manage_mods.py list             # see what's on
python tools\manage_mods.py disable-others   # disable everything except isaac-rl-bridge
# ... train ...
python tools\manage_mods.py enable-all       # restore everything after training
```

You also need to patch Isaac's `options.ini` so it doesn't pause when the window loses focus (fatal for multi-instance training) and doesn't throttle background windows to 5 FPS:

```powershell
python tools\configure_isaac.py show     # print current vs. training-friendly values
python tools\configure_isaac.py apply    # write the training-friendly values (backs up first)
# ... train ...
python tools\configure_isaac.py restore  # restore your original options.ini
```

Both scripts are reversible — `apply` backs up your original as `options.ini.pre-rl-bak`, and `restore` puts it back.

Launch Isaac once through Steam after both scripts so it picks up the changes, close, then start training.

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
| One Isaac window is fine, the OTHER lags heavily and eventually crashes | Windows 10/11 background-window throttling. `PauseOnFocusLost=0` doesn't defeat this — it's an OS-level priority cut. `train.py` now spawns children with `ABOVE_NORMAL_PRIORITY_CLASS` which is enough to defeat it. If you're launching Isaac manually, run `Get-Process isaac-ng \| ForEach-Object { $_.PriorityClass = 'AboveNormal' }` after they start. Keep every Isaac window visible on screen (not minimized) — minimized windows get an additional GPU/render throttle no priority class can override. |
| Isaac boots to intro / main menu and won't auto-start | `--set-stage=N` **requires** the `=` (Isaac's parser doesn't accept a space). `train.py` uses the correct syntax; if you're launching manually, use `isaac-ng.exe --luadebug --set-stage=1 --set-stage-type=0`. That skips both the studio logos and the menu. |
| Game crashes the instant a run starts | Almost always the Lua auto-start firing `restart 0` during the intro cinematic. Fixed on `main` — the fallback now waits ~10s and won't fire once the run is active. If you're on an older checkout, `git pull`. Also make sure the launch flag above is set so the fallback never triggers. |
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

---

## Future research directions

A detailed roadmap of advanced RL techniques we haven't implemented yet lives
in **[docs/FUTURE_WORK.md](docs/FUTURE_WORK.md)**. Contents:

- **Phase B** (medium effort, high value): distributional value function (C51),
  curriculum learning, predict-future-rewards aux task, latent variable
  conditioning (AlphaStar-style).
- **Phase C** (high effort, potentially transformative): DreamerV3-inspired
  world model with imagination rollouts, transformer-based policy.
- **Phase D** (advanced exploration): Never-Give-Up episodic novelty,
  Population-Based Training, bootstrapped Q-ensembles.

Each item includes rationale, papers, concrete implementation plan, effort
estimate, and expected gain. Refer to that doc if the current setup plateaus
or if we want to push toward superhuman play.
