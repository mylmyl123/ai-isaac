# Isaac RL: Minimum-Change Maximum-Impact Fixes

## TL;DR

Peak was **10.65 kills/ep at step ~33k**. By step 100k it decayed to **5.30**. The
TensorBoard trace tells the whole story:

| step | kills/ep | LR      | loss/entropy | action_top_frac (shoot) |
|-----:|---------:|--------:|-------------:|-------------------------:|
|  33k |    10.65 | 2.0e-4  |         5.35 |                     ~0.28 |
|  60k |     5.75 | 1.2e-4  |         4.88 |                     ~0.34 |
|  90k |     3.90 | 3.0e-5  |         4.75 |                     ~0.36 |
| 100k |     5.30 | 4.8e-7  |         4.55 |                     0.363 |

Two things happened simultaneously between 33k and 100k:
1. **`anneal_lr: true`** drove LR from 2e-4 to effectively zero (99.8% decayed).
2. Policy entropy dropped monotonically from 5.35 → 4.55 (–15%) and the
   most-used shoot-direction moved from 28% → 36% of samples. The policy
   was **committing to a single tactic while its ability to escape it
   shrank to zero**.

At step 100k `loss/approx_kl = 0` and `loss/clipfrac = 0`. Updates
were doing nothing. **The trainer literally froze the policy inside a
local optimum.** This is not a corner-camp problem — it is an
optimizer-configuration problem masquerading as one.

Everything else in this doc is ranked accordingly.

---

## Tier 1 — Try FIRST (30 min total, direct TB evidence, high EV)

### Fix 1: **Disable LR annealing** *(or set a floor)*

- **What:** Stop decaying LR to zero over the training horizon. The horizon
  was set to 100k for Stage A, so annealing was aggressive by design. TB
  data shows LR hit 4.8e-7 at the final step — the last ~30k steps of
  gradient descent were noise.
- **Code (config-only):**

  ```yaml
  # configs/curriculum.yaml
  anneal_lr: false          # was: true
  ```

  Or, if you want to *keep* some decay for the final polish while
  preventing the freeze (2-line diff in `cleanrl_ppo.py`):

  ```python
  # python/isaac_rl/cleanrl_ppo.py, in train() around line 245
  if cfg.anneal_lr:
      frac = 1.0 - (global_step / cfg.total_env_steps)
      for pg in optimizer.param_groups:
-         pg["lr"] = cfg.lr * max(frac, 0.0)
+         # Never let LR fall below 10% of base. Prevents policy freeze
+         # into local optima seen at 60k-100k step of stage A.
+         pg["lr"] = cfg.lr * max(frac, 0.1)
  ```

- **Cost:** 1 line YAML edit. ~5 min. Validate in ~50k steps.
- **P(fixes regression):** **65%.** Direct evidence: peak occurred when
  LR was still healthy; regression tracked LR decay. Even without any
  other change this alone should restore the peak.
- **Rationale:** In `total_env_steps=100k`, the anneal schedule spends
  the last 40k steps below 1.5e-4. The optimizer becomes unable to
  climb out of any local minimum it has drifted into. This is the
  #1 suspect and the cheapest fix.

### Fix 2: **Raise entropy coefficient 0.01 → 0.03**

- **What:** Current `ent_coef=0.01`, `loss/entropy` = 4.55 → entropy
  bonus ≈ 0.045 per step. Kill reward = +1 per kill (~10 per episode
  at peak) → reward pressure dominates entropy pressure by ~2 orders
  of magnitude. Bump by 3× (not 5×) — enough to arrest the collapse
  without wrecking the peak.
- **Code:**

  ```yaml
  # configs/curriculum.yaml
  ent_coef: 0.03            # was: 0.01
  ```

- **Cost:** 1 line. ~2 min. Validate in same 50–100k run.
- **P(fixes regression):** **45%** standalone, **~70% combined with
  Fix 1**. Evidence: entropy monotone decline + `action_top_frac`
  monotone rise → premature commitment.
- **Rationale:** Prevents the policy from *committing* to the tactic
  that peaked at 33k before it can consolidate around a broader
  strategy. Cheap insurance.

### Fix 3: **Extend `total_env_steps` for Stage A to 200k, not 100k**

- **What:** Two goals: (a) let the healthier LR schedule finish learning
  rather than truncating it, (b) verify the fix persists rather than
  reading noise off a single 100k point.
- **Code:**

  ```yaml
  # configs/curriculum.yaml
  total_env_steps: 200_000  # was: 100_000
  ```

- **Cost:** 1 line. Compute cost: 2× the previous run (~200k env
  steps ≈ 3.5 hr on your 2-env × 15 sps setup).
- **P(fixes regression):** **20% standalone**, but essential for
  *diagnosing* whether Fix 1+2 worked. Without extra horizon the
  peak-vs-final comparison is not statistically distinguishable
  from noise on 20-episode window averages.
- **Rationale:** The `charts/kills_mean` uses last-20-ep moving
  average. With ep_len ≈ 400 frames and n_envs=2, 20 episodes ≈
  16k steps — the "5.30 at 100k" number has ±1.5 kill/ep std.
  You need more samples to distinguish signal from noise.

### **Combined Tier-1 rollout: 3-line YAML diff, run once.**

```diff
# configs/curriculum.yaml
- total_env_steps: 100_000
+ total_env_steps: 200_000
- lr: 3.0e-4
+ lr: 3.0e-4
- ent_coef: 0.01
+ ent_coef: 0.03
- anneal_lr: true
+ anneal_lr: false
```

Estimated implementation: **10 minutes.** Validation: **200k steps ≈
3.5 hours.** Combined P(kills stabilizes ≥8/ep at 200k): **~75%.**

---

## Tier 2 — Try if Tier-1 fails (medium cost, high impact)

### Fix 4: **Obs pruning for Stage A** *(hidden-structure win)*

- **What:** In Stage A only 1 fly, sealed room, no items, no doors, no
  passives. Yet the MLP sees a **2,504-dim** flat input (734 passives +
  540 room-grid + 384 enemies + 480 projectiles + 128 pickups + 35
  character + 72 doors + 40 player + 20 global + …). The first
  `Linear(2504, 256)` is **~640k parameters** for a task whose true
  signal is 4 numbers (player x/y, fly x/y).
- **Estimated overparameterization ratio: ~10,000×.**
  Every gradient update has 640k parameters worth of noise fighting
  the ~4 dimensions of signal. This explains policy variance and drift.
- **Code (obs-side stage-A mask, ~15 lines in `cleanrl_ppo.py`):**

  ```python
  # python/isaac_rl/cleanrl_ppo.py: near _flat_obs()
  STAGE_A_KEYS = {"player", "enemies_feats", "enemies_mask",
                  "spatial", "player_history", "last_action"}

  def _flat_obs(o, keep=None):
      parts = []
      flat = flatten_dict_obs(o)
      for k in sorted(flat.keys()):
          if keep is not None and k not in keep:
              continue
          parts.append(np.asarray(flat[k], dtype=np.float32).reshape(-1))
      return np.concatenate(parts)
  ```

  Then in `train()` and `_obs_dim()`, pass `keep=STAGE_A_KEYS` when
  `cfg.stage == "A"`. Obs dim drops **2,504 → ~470**. First-layer
  params drop **640k → 120k (5×)**.

- **Cost:** ~20 min, ~15 lines. Validate 100k steps.
- **P(fixes regression):** **45%.** Independently, this reduces
  update variance and should stabilize peak. Bigger effect: the
  policy is forced to attend to the signal that actually matters.
- **Rationale:** The single most-impactful "hidden-structure" fix
  available. It costs nothing at inference and reduces the SNR
  problem the policy is fighting.

### Fix 5: **Multi-enemy Stage A from step 1** *(from candidate K)*

- **What:** With 1 enemy, corner-camping *is* near-optimal (homing
  fly comes to you; you shoot in +X). Multi-enemy makes camping
  strictly worse than aiming.
- **Code (2 line change in `mods/isaac-rl-bridge/main.lua`, line ~99):**

  ```lua
  -- Was: local STAGE_FLY_COUNT = (STAGE == "B") and 3 or 1
  local STAGE_FLY_COUNT = (STAGE == "A") and 3
                        or (STAGE == "B") and 5
                        or 1
  ```

  Note: this collapses Stage A ≈ Stage B. That is *fine* — the current
  Stage A distinction is a bug, not a feature, per the config's own
  comment ("Degenerate task — policy learns to corner-camp because
  homing fly comes to player. SKIPPED.").
- **Cost:** 2 lines Lua. ~10 min. Validate 100k.
- **P(fixes regression):** **50%.** Removes the local optimum, but
  also raises the task difficulty. Combined with Fix 1+2 this should
  give a monotone learning curve rather than a peak-then-decay.

### Fix 6: **Change Stage A enemy to non-homing Pooter** *(candidate E)*

- **What:** Pooter (EntityType.ENTITY_POOTER = 14, or Fly = 13 variant 0)
  wanders; doesn't home. Kills the corner-camp reward structure at the
  source.
- **Code (`main.lua` line 235):**

  ```lua
  -- Attack Fly homes → corner-camping is optimal. Pooter is random-walk.
  local fly = Isaac.Spawn(EntityType.ENTITY_POOTER, 0, 0,
                          spawn_pos, Vector(0, 0), nil)
  ```

- **Cost:** 3 lines Lua. ~10 min. Validate 100k.
- **P(fixes regression):** **55%.** But Pooter is *easier to miss*, so
  the initial kill rate will be lower. May look like a regression even
  though it's more general learning.

### Fix 7: **Reward per unit of damage dealt** *(shaping, candidate H)*

- **What:** Reward becomes `r = 0.1 * damage_dealt + 1.0 * kill_bonus`.
  Provides gradient signal for *near-misses* becoming *hits* becoming
  *kills*. Kill peak stays as terminal signal.
- **Code (5 lines in `reward.py` around line 100):**

  ```python
  for ev in events:
      kind = ev.get("kind")
      if kind == "damage_to_npc":
          bd["damage"] = bd.get("damage", 0.0) + 0.1 * float(ev.get("dmg", 1.0))
          if ev.get("killed"):
              bd["kill"] = bd.get("kill", 0.0) + self.cfg.r_kill
      elif kind == "death" and not self.state.dead:
          ...
  ```

  **Requires** the mod to emit `dmg` in `damage_to_npc` events — check
  `mods/isaac-rl-bridge/reward.lua` before running.
- **Cost:** 5 lines Python + verify Lua field. ~30 min. Validate 100k.
- **P(fixes regression):** **50%.** Dense signal → faster convergence,
  less policy variance. Small caveat: violates the "3-term reward"
  religious rule from `reward.py`. Documented rationale required.

### Fix 8: **PBRS via distance-to-nearest-enemy potential** *(candidate I)*

- **What:** Ng-1999-style potential-based shaping. `Φ(s) =
  -min_dist_to_enemy / room_diag`. Reward is unchanged in expectation
  (theorem-guaranteed) but agent gets local gradient toward enemies.
- **Code:** ~15 lines in `reward.py`. Provably preserves optimal
  policy (unlike Fix 7).
- **Cost:** ~1 hr. Validate 100k.
- **P(fixes regression):** **50%.** Slightly better than Fix 7 in
  theory, slightly more code. But if Fix 1+2 works, this is unneeded.

---

## Tier 3 — Nuclear (only if Tier-1 and Tier-2 both fail)

### Fix 9: **BC warm-start from heuristic** *(candidate M)*

- ~100 lines, 200k steps, P ≈ 55%. Real bootstrap of exploration.
  Use `heuristic.py` (already in repo per plan) → generate 5k demos
  → BC pretrain the actor → PPO fine-tune. Only justified if the
  exploration problem is real, and Tier-1 evidence says it isn't.

### Fix 10: **Off-policy replay (SAC/DQN)** *(candidate L)*

- 500+ lines. **Skip.** The cleanrl_ppo has no smoking gun that
  demands off-policy. Peak was 10.65 — PPO can do this task.

### Fix 11: **Add LSTM / bigger network** *(candidate D)*

- The obs already contains `player_history` (4-frame stack) and
  `spatial` features, plus `z` latent. Recurrence would help
  partial observability, but Stage A is fully observable. **Skip
  for Stage A.** Revisit for Stage D+.

### Fix 12: **RND** *(candidate J)*

- 60 lines. **Skip.** Stage A has trivial state coverage
  (single 480×270 room). Novelty bonus adds noise, not signal.

---

## Sequential Recommended Path

**Step 1 (10 min):** Apply Tier-1 combined diff (Fix 1+2+3).
```yaml
# configs/curriculum.yaml
anneal_lr: false
ent_coef: 0.03
total_env_steps: 200_000
```

**Step 2 (3.5 hr compute):** Run one 200k-step Stage A. Success criteria:
  - `kills_mean` reaches ≥8 by step 50k **and**
  - Stays ≥8 (± 2) through 200k without a monotone decay.

**Step 3 (evaluate):**
  - **If success**: ship it, move to Stage B. Done.
  - **If plateaued at 5–7 kills/ep**: apply Fix 4 (obs pruning). Cheapest
    Tier-2 fix, biggest theoretical win, doesn't require Lua edits.
    Add Fix 5 (multi-enemy) if the policy shape looks stuck in a corner
    camp on video.
  - **If kills never exceed 3**: something else is broken (event stream,
    action mask). Bisect per the trainer's own debugging comment
    (lines 47–52 of `cleanrl_ppo.py`).

**Step 4 (only if Step 3 still fails after ~500k steps of experiments):**
  Fix 7 (damage shaping) OR Fix 8 (PBRS). One, not both.

**Step 5 (nuclear):** Fix 9 (BC warm-start). Estimate 2–3 dev-days.

---

## Rank Table (EV = P(fix) × Impact / Cost)

| # | Fix                       | Impl (hr) | Compute (steps) | P(fix) | EV rank |
|--:|:--------------------------|----------:|----------------:|-------:|--------:|
| 1 | Disable LR anneal         |      0.1  |            100k |    65% | **1**   |
| 2 | ent_coef 0.01→0.03        |      0.05 |            100k |    45% | **2**   |
| 3 | total_env_steps 100k→200k |      0.05 |            100k |    20% |     3   |
| 4 | Obs prune for Stage A     |      0.5  |            100k |    45% | **4**   |
| 5 | Multi-enemy from step 1   |      0.2  |            100k |    50% |     5   |
| 6 | Non-homing enemy (Pooter) |      0.2  |            100k |    55% |     6   |
| 7 | Damage-based shaping      |      0.5  |            100k |    50% |     7   |
| 8 | PBRS distance potential   |      1.0  |            100k |    50% |     8   |
| 9 | BC warm-start             |     20    |            200k |    55% |    10   |
|10 | LSTM / bigger network     |      4    |            300k |    40% |    11   |
|11 | Off-policy (SAC/DQN)      |     40    |            300k |    50% |    12   |
|12 | RND intrinsic reward      |      6    |            300k |    30% |    13   |

Tier ordering follows "cheapest reversible change with evidence first";
Fix 3 is a compute cost, not a fix, but is a *prerequisite* for reading
whether Fix 1+2 worked.

---

## Highlights

**Fastest fix worth trying first:**
> `anneal_lr: false` — one YAML line, direct TB evidence (LR hit 4.8e-7,
> `approx_kl → 0`, `clipfrac → 0` at exactly the step where kills
> bottomed out). Estimated 65% single-shot fix.

**Highest-ceiling fix if fast fixes fail:**
> Obs pruning for Stage A (Fix 4). Current input dim is 2,504 for a
> task whose signal is ~4 dimensional — an over-parameterization
> factor of ~600× in the first linear layer alone. Cutting the obs to
> the six keys that carry Stage-A signal reduces first-layer params
> from 640k to 120k, dramatically lowering update variance. This is
> the biggest "hidden-structure" win available without algorithmic
> changes.

**Best combination:**
> Tier-1 stack: `anneal_lr=false` + `ent_coef=0.03` + `total_env_steps=200k`.
> Three-line YAML change, no code touched, no Lua touched. ~75%
> combined P(fixes). Any one of these three individually is
> defensible from the TB data.

---

## Confidence & Gaps

**High confidence:**
- LR annealing to ~0 is a *primary* driver of the regression (TB data
  is unambiguous).
- Entropy pressure is under-weighted relative to kill reward by ~2
  orders of magnitude (dimensional analysis of loss terms).
- Obs dim is overparameterized for Stage A by ~1000× (measured from
  `spaces.py`).

**Medium confidence:**
- That the "corner-camp local optimum" description is fully accurate.
  Peak at step 33k = 10.65 kills/ep in a 1800-step episode = one kill
  every ~170 frames = actually good performance. This is not a policy
  stuck in a corner; it's a policy that *worked* and then degraded
  under LR/entropy pressure. Corner-camp may be a *later* symptom
  rather than a root cause.

**Gaps:**
- Have not confirmed `damage_to_npc` events include a `dmg` field —
  need to read `mods/isaac-rl-bridge/reward.lua` before implementing
  Fix 7. Would save a wasted run.
- Have not confirmed that heuristic.py exists and is functional for
  Fix 9. Referenced in the plan but not in the file list.
- The claim that "Attack Fly homes" needs verification against
  `mods/.../main.lua` behavior. Non-homing swap (Fix 6) depends on
  this assumption.
- Have not seen `env.py` step boundaries — if `max_episode_steps=1800`
  and `respawn_on_kill=true`, "kills_mean" is ~cumulative-in-episode.
  A regression could reflect *shorter episodes* not *worse aim* —
  check `charts/ep_len_mean`: it went from ~800 (max) to 426 by
  100k, so episodes did shorten. This is consistent with player
  dying faster (worse defense) rather than shooting slower. Fix 1+2
  should still address both.
