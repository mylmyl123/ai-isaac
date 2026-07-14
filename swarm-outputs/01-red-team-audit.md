# Red Team Audit: Isaac RL Post-Reset

Evidence base: `python/isaac_rl/{cleanrl_ppo.py,reward.py,env.py,vec_env.py,spaces.py}`, `mods/isaac-rl-bridge/{main.lua,obs.lua,reward.lua}`, `configs/curriculum.yaml`, `tb_stageA_20260714-153904.json` (100 096 steps, 391 update logs).

Observed regression numbers (from TB JSON, sampled):

| step | LR | kills/ep | ep_r | ep_len | ent(shoot) | top_frac(shoot) | approx_kl | clipfrac |
|-----:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3.0e-4 | 0.0 | 0.0 | 0 | 1.60 | 0.25 | 0.013 | 0.21 |
| 24 832 | 2.4e-4 | 8.1 | 6.5 | 610 | 1.04 | 0.38 | 0.012 | 0.17 |
| **33 024** | **2.05e-4** | **10.8** | **9.0** | **804** | 1.49 | 0.37 | 0.014 | 0.20 |
| 41 216 | 1.77e-4 | 6.9 | 5.4 | 541 | 1.53 | 0.33 | 0.008 | 0.11 |
| 82 176 | 5.4e-5 | 6.0 | 4.5 | 493 | 1.43 | 0.37 | **0.001** | **0.001** |
| 100 096 (end) | 4.8e-7 | 5.3 | 3.9 | 426 | 1.33 | 0.36 | **~0** | **0** |

Peak was reached at 33% of the way through the budget while LR was still ≥2e-4; the last 60% of training was net regression while LR collapsed to zero. `approx_kl` and `clipfrac` both went to zero by step 82k — **the policy stopped updating in any meaningful way for the last 20% of the run**.

---

## CRITICAL flaws (fix or nothing else matters)

- **LR anneal → 0 over the whole budget guarantees no recovery** — `cleanrl_ppo.py:262-266`. Linear anneal on a 100 k-step budget put LR at 2.4e-5 by step 90k. `loss/approx_kl` = 0.0000 and `loss/clipfrac` = 0.0000 at end. **The policy is mathematically frozen for the last 20 k steps.** Any noise-induced drift that occurred while LR was still active could not be reversed. Peak was at 33 k when LR = 2.05e-4; regression started at 41 k when LR = 1.77e-4; by the time collapse was undeniable at 65-85 k, LR was too small to escape. Fix: cap LR floor at ≥3e-5, use cosine w/ restarts, or don't anneal at all until convergence is observed.

- **Kill counting is double-counted on overkill** — `mods/isaac-rl-bridge/reward.lua:38-52`. The condition is `killed = hp_after <= 0`, where `hp_after = math.max(0, entity.HitPoints - amount)`. `entity.HitPoints` is the **current** value at callback time. If a damage event lands on an entity whose HP is already 0 (multiple simultaneous tear hits on the same tick, tear + tear-explosion effect, area damage that ticks per-frame after the kill), the callback re-fires with HP=0 and `killed = true`. Python's `cleanrl_ppo.py:314-319` and `reward.py:87-92` both accept every `damage_to_npc + killed=True` as a distinct kill. **Peak `kills=10.8` is very likely inflated** by whichever build-up of tear-timing produced overkill bunching, and part of the "regression" is the metric losing its inflation as the policy became more consistent. This is a metric-deception mechanism, not just a code bug. Fix: dedupe by entity `InitSeed`+`FrameCount`, or track a `killed_seeds` set per tick.

- **The player's own tears are NOT in the observation** — `mods/isaac-rl-bridge/obs.lua:81-118`, `build_projectiles`. It iterates `EntityType.ENTITY_PROJECTILE` (enemy projectiles) and `ENTITY_LASER`. `EntityType.ENTITY_TEAR` (== 2, player tears) is **absent**. Combined with `player.can_shoot` being a bare bool (no cooldown countdown) and no recurrence, the agent has **zero information about whether it just fired a shot, whether that shot is en-route to a target, or how long until it can fire again**. For a task whose entire optimal policy is aim-and-shoot-at-a-moving-fly, this is close to fatal. Fix: emit ENTITY_TEAR into the projectile slot, or add a "tears_in_flight" count + a discrete `frames_until_can_shoot` scalar.

- **Reward asymmetry makes death nearly free** — `python/isaac_rl/reward.py:19-22`. Peak episode: 10 kills × (+1) + 800 × (-0.001) + 1 death × (-1) = **+8.2**. A degraded episode: 6 kills − 0.5 − 1 = +4.5. The gradient from "don't die" is bounded at −1 while the gradient from "kill more" is unbounded. In Stage A with respawn-on-death, dying is a *reset button* that costs one unit of reward and comes with a free respawn of the fly. There is no reason to invest in survival past ~2 kills. Fix: (a) r_death ∈ [−5, −10] so death is unambiguously catastrophic, or (b) truncate episode on death without terminal signal so PPO can't amortize.

- **Three of five action factors are pure noise on Stage A, and they are collapsing anyway from gradient noise** — `python/isaac_rl/spaces.py:26`, `cleanrl_ppo.py:326-340`. `use_item`, `drop_bomb`, `use_pillcard` do nothing on Stage A (no active item, 0 bombs, 0 pills, in the mod `NO_ONESHOT` isn't set so presses go to the engine but no effect). But every rollout still records these actions and computes `Σ log π(a_k)` across all 5 factors. TB confirms the leak:

  ```
  entropy_per_factor/use_pillcard   first=0.692 → last=0.313   (−55%)
  entropy_per_factor/drop_bomb      first=0.692 → last=0.480   (−31%)
  entropy_per_factor/use_item       first=0.692 → last=0.505   (−27%)
  ```

  These factors have zero causal effect on reward, yet they lose ~half their entropy from noise-driven policy gradient. This means (i) the policy gradient's implicit variance is high enough to overfit useless features, so it is *definitely* overfitting on `move`/`shoot` too, and (ii) the entropy bonus is spending its budget on preserving irrelevant factor randomness instead of preserving useful exploration. Fix: mask heads at sampling time (`bombs==0 → drop_bomb ≡ 0`; unavailable actives ≡ 0; etc.), drop the log-prob contribution from masked factors from the PPO loss.

- **Stage A is fundamentally the wrong training task** — `configs/curriculum.yaml`, `mods/isaac-rl-bridge/main.lua:180-260`. Sealed room + a homing enemy that comes to the player means:
  1. Movement is unnecessary for kill acquisition (fly comes to you).
  2. Optimal strategy is to sit still, aim at whichever cardinal direction the fly last approached from, and fire.
  3. There is no navigation, no dodging, no exploration, no item use to learn.
  4. Random-angle spawn (line 227) is the only source of variation, so the policy either overfits to one shoot direction (observed: `action_top_frac/shoot=0.36` and rising) and dies to unfavored angles, or maintains random shooting and gets sparse credit.

  **Anything you learn on Stage A does not generalize to Stage E**, because Stage E requires movement toward enemies, obstacle avoidance, and shot-line reasoning — none of which Stage A trains. The curriculum's own comment (`curriculum.yaml:11-13`) admits *"Degenerate task — policy learns to corner-camp because homing fly comes to player. SKIPPED."* — yet the run was Stage A. Fix: skip Stage A entirely (its own author agrees), or make it a stationary-enemy shooting-range so shot direction is actually informative.

---

## MAJOR flaws (real problems, not fatal)

- **No recurrence in the policy for a partially-observable environment** — `cleanrl_ppo.py:100-138`. MLP with 4-frame player history is not memory. Missing tear positions, ambiguous can_shoot state, and no way to remember "the fly was here 200 ms ago, so it's now approximately there." A GRU or Transformer over the last 8-16 frames would be a strict upgrade. The design comment (`cleanrl_ppo.py:11-13`) says "Add recurrence back if partial observability turns out to be the bottleneck" — it did.

- **Rollout is much shorter than one episode** — `cleanrl_ppo.py`, `curriculum.yaml`. rollout_length=128, n_envs=2 ⇒ 256 samples per update. Episode length is 300-800 ticks. **Every rollout ends mid-episode**, so 100 % of updates rely on the value bootstrap `V(s_{T+1})` for most of the samples. With a shared, poorly-normalized value head (see below), bootstrap error dominates the advantage estimate in the middle of learning. Fix: rollout_length=512 or 1024, or n_envs=8 with rollout=128.

- **Value head is shared with the actor and trained with `vf_coef=0.5`** — `cleanrl_ppo.py:113,289`. Value loss magnitudes stayed at 0.01-0.08 across the run. That looks like "critic converged" but it is more consistent with **the critic predicting the mean and being right most of the time** because r=+1 kill events are rare (≈10 / 800 ticks = 1.25 %). Advantages therefore reflect very noisy TD residuals rather than accurate value estimates. Separate critic (independent 256×256 trunk) removes gradient interference and lets the critic actually specialize.

- **Advantage normalization at minibatch level, not batch level** — `cleanrl_ppo.py:288`. `mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)` on 64-sample minibatches. With ~1 kill event per ~150 ticks, most minibatches contain 0 or 1 high-advantage sample. Minibatch-level norm then rescales that lone sample against near-zero neighbors, producing artificially amplified advantages when the sample is present and near-uniform noise when it is not. Standard CleanRL does this, but standard CleanRL uses 2048×8 rollouts, not 128×2. Fix: normalize once over the full batch of 256.

- **Terminal-obs handling on mod-restart path returns the NEW episode's first observation as the "terminal" obs** — `env.py:207-232`. On the mod_restart branch, `reconnected_raw` is the post-restart handshake obs (first tick of a fresh episode), not the pre-death terminal frame. That obs is then wrapped with `_build_obs` and returned with `terminated=True`. Downstream, `vec_env.py:52-56` stashes it as `terminal_obs` and DreamerV3 (or any future critic training that consumes terminal_obs) will train the continue/reward decoder on the *wrong* frame. PPO happens to be safe because `done=True` zeroes the bootstrap in `Rollout.compute_gae`, but the design invites bugs. Fix: keep the last pre-death raw obs in `self._last_raw` on every successful step and return that on the mod_restart terminal.

- **Broadened HP-based death detection can double-fire** — `env.py:230-240` (mod_restart adds r_death if `not shaper.state.dead`) and `reward.py:99-104` (HP-based fallback also adds r_death and sets `state.dead=True`). Order of operations: shaper runs first on the last live tick, sees hp_red=0 and hp_soul=0, sets dead=True, adds r_death. Next tick the mod cycles the socket, env sees the ConnectionError, goes into the mod_restart branch, checks `if not shaper.state.dead` — dead is True, so it skips the extra r_death. That path is safe. **But** the `hp_red==0 and hp_soul==0` fallback triggers on any tick where the raw obs shows those values, and Isaac exposes hp_red=0 for characters like The Lost (1-HP form) as a normal state (Stage A uses Isaac only, so this happens to be safe today, but the fallback code is a latent bug for later stages).

- **Kill counter in the trainer uses events from `info["raw"]`** — `cleanrl_ppo.py:314-320`. `info["raw"]` on the mod_restart terminal path is the NEW-episode raw (per bullet 5 above). So the kill counter can miss any kill events emitted on the death tick, and can conversely pick up spurious first-tick events of the next episode as belonging to the ending one. Combined with the overkill double-count, the `kills/ep` metric on TB is not a trustworthy training signal — it has at least two systematic biases in different directions.

- **`_flat_obs` computes `flatten_dict_obs(o)` twice per step, per env** — `cleanrl_ppo.py:74-79`. Not correctness, but the sync vec env is stepping ~15 Hz on 2 envs; that's ~9 M redundant dict flattens per 100 k-step run. Not the bottleneck, but symptomatic of unaudited hot paths.

- **The 733-dim passives MultiBinary is ALL-ZERO on Stage A** — `spaces.py:88`. First layer of the MLP is `Linear(2606, 256)` and >900 of those inputs never fire on Stage A. Not damaging (zero inputs → zero contribution regardless of weight) but every trained weight into those dead inputs is dead memory that a future stage will backfill with a totally different distribution — the network implicitly assumes a fixed input distribution that will change abruptly on stage transition. This is a big deal for the resume-across-stages design in the curriculum comment.

- **Enemy features flatten variable-length entity list into fixed 24×16 slots** — `spaces.py:127-131`, `obs.lua:52-79`. The MLP has to learn slot-invariance from scratch: enemy in slot 0 vs slot 5 should yield the same policy, but MLP treats them as independent inputs. Attention-over-entities would be structurally correct here. Given the rollout size (256 samples/update) it's questionable whether the MLP has enough signal to learn slot invariance for anything past slot 0.

- **rewards are NOT normalized (no return normalization, no reward scaling)** — nowhere in the training loop. Reward magnitudes span ~[-1, +10] per episode. GAE returns feed directly into the value MSE loss. Combined with the advantage normalization done per-minibatch, this is inconsistent — advantages are unit-scale but returns are not. Standard PPO practice is per-batch return standardization or a running reward-scale tracker.

---

## MINOR flaws (worth fixing eventually)

- **`spaces.py:PLAYER_DIM=40` but only 20 fields are actually written** — `_PLAYER_FIELDS` has 20 names, `PLAYER_DIM=40`. The trailing 20 slots are always zero. Not harmful, but wastes half the player-vector budget.

- **`ent_coef=0.01` is a global weight applied to summed entropy over 5 factors**. Effective per-factor coefficient is ~0.002 (for the 5-way shoot factor after summing). Standard MultiDiscrete PPO papers use ent_coef proportional to `1/K` or per-factor coefficients. The single scalar here means: as factors collapse (particularly the useless ones), total entropy drops linearly and the exploration bonus proportionally weakens, which accelerates further collapse. Positive-feedback loop.

- **Character encoding uses 35-way one-hot on Stage A where character is always Isaac (index 0)** — `spaces.py:_decode_character`. 34 dead inputs.

- **`_decode_active_items` divides item ID by 730 to normalize** — `spaces.py:340-360`. Item IDs are categorical, not ordinal. Item 100 and item 200 are more different than item 100 and item 101, but the encoding treats them as similar (0.137 vs 0.274 vs 0.137 vs 0.138). Should be one-hot or embedding.

- **`build_room_grid` uses module-level shared arrays and returns them wrapped in a fresh table each call** — `obs.lua:132-175`. The comment says "Do NOT mutate these arrays elsewhere," but they are mutated in place on every call. If any consumer holds a reference across calls (e.g., a co-routine, a debug snapshot, a JSON encode that yielded), it will see the arrays overwritten. Isaac Lua is single-threaded so it's safe today, but fragile.

- **`_try_accept_after_close` wait_s=3.0 and mod's `frames_since_game_started<90` guard interact weirdly**. Mod ignores Python's reset command for 90 ticks (~1.5 s at 60 Hz). Python's `env.reset()` sends a `{"reset": True}` after every terminated episode. Mod drops it, but Python's env.py then closes the client socket and waits for a NEW accept — the mod won't reconnect on its own until MC_POST_GAME_STARTED fires again. If the mod's own death handler already reconnected within the 90-tick window, this second reset can leave the pipeline in a state where Python is `accept()`-waiting and the mod is still connected to a now-closed socket. Real symptom: intermittent `Isaac did not connect within 300s` timeouts. Hard to see in Stage A because deaths are frequent enough that the next `MC_POST_GAME_STARTED` fires quickly.

- **`SPATIAL_DIM=8` unit-vector-to-open-door is always zero on Stage A** because doors are sealed. That's fine, but along with all-zero passives / cards / pills / trinkets it means the flat obs has >1500 always-zero dims out of 2606 on Stage A. **58 % of the input is dead data**.

- **`orthogonal_(head.weight, gain=0.01)` on the action heads and `gain=1.0` on the trunk and value head** — `cleanrl_ppo.py:115-121`. Standard, but the value head gain=1.0 makes the initial critic output ~O(1) on random obs while the actor outputs ~O(0.01). Value loss dominates the initial gradient signal to the shared trunk for the first few updates.

---

## SMOKING GUNS in the observed data

The 10.8 → 5.3 regression has three candidate mechanisms; the evidence best supports #1 and #2 acting jointly. #3 is a plausible metric-inflation contributor at peak.

### Hypothesis 1 (highest confidence): LR annealing froze the policy in a shooting-direction local minimum.

Evidence:
- Peak at step 33 024 (33% of budget). Regression starts at step ~41 000.
- `loss/approx_kl` drops from ~0.014 at step 33 k to 0.001 by step 82 k to ~0 at end.
- `loss/clipfrac` drops from 0.20 at step 33 k to 0.001 by step 82 k to 0 at end.
- LR at step 82 k = 5.4e-5; at 90 k = 2.4e-5; at 100 k = 4.8e-7.
- Once `clipfrac→0` and `approx_kl→0`, the policy is by definition not updating in any direction. Whatever suboptimum it settled into can't be escaped.

This is not "the policy learned then unlearned"; it is **"the policy learned, then noise pushed it a little in a bad direction while it could still move, then the LR anneal glued it there."**

### Hypothesis 2 (high confidence): Shoot factor entropy collapsed to a narrow direction preference, and the environment's random fly-spawn angles punish that preference.

Evidence:
- `entropy_per_factor/shoot`: 1.60 → 1.33 (max is ln 5 = 1.61). Loss = 17 % of max entropy.
- `action_top_frac/shoot`: 0.25 → 0.36 (uniform is 0.20). Policy is ~1.8× as biased toward one direction as random.
- `stage0_spawn_fly` in `main.lua:220-270` spawns the fly at a random position 200-500 px from the player at a random angle every time.
- If the shoot head fires "down" 36 % of the time regardless of fly position, then fly-spawns in the upper 3/4 of the room (75 % of cases) get near-zero tear-on-target rate for ~5-30 frames, during which the fly closes and hits the agent.
- Ep length dropped from 800 → 426 alongside the shoot-entropy collapse — **the agent is dying faster because it can't hit off-axis flies**.

The action-space design forces the policy to *choose one* shoot direction per tick (not a distribution over directions). Any bias in the head amplifies into missed shots for adversarial spawn angles.

### Hypothesis 3 (medium confidence): Peak "kills=10.8" was partially inflated by overkill double-counting; some of the "regression" is metric deflation as tear-timing became more consistent.

Evidence:
- `reward.lua:38` uses `killed = math.max(0, entity.HitPoints - amount) <= 0`. HitPoints is the pre-damage value; if two damage events land the same tick and both fire the callback, both can compute `hp_after=0` and both set `killed=True`.
- Attack Fly has ~4-5 HP against Isaac's base 3.5-damage tears. It usually dies in one hit. Two-tear same-tick hits at close range on a low-HP enemy are entirely plausible and would double-count.
- Peak `ep_len=804` and `kills=10.8` together imply one kill every 74 ticks (~5 seconds). That is *unrealistically fast* for a homing fly that takes >200 ticks to close from 500 px. A simpler explanation: the peak "kill count" is inflated by ~1.5-2×.
- After the entropy collapse (Hypothesis 2), the policy became a more consistent one-shot killer with cleaner tear timing, producing fewer simultaneous hits and less double-counting.

**This is the non-obvious deception mechanism.** The operator watches `kills/ep` in TensorBoard and treats 10.8 as ground truth. It might be closer to 6-7 true kills with the rest being overkill artifacts. The "regression" from 10.8 → 5.3 is partly a policy problem and partly the metric becoming *more honest* as tear timing tightened. The system is fooling its operators about *when it was actually good*.

### Hypothesis 4 (lower confidence, but worth checking): Useless-factor entropy collapse consumed the ent_coef budget.

Evidence:
- `entropy_per_factor/use_pillcard` dropped 0.69 → 0.31 (−55 %, from a factor that has ZERO causal effect on reward on Stage A).
- Total entropy budget is a global `ent_coef * summed_entropy`. As the three useless factors leak entropy toward one arbitrary value, that lost entropy is invisible to the actual task but still counts against the coefficient budget. The bonus that *would* have kept shoot exploration alive was instead being spent (fruitlessly) trying to keep pillcard exploration alive.
- Net effect: the ent_coef=0.01 that looks reasonable if you assume 5 factors × meaningful actions is effectively closer to 0.004 per meaningful factor.

---

## Confidence and gaps

- **High confidence:** items 1-4 (LR anneal, kill-counting bug, missing tears, reward asymmetry). All directly readable from the code with matching TB evidence.
- **Medium confidence:** the terminal-obs handling contamination is a real design issue but PPO's `done`-mask makes it non-fatal today. Would matter more for Dreamer or offline RL.
- **Gaps I did not verify:**
  - I did not run the code to confirm the overkill-double-count actually fires in practice. It is a code-shape bug; whether it fires depends on Isaac's `MC_ENTITY_TAKE_DMG` semantics on 0-HP entities. **Recommend a quick unit test: fire two damage events into a mock entity with HP=1 on the same tick, count `killed=True` occurrences.**
  - I did not measure the actual per-episode `kills` distribution from `episodes.csv` — that would let you check whether peak-kill episodes have a suspicious multi-kill-per-tick pattern (e.g., 3 kills in 2 consecutive ticks).
  - I did not check whether `stage0_spawn_fly`'s "8 random candidates" ever falls through to the fallback (~300 px from player) frequently enough that spawn distance is bimodal. If so, the reward variance across episodes is not just from policy noise but from environment noise.

## What to fix first, in order

1. **Cap LR anneal floor at 3e-5** or don't anneal until convergence is observed. This is a one-line change and should be tested in isolation before anything else.
2. **Fix the kill-counting overkill bug** in `reward.lua` (dedupe by `entity.InitSeed` per tick, or drain a "killed_seeds_this_tick" set).
3. **Emit `ENTITY_TEAR` into the projectile stream** so the agent can see its own shots.
4. **Mask the three dead action factors on Stage A** (or drop them from the action space for Stage A entirely).
5. **Skip Stage A.** Its own author already decided it is degenerate. The next honest test is a Stage-B (3 flies + require room clear) or a stationary-shooting-range task where shoot direction genuinely varies with fly position.

If items 1-5 are done and Stage B still regresses in the same pattern, then the architectural bets (add recurrence, separate critic, larger rollout, attention over entities) become the next round.
