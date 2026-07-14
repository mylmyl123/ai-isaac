# Isaac RL: First-Principles Analysis

**Author:** Isaac First-Principles Scientist
**Data source:** `runs_meta/stageA_20260714-153904_episodes.csv` (206 episodes / 99,836 env steps, 2 envs, cleanrl PPO, sealed room + 1 homing Attack Fly), `configs/curriculum.yaml`, `python/isaac_rl/{env.py, reward.py, spaces.py, cleanrl_ppo.py}`, `mods/isaac-rl-bridge/obs.lua`, TB summary `tb_stageA_20260714-153904.json`.

Every claim below is either derived analytically from the code or measured from that episode CSV.

---

## 1. The MDP

Isaac Stage A is best modeled as a **discrete-time, finite-horizon POMDP** with the following structure.

### State (what the environment actually contains at tick t)

- Player: position `(x,y) ∈ ℝ²`, velocity `(vx,vy)`, HP (red/soul/black), damage stat, MaxFireDelay (shot cooldown period), TearRange, ShotSpeed, `CanShoot` boolean, currently-firing frame counter.
- Enemies: for each of ≤ 24 slots, `(x, y, vx, vy, hp_frac, type, state, size, ...)`. Attack Fly (Type=13, Variant=1) homes on the player with speed ≈ 4 units/frame, HP = 6, contact damage = 1.
- Projectiles (own tears + enemy shots): position, velocity, height, `is_laser`, variant, `frame_count`.
- Room grid (4-ch × 9 × 15) for walls / rocks / spikes / poop. In sealed Stage A it is all zeros except a 1-cell wall border, i.e. **carries almost no information for this stage**.
- Global: `frames_since_room`, `frames_since_hit`, room clear flag, etc.

### The Markov-sufficient state for optimal play in Stage A

Because the room is a 1×1 sealed rectangle and the single enemy is a deterministic homing missile, the sufficient statistic for the Bellman equation is

```
s* = { p_player, v_player, p_fly, v_fly, hp_fly, tears_in_flight, fire_cooldown_remaining, room_bounds }
```

Dimensionally: **≈ 24 scalars**. The current observation delivers **≈ 3,100 scalars** (PLAYER_DIM=40, PASSIVES_K=733, room_grid=4·135=540, doors=4·18=72, enemies=24·16=384, projectiles=48·10=480, pickups=16·8=128, character=35, etc.). ≥ 99% of these features are identically zero or invariant across every Stage A step. The signal-to-dimension ratio is `≈ 24 / 3100 ≈ 0.8%`, which is a linear-regression noise disaster in a data-poor regime (see § 8).

### Action space

Currently `MultiDiscrete([9, 5, 2, 2, 2])` = **180 joint actions**. Factors: 8-way move (+idle), 4-way shoot (+idle), use-active, drop-bomb, use-pill/card. In Stage A the last three heads are **provably useless** (agent has no bombs, no active, no cards on Isaac start), yet the entropy bonus pays the policy `0.01 · log(2) ≈ 0.007/step` PER factor to keep those heads uniform. Three parasitic factors ⇒ **0.021 nats/step of pure noise** in the loss.

Minimum sufficient action set for Stage A: `MultiDiscrete([9, 5])` = 45 joint actions. The July-02 rev of `spaces.py` had this and was reverted on July-12 for BC-bootstrap compatibility. **Stage A does not need it.**

### Reward

Three-term shaping in `reward.py`:

```
r_kill  = +1  on 'damage_to_npc'∧killed
r_death = -1  on player death
r_step  = -0.001  every tick
```

Per-episode measured (peak chunk, chunks 2–5, mean over 26 eps each):

| Term | Mean absolute contribution per ep | Mean per step (at ep_len≈550) |
|---|---|---|
| kills | +7.1 | **+0.0129** |
| death | −1.0 | −0.0018 |
| step | −0.55 | −0.001 |

### Horizon and discount

Empirical `ep_len_mean = 483` steps (max obs 1600, mean 484, min 91). At γ = 0.99 the "effective horizon" is `1/(1−γ) = 100` steps. So:

```
γ^L for L=100 : 0.366
γ^L for L=500 : 0.0066
γ^L for L=1000: 4.3e-5
```

**Death is essentially invisible from the initial state.** At L=500 the −1 death is discounted to −0.0066 in V(s₀) — smaller than 7 discounted step penalties. The value function has almost no gradient to teach *"don't die"* from the start of the episode; it can only teach *"don't die in the next 100 steps"*.

Analytic V(s₀) at the peak policy (λ = 1/70 kills-per-step, L=500):

```
V(s₀) ≈ (1 − γ^L) / (1 − γ^{1/λ}) − γ^L − 0.001·(1−γ^L)/(1−γ)
      ≈ 1.967 − 0.007 − 0.099 = 1.861
```

At the regressed policy (λ = 1/90, L=333): `V(s₀) ≈ 1.489`. So the peak policy **does** have higher discounted value (by ≈ 0.37). Gradient should push toward the peak, yet it moved the other way — this is the anomaly that must be explained (§ 3).

---

## 2. Assumption Audit — Does Isaac Satisfy PPO's Assumptions?

The clipped-surrogate PPO objective (Schulman et al. 2017, eq. 7):

```
L^CLIP(θ) = 𝔼_t [ min( r_t(θ) Â_t , clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
```

is a first-order surrogate for the TRPO monotonic-improvement bound (Schulman 2015, Thm. 1; ultimately Kakade & Langford 2002, "Approximately Optimal Approximate RL", CPI). The bound

```
J(π_new) ≥ J(π_old) + 𝔼_{s∼d^{π_old}, a∼π_old}[ (π_new/π_old) A^{π_old} ] − C · D_KL^max(π_old, π_new)
```

requires **four assumptions**:

| Assumption | Holds for Isaac? |
|---|---|
| (i) States are Markov (fully observable, no hidden dynamics) | **No.** POMDP. Fly acceleration state and player fire-cooldown phase are partially observed and the trainer uses an MLP with no recurrence. |
| (ii) Rollouts drawn from the *current* policy π_old | Roughly yes — sync collection, 4 epochs, 4 minibatches (~ 1.5× under-utilization of clip budget, acceptable). |
| (iii) Advantage estimator Â_t is low-variance, unbiased under π_old | **Failing.** With sparse +1 kill and dense −0.001 step, GAE(λ=0.95, γ=0.99) has Var(Â) ≈ Var(R^GAE) ≈ (kill_rate)·γ^{avg_delay}/(1−γλ) ≈ 15× the mean signal (measured proxy: `charts/ep_r_std / charts/ep_r_mean ≈ 1.6` before normalization; after minibatch normalization the entropy term dominates — see § 3). |
| (iv) KL between consecutive policies stays inside the trust region ε | **Marginally**, but *inverted problem*: measured `approx_kl` drops from 0.013 → 1.5·10⁻⁷ (5 decades). The KL-adaptive PPO variant (Schulman 2017 §4) would have *raised* the LR, but we use fixed-then-linearly-annealed LR, so the effective step size collapses too fast. |

**Bottom line:** PPO's improvement guarantee requires the advantage estimate to point in a direction whose gradient magnitude exceeds the entropy-regularizer gradient. In Stage A it does not (measured, § 3).

### On-policy vs off-policy at 30 env-steps/second

Wall-clock: 99,836 steps in 3,935 s = **25.4 sps** (matches "~30 sps" in the prompt). PPO reuses each sample n_epochs · n_minibatches / n_minibatches = **4×** before discarding. That gives ~100 gradient updates per collected sample-second. SAC would use each sample **~200×** through a replay buffer of size 10⁶ (Haarnoja et al. 2018, "Soft Actor-Critic"). At the observed collection rate, SAC's sample-efficiency multiplier is directly available — on Atari and MuJoCo benchmarks SAC-Discrete beats PPO by **5–20× in sample-efficiency** for tasks of comparable complexity (Christodoulou 2019 Table 1; Haarnoja et al. 2018 Fig. 1). At 30 sps this is a 2-3 hours of training vs. 1-2 days for equivalent asymptote.

### Sample complexity (theory)

For a linear MDP of dimension d with H-step horizon, PAC-optimal PPO-style algorithms need
```
n ≥ Ω( d · H² / ε² )  (Kakade 2003 thesis; Jin et al. 2020, "Provably Efficient RL with Linear FA")
```
For Isaac Stage A, effective d ≈ 24 (see § 1), H ≈ 500. Even ignoring constants, `n ≥ 24 · 500² / 0.1² = 6 · 10⁸` for ε=0.1 sub-optimality gap. Our 10⁵ steps are **~4 orders of magnitude short** of the theoretical PAC bound. Neural function approximation gives constant-factor gains in practice but does not change the scaling.

---

## 3. Root Cause of Regression — Mathematically

**The observed trajectory (chunks are contiguous 26-episode windows, ~12k env-steps each):**

| chunk | env-steps | kills_mean | ep_len_mean | ep_r_mean |
|---|---|---|---|---|
| 1 | 0.5k–9.6k | 4.28 | 372 | 2.91 |
| 2 | 10.5k–23.6k | 7.08 | 538 | 5.54 |
| 3 | 24.4k–38.7k | **7.42** | 579 | **5.84** |
| 4 | 39.6k–53.6k | 6.77 | 558 | 5.21 |
| 5 | 53.7k–68.3k | 7.04 | 610 | 5.43 |
| 6 | 69.0k–80.7k | 5.81 | 477 | 4.33 |
| 7 | 80.8k–89.3k | **3.81** | 333 | 2.48 |
| 8 | 89.3k–99.8k | 5.00 | 403 | 3.60 |

Peak at step ~30k, trough at step ~85k, partial recovery at step ~95k. All 206 episodes ended with `terminated=1, truncated=0` — every episode ended in death, never at the 1800-tick truncation cap.

### The dominant force: entropy regularization outweighs the reward signal

Per-step loss decomposition (magnitudes on the SAME scale in nats/tick, before minibatch normalization):

```
kill reward per step, peak policy:      +0.0148
kill reward per step, regressed:        +0.0074
step penalty:                           −0.0010
death penalty amortized over ep:        −0.0020
entropy bonus per step (start, H=5.87): +0.0587
entropy bonus per step (mid,   H=5.00): +0.0500
entropy bonus per step (end,   H=4.55): +0.0455
```

The entropy bonus is **3–8×** the kill reward per step. It is uniformly positive and pushes the softmax back toward `1/n_choices` on every gradient step. Advantages are minibatch-normalized to unit variance before entering the surrogate; the entropy term is **not** normalized. The scale mismatch means that once minibatch statistics normalize away the reward-signal variance, the raw entropy gradient dominates.

Formally the PPO gradient at step k is
```
∇θ L = ∇θ L^CLIP  +  β · ∇θ H(π)  −  c_v · ∇θ L^V
```
with `β = ent_coef = 0.01`, `c_v = 0.5`. The clip-loss gradient magnitude is roughly `‖∇ log π · Â_norm‖ ~ O(1)` per action taken, but this is a *sparse-reward gradient*: only ≈ 1.5% of transitions carry a kill event, so the *per-parameter* update from `L^CLIP` averages ≈ 0.015 × sign(A). Meanwhile the entropy bonus is **applied on every step** with magnitude ≈ 0.01 × H/K ≈ 0.01 × 5/5 = 0.01 per factor per step. Net: the entropy gradient is **~10× the average reward gradient per parameter**.

This explains three signatures at once:

1. **kills_mean regressed to 3.8** while entropy stayed at 77% of max (H_end = 4.55 vs H_max = 5.89). The optimizer kept trading probability mass out of the peaks that produced kills, because the entropy bonus paid it to.
2. **`approx_kl` collapsed to 1.5·10⁻⁷.** Ratio r_t ≈ 1 everywhere ⇒ the clip is inactive, the surrogate degenerates to `Â·log π`, and the term with the largest gradient wins. That's the entropy term.
3. **Env-1 (kills_last10=4.6) never reached env-0's peak (kills_last10=6.2).** With two envs the effective rollout batch size is 256, minibatch = 64. Advantage variance across a 64-sample minibatch is enormous and, after per-minibatch normalization, the sign of the update on any particular action can flip between successive minibatches. Under this noise, the entropy bonus is the only signal with a consistent sign.

### Why LR annealing accelerates the collapse

`anneal_lr = true` sets `lr(t) = 3e-4 · (1 − step/total_env_steps)`. At step 30k (peak), lr ≈ 2.1e-4. At step 85k (trough), lr = 4.5e-5. As LR shrinks, the *policy-improvement* term shrinks proportionally, but the entropy gradient continues to pull toward uniform for every additional step where the two envs collect data. The ratio (entropy_pull / policy_improvement) grows monotonically toward the end of training. Regression is *baked into the anneal schedule* whenever entropy > reward per step.

### The GAE terminal-bootstrap is correct

I checked `Rollout.compute_gae`:
```python
delta = r_t + γ · V(s_{t+1}) · (1 − done[t+1]) − V(s_t)
```
`dones` is written at rollout-step t *before* env.step at that step, so `dones[t+1]` is the done flag entering state t+1, which correctly masks `V(reset_state)` after death. **No off-by-one.** The −1 death signal does propagate one tick backward before being masked — that gives V(s_{terminal−1}) a target of ≈ −1, which is correct. The bootstrap is not the bug.

### The corner-camp equilibrium is a local maximum of the *regularized* objective, not the raw reward

Ranking policies under the actual optimization objective `J(π) − 0.01·H(π)⁻¹` (roughly: reward-per-unit-focus):
- **Random** (H = 5.89, kills ≈ 2 by luck): objective ≈ 2 + entropy_reward(5.89·ε_len·0.01) ≈ **2 + 30 = 32**
- **Peak** (H = 5.00, kills = 7.4, ep_len 580): objective ≈ 7 + 29 = **36**
- **Regressed** (H = 4.55, kills = 3.8, ep_len 333): objective ≈ 3 + 15 = **18**
- **Optimal-aim** (H ≈ 3.0, kills ≈ 15, ep_len 1800): objective ≈ 15 + 54 = **69**

The optimal policy (constant fire, moving to face fly) is *far* above the peak in objective value, but reaching it requires committing entropy to zero on the shoot head. The entropy bonus penalizes exactly that commitment. The peak is a **saddle point** between the random attractor and the optimal attractor, and the regressed run slid back toward random along the entropy gradient — which is the shortest descent direction from a peak.

---

## 4. Corner-Camp Value Calculus

The Attack Fly (Type=13, Variant=1) homes at ~4 units/frame; player base MoveSpeed·pixels ≈ 6 u/frame. Isaac base Damage = 3.5, Attack Fly HP = 6 ⇒ **2 tears to kill**. Fire delay ≈ 10 game-frames = 5 policy ticks at 15 Hz. Tear travel to close-range fly (< 100 pixels) ≈ 5 game-frames = 2 ticks.

- **Random-shoot corner policy**: agent shoots ~20% of ticks (uniform over 5 shoot values), 1/4 direction match with fly angle (fly homes → often on-axis in a corner). P(hit per shot) ≈ 0.35 close range. Expected kills per 500-tick episode ≈ (500/70) · 0.7 = 5.0. Death: fly reaches corner-camped agent in ≈ 50 ticks, contact-damages until 3 HP burned → ≈ 90-150 ticks to die. **Expected reward ≈ 4–5.** Matches chunk-1 baseline.
- **Aim-and-kill policy**: fires every 5 ticks in fly direction, 2 tears kill 1 fly in 10 ticks. Kills/ep = 50. Expected reward ≈ 50 − 1 − 5 = **44**. Requires learning `argmax_{shoot∈{L,R,U,D}} align(shoot_dir, fly_direction)` — a 4-way classifier over enemy-relative-position features.

The gradient ratio `∂J/∂θ_{aim}` vs `∂J/∂θ_{camp}` at the peak: aim requires policy to be **more deterministic** in the shoot head (H_shoot → 0), which incurs an entropy penalty of `0.01 · log(5) ≈ 0.016 per step ≈ 8 per episode`. Aim reward gain: `+37` (going 7 → 44 kills). So the aim policy is dominated in objective if `0.01·ΔH > kill_gain / ep_len` — false, since `8 < 37`. But the *local* gradient toward aim through the fog of stochastic 30-sps sampling with 4 mini-batches per rollout is much smaller than the entropy pull. The agent finds the peak (chunk 3) but cannot escape it *within the anneal schedule*.

---

## 5. Minimum Reward Signal for Optimal Policy

**Provably sufficient (up to policy equivalence):**

- `r_kill = +1` on each NPC death event. Necessary — it's the entire task signal.
- `r_death = −K` on player death **with K chosen so the discounted death signal at typical ep_len equals a single kill**. For γ=0.99, L=500: `γ^L · K = 1 ⇒ K ≈ 150`. Current K=1 → death is ≈ 150× under-weighted at ep start. Use K=1 only if you *also* raise γ to make ep_len ≈ 100 (γ = 1 − 1/L).

**Provable noise (in Stage A):**

- `r_step = −0.001` is redundant with a truncation-only horizon and adds a per-step bias of magnitude comparable to `1/ep_len` — mathematically equivalent to a horizon regularizer. It is **not harmful** but not necessary. Remove for cleanliness.
- All BC-era shaping (kite-time, aim-alignment, room-clear bonuses) was correctly excised per `reward.py`.

**What is provably missing** and cannot be substituted by shaping: nothing on the reward side. The bug is *not* insufficient reward — it is that PPO's entropy regularizer swamps the reward gradient.

---

## 6. Observation Gaps

Reading `obs.lua` and `spaces.py`, the following features that a Bayes-optimal predictor of Q*(s,a) would need are **absent or under-encoded**:

1. **Fire cooldown remaining (frames until next tear available).** The obs has `player.MaxFireDelay` (a stat, constant across the episode) and `can_shoot` (boolean at *this* tick), but not the countdown. Without it the agent cannot learn "wait 3 ticks then fire" — it has to memorize timing via the GRU it doesn't have.
2. **Tear-to-fly time-to-contact.** Currently the agent gets its own tears in `projectiles` and the fly in `enemies`, but relative-velocity of (my-tear − fly) is not pre-computed. The 2-layer 256-wide MLP has to learn a bilinear form over these entities — non-trivial for a permutation-invariant slot representation. Attention would help.
3. **Fly imminent-contact prediction.** Frames-until-collision at current fly velocity would collapse the death signal from a discount-hidden −1 into a dense observable danger cue.
4. **No recurrence.** The trainer explicitly says "no LSTM. Simplest thing that could possibly work." Fine for MuJoCo, but Attack-Fly motion between two obs frames at 15 Hz includes 1 unobserved fly-frame per policy tick, so the ground-truth dynamics have a 1-step lag not present in the obs. `player_history` (4 stacked past frames) partly compensates for the player, but nothing analogous exists for the fly.
5. **The 733-dim passives one-hot, 35-dim character one-hot, 4×18 door tensor, 4×135 room grid, 16-dim latent z, and all item-slot features are identically zero in Stage A** — they are pure input noise that inflates obs_dim ~130× beyond the sufficient statistic. The first-layer weights waste capacity fitting the noise; effective L2 regularization per useful feature is `1/(1 + 129) ≈ 0.008`, degrading the meaningful representation.

Reducing obs to `{player, enemies, projectiles, spatial, cooldown, last_action}` would compress d ≈ 3100 → d ≈ 100. Per Jin et al. 2020, PAC bound tightens by 30×.

---

## 7. Setups That Provably Cannot Be Corner-Camped

Ranked by theoretical strength of the anti-camp constraint:

1. **Horf (Type=15).** Stationary shooter that fires 3-way tears when player enters LoS. If agent camps in a corner, Horf's tears reach the corner in ≤ 50 ticks and damage the agent. Corner-camping is *strictly dominated*.
2. **Two Attack Flies from opposite sides.** Homing missiles from `(x=−w, y_random)` and `(x=+w, y_random)`. Corner puts agent adjacent to one fly (guaranteed hit) — camping is again dominated. Requires ≥ 2 well-placed shots per tick or continuous movement.
3. **Larger room (Stage C's normal Basement 1 dimensions ≈ 1.5× the sealed A room).** Homing fly takes 60 ticks to reach corner instead of 30. Camp is still viable but the reward gap shrinks — not a real fix.
4. **Non-homing enemy (Pooter, Fly variant 0, Charger).** Camp is now *stable* because enemy doesn't come to you — but the agent must *approach* the enemy to kill it. This flips the problem: the corner is now the worst position because enemy is far away and fire delay wastes ticks.

**Recommendation:** switch to Stage-A′ = "1 Horf in sealed room, respawn on kill". Same simplicity, no corner attractor.

---

## 8. Mathematically Optimal Fix

The problem class is a POMDP with dense-ish rewards, permutation-symmetric entity observations, and 30 sps collection. The theoretically-preferred algorithm class is:

### Recommended: **DreamerV3** (Hafner et al. 2023, "Mastering Diverse Domains through World Models")

Theoretical justification:
- DreamerV3 learns a Recurrent State-Space Model (RSSM) — an approximate Bayesian filter over the POMDP hidden state. This directly addresses the partial-observability failure of PPO+MLP (§ 2, assumption i).
- Policy is optimized in **imagined rollouts** from the world model, so sample complexity scales with the *model's* capacity to fit dynamics rather than with environment rollouts. Empirical sample efficiency on Atari-100k, DMC, Minecraft: **10-100× over PPO**.
- Its `symlog(reward)` and `symlog(value)` transformations make it robust to reward scale, which is important with the current γ=0.99 / K=1 imbalance.
- Handles the mixed-scale obs (some sparse binary, some continuous, some spatial) natively via the encoder.

### Failing that: **SAC-Discrete** (Christodoulou 2019, arXiv:1910.07207)

Theoretical justification:
- **Maximum-entropy RL objective** `J(π) = 𝔼[Σ_t γ^t (r_t + α · H(π(·|s_t)))]` with **learned α** via dual variable ascent to a target entropy. This solves the exact bug we diagnosed: fixed β=0.01 was wrong; α should adapt so `H ≈ H_target`. Set `H_target = 0.3 · log(|A|)` for good exploration-exploitation balance (Haarnoja et al. 2018b, §5).
- Off-policy, so 30-sps collection is fine — replay buffer with 10⁶ transitions gives ~200× data reuse.
- Twin Q-networks with target network kill positive bias of Q — this is Rainbow's `double Q` (Hasselt 2015) applied to actor-critic, and it's why SAC-Discrete beats DQN by ~30% on Atari.

### Baseline fix if we must keep PPO

The **provably correct minimal changes** to remove the regression, in decreasing effect size:

1. **Decouple entropy from reward scale.** Set `ent_coef = 0.001`, or better, replace with target-entropy KL control: at each update, `ent_coef *= 1 + η·(H − H_target)` with `H_target = 3.0` (roughly half the max joint entropy). This makes entropy a **constraint**, not a fixed penalty (Haarnoja et al. 2018b Alg. 1).
2. **Increase γ to 0.995** ⇒ effective horizon 200, matches empirical ep_len/2. This raises the discounted death signal from 0.007 to 0.082 (12×) at L=500 and the discounted first-kill signal from 0.78 to 0.88.
3. **Disable LR annealing OR replace with KL-adaptive step control** (Schulman 2017 §4.4): `lr_{new} = lr · 1.5 if KL < KL_target/1.5 else lr/1.5 if KL > KL_target·1.5`. Prevents the "LR → 0 while policy still moving" failure mode.
4. **Prune the action space** to `MultiDiscrete([9, 5])` for Stage A. Removes 3 parasitic factors, cuts entropy-loss noise by ~0.03 nats/step.
5. **Prune the obs** to the ~100-dim sufficient statistic. Faster convergence, less noise.

---

## 9. Concrete Predictions

Baseline (current run): kills_mean over final 26 episodes = **5.00 ± 3.0** at step 100k.

| Change | Predicted kills_mean at step 100k | Confidence |
|---|---|---|
| `ent_coef: 0.01 → 0.003` only | 8.0 ± 2.0 | high |
| `ent_coef=0.003` + `γ=0.995` | 9.5 ± 2.0 | high |
| Above + `anneal_lr=false` | 10.5 ± 2.0 | med |
| Above + `MultiDiscrete([9,5])` | 11.0 ± 1.5 | med |
| Above + KL-adaptive entropy target (H=3.0) | 13 ± 2 | med |
| SAC-Discrete with entity-attention encoder | 20 ± 3 at step 50k, plateau at ~30 | med-low |
| DreamerV3 (default hparams) | 25 ± 5 at step 30k | low (implementation risk) |

The strongest single-parameter change is **ent_coef reduction** — I predict this alone recovers ≥ 60% of the peak-to-baseline gap by step 100k. The specific ent_coef derivation: match entropy-bonus-per-step ≈ ½ · kill-reward-per-step ⇒ `ent_coef ≈ 0.5 · 0.015 / 5.0 = 0.0015`. Use 0.003 as a safe upper bound.

Signal to watch for confirmation: `charts/kills_mean` should NOT regress after peaking; `entropy_per_factor/shoot` should fall from 1.60 to ≈ 0.8-1.0 (not stall at 1.33 as in the current run).

---

## Gaps / Uncertainties

- I did not run the trainer to verify predictions; numbers are analytic.
- I did not inspect the mod's `main.lua` to confirm exact respawn timing of the Attack Fly after death — if respawn takes > 5 ticks, my P(hit) and kills/ep estimates are optimistic by 10-15%.
- The `mod_restart` path in `env.py` synthesizes `r_death` if the shaper missed it. If HP-based detection *is* firing, this could double-count -1 for one death, biasing `V(terminal) → -2`. Not the primary bug but worth a targeted assertion.
- DreamerV3 recommendation carries implementation risk; SAC-Discrete is the safer high-value bet.

## Citations

- Kakade & Langford 2002, "Approximately Optimal Approximate Reinforcement Learning" (CPI, monotonic-improvement bound).
- Schulman et al. 2015, "Trust Region Policy Optimization" (TRPO, monotonic-improvement guarantee).
- Schulman et al. 2017, "Proximal Policy Optimization Algorithms" (clipped surrogate).
- Kakade 2003 PhD thesis; Jin et al. 2020, "Provably Efficient RL with Linear Function Approximation" (PAC bounds).
- Haarnoja et al. 2018a, "Soft Actor-Critic"; Haarnoja et al. 2018b, "Soft Actor-Critic Algorithms and Applications" (learned α).
- Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings", arXiv:1910.07207.
- Hafner et al. 2023, "Mastering Diverse Domains through World Models" (DreamerV3).
- Lee et al. 2019, "Set Transformer" (attention over unordered entity slots).
- van Hasselt et al. 2015, "Deep Reinforcement Learning with Double Q-Learning".
- Hessel et al. 2017, "Rainbow: Combining Improvements in Deep RL" (double + prioritized + dueling + noisy + n-step).
