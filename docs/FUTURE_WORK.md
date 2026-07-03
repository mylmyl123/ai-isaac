# Future Work: Advanced RL Techniques for Isaac AI

This document captures **research directions we've decided NOT to pursue right
now** but which are ready to implement if the current setup underperforms or
if we want to push toward superhuman play. Each item includes rationale,
estimated effort, expected gain, and concrete implementation plan.

Read this in conjunction with `README.md` (current state) and the model /
ppo / reward code for what we've already built.

Techniques are grouped by **Phase** (implementation size) and referenced
by their original ID from the July 2026 planning discussion.

---

## Phase B — Standard Advanced Techniques (medium effort, high value)

Consider these if Phase A (the current implementation) plateaus or if we want
another 2-3x sample efficiency boost.

### B1. Distributional Value Function (C51 / DeepMind style)

**Why**: our reward distribution has extreme variance (-3 death to +50 mom
kill, plus dense ±0.001-1.0 shaping). A scalar value function has to compress
this into one number and averages catastrophically. A **distributional value
function** predicts an entire distribution over returns, which handles the
variance natively.

**Papers**: Bellemare 2017 (C51), Dabney 2018 (QR-DQN), Hafner 2023 (DreamerV3's
twohot categorical).

**Approach**:
1. Replace `self.value_head = nn.Linear(gru_dim, 1)` with
   `self.value_head = nn.Linear(gru_dim, n_atoms)` where `n_atoms = 51`.
2. Define support: `values = torch.linspace(v_min, v_max, n_atoms)` where
   `v_min=-20, v_max=20` (in symlog-space; corresponds to raw returns ±4.8B).
3. Compute `p = F.softmax(value_head_out, dim=-1)`.
4. Expected value: `V(s) = (p * values).sum(-1)`. Use this for advantage
   computation (compatible with existing GAE).
5. For the loss: project the TD target distribution onto the atom support
   (categorical projection, ~30 lines of code) and use KL divergence between
   predicted and projected target as the value loss (instead of MSE).

**Effort**: ~1 day of implementation + tuning.
**Expected gain**: 20-40% better sample efficiency on high-variance domains
according to Rainbow ablation.

**Files to touch**: `model.py` (head change), `ppo.py` (loss change), yaml
(`n_atoms`, `v_min`, `v_max`), tests.

---

### B2. Curriculum Learning

**Why**: Isaac's difficulty ramps aggressively. Learning combat + navigation +
item usage + progression all at once is hard. Curriculum learning starts with
simplified environments and gradually increases difficulty as the agent masters
each stage.

**Papers**: Bengio 2009 (original curriculum RL), Justesen 2018 (procgen),
Portelas 2020 (survey), Team OpenAI 2019 (Sonic curriculum).

**Approach**:
1. Modify `mods/isaac-rl-bridge/main.lua` to accept a "curriculum stage" flag
   from Python at reset time via `set_curriculum(stage)`. Stages:
   - **Stage 0**: single empty room (no enemies) — teach movement, doors.
   - **Stage 1**: single room with 1-2 weak enemies (fly, spider) — teach shooting.
   - **Stage 2**: single room with normal Basement 1 spawns — full combat.
   - **Stage 3**: Basement 1 full run — navigation across rooms.
   - **Stage 4**: Basement 1 + boss — full-floor challenge.
   - **Stage 5+**: multi-floor progression.
2. Track `success_rate` per stage in the trainer.
3. When `success_rate > 0.8` for stage N over the last M episodes, promote
   to stage N+1. If success at stage N+1 drops below 0.3, demote back.
4. Env sends curriculum stage to Isaac via handshake; Isaac spawns
   appropriate room layouts.

**Effort**: 2 days (mostly Lua mod work).
**Expected gain**: 2-5x sample efficiency documented consistently across
procgen / Sonic / Minecraft. For our compute budget, this might be the single
highest-ROI item.

**Files to touch**: `mods/isaac-rl-bridge/main.lua`, `mods/isaac-rl-bridge/obs.lua`,
new `python/isaac_rl/curriculum.py`, `ppo.py` (curriculum scheduler),
yaml (curriculum policy config).

---

### B3. Predict-Future-Rewards Auxiliary Task

**Why**: our current aux losses predict *current-state* summaries (nearest
enemy distance, enemy count, projectile distance). These force the trunk to
be *descriptive*. Predicting *future* rewards forces it to be *predictive*,
which is exactly what the value function needs.

**Papers**: UNREAL (Jaderberg 2017), reward prediction as aux task in ATARI RL.

**Approach**:
1. Add new head: `self.reward_pred_head = nn.Linear(gru_dim, N)` predicting
   the reward at the next N ticks (e.g. N=8).
2. During PPO update, targets are computed from the rollout: for each
   state s_t, target is `[r_{t+1}, r_{t+2}, ..., r_{t+N}]`.
3. Loss: MSE between predicted future rewards and actual future rewards,
   weighted by `cfg.reward_pred_coef` (default 0.1).
4. Bonus: also predict `damage_taken_flag` at the next N ticks (binary).
   Forces the network to model imminent damage — helps dodging.

**Effort**: 1 day.
**Expected gain**: better representations → faster value convergence.

**Files to touch**: `model.py` (new head), `ppo.py` (target computation +
loss), yaml.

---

### B4. Latent Variable Conditioning (AlphaStar-style)

**Why**: our policy is deterministic-conditional. It picks the same action
for the same state every episode. AlphaStar samples a *latent strategy
variable* z at episode start and conditions the policy on z. Different z
values → different play styles (aggressive, defensive, exploratory). This
gives the policy strategic diversity and helps escape local optima.

**Papers**: Vinyals 2019 (AlphaStar), Eysenbach 2019 (DIAYN),
Achiam 2018 (VIC).

**Approach**:
1. Add `z_dim: int = 16` to `PolicyConfig`.
2. At episode start, sample `z ~ N(0, I)` in the trainer, hold constant
   for the whole episode.
3. Concatenate z to every obs before the trunk MLP.
4. Optionally: train a "z_encoder" that maps from trajectory statistics
   back to z, encouraging z to actually encode strategy (DIAYN-style).

**Effort**: 1-2 days.
**Expected gain**: modest raw perf gain, big diversity gain, helps explore
strategy space. Especially valuable for boss fights where different
strategies work for different bosses.

**Files to touch**: `model.py` (z input), `env.py` (z sampling per episode),
`ppo.py` (z-conditioned rollout), yaml.

---

## Phase C — Model-Based RL (high effort, potentially transformative)

The single biggest possible sample-efficiency gain, but also the biggest
implementation risk.

### C1. Learned World Model (DreamerV3-inspired)

**Why**: DreamerV3 achieves state-of-the-art on 150+ tasks with a **single
set of hyperparameters** and 10-100x better sample efficiency than model-free
methods. For a compute-limited setting like ours, this is the endgame.

**Papers**: Hafner 2019, 2020, 2023 (Dreamer series), Ha & Schmidhuber 2018
(original world models), Kaiser 2019 (SimPLe on Atari).

**Approach** (simplified from full DreamerV3):

1. **World model architecture**:
   - Take our existing trunk encoder (produces feature vector `h_t` from obs).
   - Add a recurrent transition model: `p(z_{t+1} | z_t, a_t)` where z is a
     learned latent (e.g. 512-dim).
   - Prediction heads:
     - `p(obs_features | z)` — reconstruction (drives representation learning).
     - `p(reward | z, a)` — reward prediction.
     - `p(done | z, a)` — episode termination prediction.
     - `p(discount | z, a)` — for handling variable-length episodes.

2. **Actor-critic in latent space**:
   - Value function conditions on z, not raw obs.
   - Policy conditions on z.
   - Training loop:
     a. Collect real experience (as we do now).
     b. Update world model on real experience (supervised).
     c. **Imagination rollouts**: from every real state, roll out H=15 steps
        purely in the world model. Compute imagined returns.
     d. Update value function on imagined returns.
     e. Update policy via imagined-return gradients (differentiable
        world model enables reparameterisation).

3. **Key DreamerV3 tricks** (already partly implemented):
   - Symlog on obs and rewards ✅ (rewards done, obs would be new).
   - Twohot categorical value target (see B1).
   - Free-bits KL regularisation.
   - Percentile-normalised advantages.

**Effort**: 2-3 weeks for a working prototype, another 1-2 weeks for tuning.
Big implementation. High bug surface.

**Expected gain**: **5-20x sample efficiency**. Could master Basement 1 in
hundreds of thousands of frames instead of millions.

**Risks**:
- World model quality bounds everything downstream. Bad model → bad policy.
- Requires careful KL, reconstruction, imagination-horizon tuning.
- Bugs are hard to diagnose because errors can hide in any of 4 loss terms.

**Files to touch**: essentially a new subsystem. `python/isaac_rl/world_model.py`
(new, ~1500 lines), major changes to `ppo.py` (or new `dreamer.py`), new
replay buffer for real experience storage, new tests.

---

### C2. Transformer-Based Policy

**Why**: for very long-horizon strategy (e.g. deciding whether to buy an item
in a shop room based on health/coins/upcoming floor), GRU state degrades. A
transformer with explicit attention over recent history would preserve more
context.

**Papers**: Parisotto 2020 (Stabilising Transformers for RL), Chen 2021
(Decision Transformer).

**Approach**:
1. Replace GRU with a small transformer (4-6 layers, 256-dim, 4 heads).
2. Input: sequence of last 32-64 trunk features.
3. Output: attention-weighted representation for the current step.
4. Use rotary positional encoding for stability.

**Effort**: 1-2 weeks (mostly re-tuning; transformers are finicky for RL).
**Expected gain**: modest (10-20%) improvement for long-horizon strategy.
GRU is usually sufficient for our task horizons.

**Not recommended unless** we hit a specific long-horizon bottleneck that
GRU visibly fails on.

**Files to touch**: `model.py` (major surgery), potentially `ppo.py`
(BPTT changes if we switch to attention over rollouts).

---

## Phase D — Advanced Exploration (only if we hit exploration bottleneck)

The below are exploration-heavy techniques for hard-exploration games.
Isaac isn't strictly a hard-exploration game (dense rewards + linear
progression), so these are lower priority.

### D1. Never-Give-Up (NGU) Episodic Novelty

**Why**: our existing RND provides *long-term* novelty (visited states
across all episodes). NGU adds *episodic* novelty (visited states in the
current episode). Prevents the bot from re-visiting the same room 10 times
in one run.

**Papers**: Badia 2020 (NGU), Badia 2020 (Agent57).

**Approach**:
1. Train an inverse dynamics net: `p(a | s_t, s_{t+1})`. This learns
   controllable-state embeddings (ignoring uncontrollable aspects like
   enemy movements).
2. At each step, look up the current state embedding in an episodic memory
   of visited states (k-nearest neighbours).
3. Episodic reward: `1 / sqrt(1 + n_neighbours_at_distance_d)`.
4. Combine with RND long-term novelty for full NGU signal.

**Effort**: 2 days.
**Expected gain**: only helps if we observe visible re-visitation loops.

---

### D2. Population-Based Training (PBT)

**Why**: train N policies in parallel with different hyperparameters
(exploration, LR, entropy coef, etc.). Periodically kill worst, clone-and-
mutate best. Effectively automates hyperparameter search + gives population
diversity as a side effect.

**Papers**: Jaderberg 2017 (PBT), Vinyals 2019 (AlphaStar's league).

**Approach**:
1. Run 4-8 PPO instances in parallel with different hyperparameters.
2. Every K episodes, evaluate each policy on a held-out task.
3. Bottom 25% get replaced by clones of top 25% with mutated hyperparameters.
4. Simpler variant: single training run but keep 4 policies as a "team"
   and periodically distill the best into the others.

**Effort**: 3-4 days.
**Expected gain**: better hyperparameters found automatically; ~20-30%
improvement. Requires 4x compute (but wall-clock same if you have 4x envs).

---

### D3. Bootstrapped Q-Ensembles / Uncertainty-Driven Exploration

**Why**: multiple value/policy heads → Bayesian uncertainty over actions.
Actions with high posterior variance are explored more aggressively.

**Papers**: Osband 2016 (Bootstrapped DQN), Chen 2017 (UCB Q-learning).

**Approach**:
1. Train N=5 policy/value heads on bootstrapped batches (each head sees
   only a random subset of the rollout).
2. At action-selection time, sample one head per step (encourages
   commitment to a strategy for a whole episode).
3. Alternatively: use posterior variance as intrinsic reward.

**Effort**: 2 days.
**Expected gain**: modest, mostly helps sparse-reward tasks. Isaac is
dense-reward so this is low priority.

---

## Item cross-reference table

| Phase | ID | Item | Effort | Expected gain | Recommended? |
|-------|-----|-----|--------|---------------|--------------|
| A | A1 | Frame stacking (player history) | 30min | Small (dodging) | ✅ Done |
| A | A2 | Larger network | 1h | Small-medium | ✅ Done |
| A | A3 | n_envs scale-up | config | Wall-clock speedup | ✅ Documented |
| A | A4 | Extended BC + demos | config | Better BC baseline | ✅ Documented |
| A | A5 | LR warmup | 20min | Prevents early instability | ✅ Done |
| A | A6 | Weight decay (AdamW) | 5min | Mild regularisation | ✅ Done |
| **B** | B1 | Distributional value (C51) | 1 day | Medium (variance) | If B1 needed |
| **B** | B2 | Curriculum learning | 2 days | 2-5x sample eff | High-ROI |
| **B** | B3 | Predict-future-rewards aux | 1 day | Small-medium | Optional |
| **B** | B4 | Latent variable (AlphaStar) | 1-2 days | Strategy diversity | Optional |
| **C** | C1 | Learned world model | 2-3 weeks | **5-20x sample eff** | Transformative |
| **C** | C2 | Transformer policy | 1-2 weeks | Small | Not recommended |
| **D** | D1 | NGU episodic novelty | 2 days | Small (dense rewards) | Low priority |
| **D** | D2 | Population-based training | 3-4 days | Medium (auto HP tuning) | Optional |
| **D** | D3 | Ensemble uncertainty | 2 days | Small (dense rewards) | Low priority |

---

## Which items would we implement NEXT if we continued?

If we picked up this work again, the order I'd recommend:

1. **B2 (Curriculum)** — highest ROI, well-documented gains, moderate effort.
2. **B1 (Distributional value)** — if value loss stays high after B2.
3. **C1 (World model)** — if we have 3-4 weeks and want transformative
   sample efficiency.

**Skip unless specific symptom hits**: B3 (unless representations feel weak),
B4 (unless we need strategic diversity), C2 (unless long-horizon reasoning
fails), D1-D3 (unless exploration is stuck).

---

## References

- Vinyals et al. 2019 — "Grandmaster level in StarCraft II" (AlphaStar).
- Berner et al. 2019 — "Dota 2 with Large Scale Deep Reinforcement Learning" (OpenAI Five).
- Hafner et al. 2023 — "Mastering Diverse Domains through World Models" (DreamerV3).
- Bellemare et al. 2017 — "A Distributional Perspective on Reinforcement Learning" (C51).
- Andrychowicz et al. 2020 — "What Matters in On-Policy Reinforcement Learning".
- Engstrom et al. 2020 — "Implementation Matters in Deep RL: PPO and TRPO".
- Schmitt et al. 2018 — "Kickstarting Deep Reinforcement Learning" (already implemented).
- Jaderberg et al. 2017 — "Reinforcement Learning with Unsupervised Auxiliary Tasks" (UNREAL).
- Badia et al. 2020 — "Never Give Up" / "Agent57".

Last updated: 2026-07-02 (post Phase-A implementation).
