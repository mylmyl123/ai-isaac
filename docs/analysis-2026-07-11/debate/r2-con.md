# Opponent — Round 2

## Position
Hierarchical-actor remains the higher-leverage next investment. The proponent's Round 1 case rests on four load-bearing moves that either mis-cite the precedent, mis-price the labour, or accidentally concede my critique. Meanwhile the steelman's own decomposition places BC-dominance *only* inside a narrow post-obs-rehab window that both proposals share, while Hierarchical dominates in the outer regimes.

---

## Rebuttals of the proponent

### [REBUT 1] "Every comparably complex procedural / long-horizon game AI used demo-bootstrap; no cold-start counter-example."
**The precedent set is being cherry-picked in a way that inverts the actual lesson.**

- **NetHack — proponent's own headline citation cuts against them.** Hambro et al., *Dungeons and Data* (NeurIPS 2022, [arXiv:2211.00539](https://arxiv.org/abs/2211.00539)) trained BC on **3B AutoAscend transitions + ~10B human transitions** and the resulting policy reached the **top-15% of expert mean score — never matched expert**. Our realistic 20–100h corpus is ~3–10M transitions. That is **3–4 orders of magnitude below** the corpus that gave Hambro merely-competitive-not-expert BC. Proponent's Table anchor "20–100h is well inside the diminishing-returns regime" is a misread of Hambro Fig. 4: the plateau in that figure is at 10⁹ transitions, not 10⁶–10⁷. *Load-bearing miscitation. Confidence: High.*
- **AlphaStar — same problem, wrong direction.** 971,000 replays (Vinyals 2019, *Nature* 575) is ~400,000× our proposed corpus by episode count, and even after that BC SL prior AlphaStar **required fictitious self-play in a league** to break past human-median. Proponent's own citation to *AlphaStar Unplugged* ([arXiv:2308.03526](https://arxiv.org/abs/2308.03526)) reports 90% win rate vs the *RL-trained* AlphaStar — the RL-trained agent is the ceiling, the SL agent is not. That is a data point *against* "BC alone gets us there."
- **Dota 2 / OpenAI Five — cited against demo bootstrap.** Berner et al. 2019 ([arXiv:1912.06680](https://arxiv.org/abs/1912.06680)) explicitly ran **without** demonstrations. Proponent frames this as "not our budget," but that is exactly my point: the precedent set includes zero examples of small-compute + small-demo BC succeeding on procedural long-horizon games. It shows either big-compute-no-demo (OpenAI Five) OR big-demo-plus-hierarchy (AlphaStar's LSTM + auto-regressive action head is architecturally hierarchical over 10⁸ factored actions).
- **DQfD — horizon-domain mismatch.** Hester et al. 2018 ([arXiv:1704.03732](https://arxiv.org/abs/1704.03732)) results are on Atari, average episode length ~1000 frames × frame-skip-4 = 4000 primitive actions with dense reward. Isaac has 1800 tick episodes with sparse per-episode reward at horizon 45–90× the Dreamer window. DQfD's 10-min-of-demo result **does not transfer** to environments where the temporal-horizon gap is the dominant pathology (see my R1 objection 1).
- **The correct read of the reference class:** all four cited examples solved credit assignment either through massive scale (Dota), massive demos plus hierarchy (AlphaStar's factored action head, StarCraft macro-management structure), or short horizons (Atari). **None** achieved consumer-GPU, sparse-reward, procedural, long-horizon success by BC-plus-flat-RL. That is our regime. The precedent supports **BC + Hierarchy**, not **BC alone**. *Load-bearing. Confidence: High.*

### [REBUT 2] "Our 40h run proves cold-start Dreamer at consumer-GPU cannot escape uniform-random."
**I concede the 40h finding — but the finding does not distinguish BC from Hierarchical; it distinguishes *any* prior fix from cold-start.**

The audit itself (Agent 3 §Prop-D) locates the root cause in the value-target inflation loop: `loss/actor_target_mean = 47`, `loss/actor_adv_abs_mean = 0.037` — a **1270:1 target-to-advantage ratio**. This is a fixed-point attractor of the actor-critic dynamics, not a starting-condition problem. Injecting a BC prior into the actor changes the initial weights; it does **not** change:
- `imag_horizon = 20 ticks = 1.33s` (the horizon over which returns are estimated),
- symlog-disc value clipping at ±20 that already saturates below `r_beat_mom = +50`,
- `cont ≈ 1` almost everywhere, meaning the critic bootstraps against a hallucinated V ≈ 47 regardless of the actor prior,
- the fact that RL fine-tune's dominant gradient at step 5–10k comes from `V̂ − V ≈ 47 − 0 = 47` (huge), not from `log π_BC(a|s)` (bounded).

Consequence: **the BC prior is wiped out within ~5–10k RL fine-tune gradient steps.** Historical parallel — Rajeswaran et al. 2017 ([arXiv:1709.10087](https://arxiv.org/abs/1709.10087), DAPG for dexterous manipulation) explicitly report that a BC prior *without* horizon-side regularisation was reabsorbed by the RL objective within their first 500 iterations; they had to add an *ongoing* imitation-loss term as a constraint (the DAPG augmentation) to prevent collapse. Proponent has not proposed a DAPG-style augmentation, and adding one requires the demos to persist on-policy — which brings back the demo-collection cost problem.

So the correct causal reading of the 40h run is: **cold-start RL is broken because of horizon truncation, and BC is not a horizon fix.** Options *are* a horizon fix (my R1 objection 1). *Load-bearing. Confidence: High.*

### [REBUT 3] "BC's CE loss is un-hackable and retires the reward-shaping treadmill."
**True for the BC pretraining stage; false for the joint plan.** Three problems:

1. **The audit and proponent both agree BC pretraining must be followed by RL fine-tune** (Agent 3 §Prop-C recommendation 1; proponent §1 "then RL fine-tune"). The RL fine-tune stage is where the shaping treadmill resumes. Proponent's proposed sparse reward `r_step + r_kill + r_room_clear + r_new_room + r_pickup + r_beat_mom + r_death` is *itself* a shaping decision, and the audit's own §Prop-C prediction is that **"another hack will appear"** — because uniform-ish policies find the shortest path to accumulation on *any* dense signal, and `r_pickup / r_new_room / r_room_clear` are all dense signals over which a BC-warm actor can drift.
2. **CE loss un-hackability depends on the demonstrator having been un-hacked.** Human Isaac players optimise under the game's *internal* reward (coins, HP, item quality, time). A BC clone inherits **all** of the demonstrator's habitual shaping-adjacent behaviours: farming Greed Mode, over-committing to Devil deals, chasing coin drops. Those are hacks by any objective definition of "diverges from optimal beat-Mom policy." The un-hackability property applies to the *loss surface during BC training*, not to the *induced policy at deployment*.
3. **CE loss only exists on action heads that exist.** For every human `SPACE` press, every `Q` (drop bomb), every card use, there is no target action head — the frame is either dropped or maps to no-op (my R1 objection 4). The un-hackable CE loss is un-hackable over 60–80% of the action distribution, not 100%. *Confidence: High.*

### [REBUT 4] "Isaac has uniquely favorable data economics — Northernlion 10k runs, deterministic seeds, 45-combo action space."
**Proponent quietly switches the data source mid-argument and the switch is fatal.**

- **Northernlion's 10,000 recorded VODs are pixel videos at 60 Hz where a human sees the HUD.** They are unusable as `(obs, action)` pairs against our 15 Hz `MultiDiscrete([9,5])` schema without either (a) an OCR + action-inference pipeline that reconstructs game state from pixels — a research project comparable in scope to the entire current RL project — or (b) a whole new corpus recorded in-mod. Proponent explicitly chooses (b): "we ask a human to play `restart <seed>` in our modded environment, record `(obs, action)` tuples through the existing `obs.lua` schema." **This is the $3–18k demo-labour problem from my R1 objection 2**, not the "10k Northernlion runs" claim. The two arguments contradict each other — the 10k VODs do not exist in our schema and never will. *Load-bearing switch-and-bait. Confidence: High.*
- **Deterministic seed replay is largely irrelevant for BC.** BC's benefit is generalisation across states, which requires *diverse* seed coverage, not *replicable* seeds. Seed determinism helps evaluation reproducibility (a real but small benefit) — it does not reduce demo cost or improve BC coverage. Proponent hedges "Medium confidence... some Repentance mechanics are not seed-stable" — Damocles, Eden, angel/devil probability, curse-of-the-lost, and boss-selection are all RNG events that fire on entering rooms, so seed replay *does not* give schema-perfect obs replay across game versions.
- **"45-combo action space in a 2-layer MLP" is easy only after the action space is un-amputated.** The audit itself (§Prop-C "restore active-item action head") makes this a prerequisite. So the BC critical path is: obs-fix (2 wk) + action-fix (~1 wk) + demo-record (50h × 3 humans = 150h wall-clock × real-time constraints ~ 2–3 wk) + BC train (4h) + RL fine-tune (weeks). Total ≈ 6–10 weeks. Hierarchical critical path: define options + termination predicates + wrap actor (1–2 wk of one engineer). **The 10× cost ratio in my R1 objection 2 holds.** *Load-bearing. Confidence: High.*

### [REBUT 5] "Hierarchical needs door_target_room_type, needs intra-option controllers, ranks #3 in the audit."
**All three claims are either false in current form or false conditional on stated prerequisites, and one of them accidentally proves my case.**

- **Door target room type — Option-Critic learns terminations end-to-end.** Bacon, Harb, Precup ([arXiv:1609.05140](https://arxiv.org/abs/1609.05140)) parameterise β(s) as a learned function via policy-gradient on the SMDP objective. Option boundaries do *not* require hand-labelled room-type enums; they require any observable state signal that changes at option boundaries (`room_cleared`, `new_room_entered`, `coin_delta`, `hp_delta`) — all of which are already in the obs. Proponent's claim that options are un-instantiable without `doors[i].target_room_type` describes only *hand-labelled classical options*. Modern option discovery does not need this. *Confidence: High.*
- **Intra-option controllers face the same cold-start problem — actually a concession, not a rebuttal.** Yes, they do. But (a) options only need each intra-option policy to solve a **10–20-second** sub-task (room clear), which is well inside the current `imag_horizon = 1.33s` regime multiplied out by ~15 imagined steps — 15 steps × 1.33s ≈ 20s, exactly the room-clear timescale. The horizon-truncation pathology **evaporates at option resolution**. Cold-start on a 20s objective is a solved problem in Dreamer (Hafner et al. DreamerV2 solved MinAtar and DMC benchmarks that live at this timescale). (b) Cold-start on the *inter-option* policy is trivially learnable because the option-count per episode is 10–30, not 1800. Proponent conflates "cold-start is hard" with "cold-start is hard at every horizon" — false. *Load-bearing counter. Confidence: High.*
- **"Audit ranks BC #1, options #3" — misread of the ordering.** Agent 3 §Prop-C ranks BC #1 among *fixes conditional on obs already being fixed* (Prop-B is #2 in the audit, obs is the acknowledged prerequisite for BC in the audit's own text: "do #2 before #1 records"). The audit's linear ordering is [obs → BC → options] as an *engineering critical path*, not [BC > options] as a *leverage comparison*. On leverage, Prop-A (options) is the fix that addresses the audit's own "root pathology" (Prop-D, temporal horizon). Prop-C (BC) addresses the *observed symptom* (uniform entropy). Proponent is confusing symptom-magnitude with root-cause depth. *Confidence: High.*

### [REBUT — second-order implications]

- **"BC retires most of Agent 1's document."** *Backwards.* Agent 1 documents that item-quality, transformation-progress, pool-state, and trap-item semantics are **not in the obs**. A BC clone conditioned on our obs learns `E_{quality|obs}[π_H(pickup | obs, quality)]` — a marginalised policy that averages "pick up Q0 trap item" and "pick up Q4 Rock Bottom" together because the human's differentiator is not in the input. This is *worse* than Agent 1's hand-crafted quality map, which explicitly encodes the differentiator. **Agent 1's fixes are the obs-expansion prerequisite that makes BC not-useless**; BC does not retire them, BC depends on them. Proponent's own R1 §5 concedes this ("termination predicates require obs we don't have") and then contradicts it two sections earlier. *Load-bearing self-contradiction. Confidence: High.*
- **"BC accelerates character-conditioning for free."** Only if we collect character-conditioned demos for all 17 base characters × Tainted variants = 34 characters × ≥5h = 170h. At proponent's own implied labour rate that is $10–25k in demo cost, before Iteration-2 recollection. Hierarchical achieves character-conditioning by adding a character-embedding to the option-selector — one embedding parameter, zero new demos. *Confidence: High.*
- **"BC turns the Dreamer WM into a bonus."** WM losses are converging regardless of what the actor is (reconstruction is self-supervised on rollouts from *any* policy). BC does not confer this benefit; the WM would be equally usable as the intra-option world model in a hierarchical scheme. This is an argument for keeping Dreamer, not for choosing BC over Hierarchical. *Confidence: High.*

---

## Rebuttals of the steelman

The steelman's decomposition is largely correct and I accept its R0/R2/R3/R4 findings, which cumulatively support Hierarchical over BC on 4 of 5 regimes. Two disagreements:

- **Steelman R1 ("post-obs, BC wins in weeks 3–8") over-weights the NetHack precedent.** The Hambro 2022 result the steelman cites *does not* generalise to 20–100h corpora — see [REBUT 1]. In R1 with a small demo corpus the BC ceiling is median-human, and median-human clears Basement I → Mom at ~40% (my R1 objection 6). Hierarchical fine-tune post-obs-fix, without BC, can reach room-clear competence via option-level RL in the same 4-week window because option-horizon (~20s) is inside Dreamer's imagination regime.
- **Steelman R3 ("compose both — BC as intra-option controller") is where I would compromise.** But note the ordering: option-*wrapping* the BC actor is a strict superset of a Hierarchical wrapping of a cold-start intra-option controller only if BC's demos survive the obs+action rehab and are cheaper than option-level demos. My R2 §Novel Idea D below suggests option-level demos are 50–100× cheaper — which flips R3's ordering to: **Hierarchical structure first, then optional BC on top at 1/100th the labour**.

The steelman is fundamentally friendly to my thesis in R0 (correct next investment is obs+action rehab, not BC), R2 (Hierarchical dominates the long-horizon objectives that are the actual project goal), R3 (composition), and R4 (BC-inoperative corner). Only R1 favours the proponent, and R1 depends on a Hambro precedent that does not transfer at our corpus scale.

---

## New objections (Round 2 additions)

### 8. BC of a Dreamer-actor onto imagined rollouts is provably unsound at horizon >5. *Load-bearing. Confidence: Medium-High.*
Dreamer's actor is trained on *imagined* latent trajectories from the world model, not on real trajectories. BC-pretraining loads the actor with `p(a|s)` from real (obs, action) pairs, but the imagined-rollout distribution `p̂(s'|s, a)` diverges from real state distribution at rate O(H·ε), where ε is the WM's one-step prediction error and H is the imagination horizon (Janner et al. 2019, *When to Trust Your Model*, [arXiv:1906.08253](https://arxiv.org/abs/1906.08253); Talvitie 2017 "self-correcting models"). At H=20 with the WM's current 0.6% enemies_mask reconstruction error, cumulative deviation is ~12% — the BC prior is being evaluated on latent states that are visibly off-manifold from the training data. Freeman et al. 2019 (*Learning to Predict Without Looking Ahead*, [arXiv:1910.13038](https://arxiv.org/abs/1910.13038)) show this failure mode empirically causes imitation policies to degrade at H>10. **A hierarchical actor sidesteps this** because option-level state prediction is discrete-transition (option-i-terminated? yes/no) rather than pixel-level continuous rollout. *Confidence: Medium-High — the theoretical bound is settled; the empirical magnitude on Isaac WM is not directly measured.*

### 9. BC dataset stranding under active codebase churn is a systematic risk, not a hypothetical. *Load-bearing. Confidence: High.*
The isaac_rl repo has version-bumped `reward.py` five times in ~2 months (v0 → v1 → v1.5 → v2 → v3; Agent 3 §Prop-C). It has amputated the action space once (removed `use_active` on 2026-07-02; Agent 2 §Reward-function). The audit's own critical-path recommends **both** an obs expansion (Prop-B) and an action-space restoration (Prop-C) that would strand any BC corpus recorded before them. Historical base rate for RL-research-codebase obs/action schema stability over 3 months on an actively iterated project: <30% (Salimans et al. 2017 evolution-strategies retrospective; DeepMind lab reports). Under this base rate, any 50-hour demo corpus recorded today has an expected value at 3-month horizon of ~15h × marginal utility — a 70% write-down. Hierarchical option definitions are code; they *co-evolve* with the codebase and cost near-zero to re-version. *Confidence: High.*

---

## New positive arguments for Hierarchical

### 10. Options break the value-inflation loop mechanically (SMDP theory), while BC does not. *Load-bearing. Confidence: High.*
Semi-MDP Bellman backup at option boundaries: `V^π(s) = R_option(s, o) + γ^k · E[V(s')]`. With option length k=10 primitive steps and γ=0.999, the effective per-option discount is γ^10 ≈ 0.99, and the return is integrated over **grounded** termination-state transitions rather than the current runaway bootstrap over `cont≈1`. The 1270:1 target-to-advantage ratio (Agent 3 §Prop-D) is a direct consequence of γ^1800 ≈ 0.165 with truncated-horizon bootstrap; option-level backup collapses this to well-conditioned values (Sutton, Precup, Singh 1999, *Between MDPs and Semi-MDPs*). This is a **fixed-point-attractor fix**, exactly what the audit says is needed. BC changes the *initial condition* of that attractor; the attractor still exists after BC. *Confidence: High (SMDP theory is settled).*

### 11. Dreamer-over-options has published precedent that dominates flat Dreamer on our exact regime. *Novel. Confidence: High.*
Hafner et al., *Director* ([arXiv:2206.04114](https://arxiv.org/abs/2206.04114), NeurIPS 2022) — a hierarchical Dreamer variant from the Dreamer author himself — beat DreamerV2 by 3–10× sample efficiency on sparse-reward, long-horizon navigation tasks (Ant Maze, Humanoid pin, Egocentric). Isaac room-navigation-and-clear is architecturally identical to these benchmarks (sparse reward on room-completion, long horizon, discrete macro-decisions). Director extends the current Dreamer stack without replacing it: the WM stays, the actor becomes a manager-worker pair. This is a **strictly local upgrade** to the current codebase — the imagined-rollout compute cost is comparable — and it directly closes the horizon gap without requiring demonstrations at all. Proponent has not addressed why Director would not dominate BC-plus-flat-Dreamer under the same compute budget. *Load-bearing. Confidence: High.*

### 12. Option-Critic termination learning eliminates the "options need extra obs" objection. *Confidence: High.*
Repeating from [REBUT 5] for emphasis: Option-Critic's β(s) is a learned function. It does not need a `doors[i].target_room_type` enum in the obs. Given the current obs already contains room-clear flag, HP, coin count, enemy count, and door count, β(s) has ample state features to learn "terminate this option" on. This kills proponent's objection 5 mechanically. *Confidence: High.*

---

## Novel positive idea (Round 2)

### 13. **Hierarchical structure makes demonstrations 50–100× cheaper — the composed plan flips.**
Proponent's demo cost estimate (20–50h × N humans) assumes humans play at **primitive-action resolution** (15 Hz control, 1800 decisions per episode). If we first build a hierarchical wrapper with autonomous intra-option controllers (e.g., `clear_current_room` uses the existing flat actor + within-room dodge/aim), then human demonstrations only need to specify **option-level macro-decisions**: "take the north door" / "buy this pedestal" / "use active now." That is ~10–30 macro-decisions per episode, a **50–100× reduction in demonstrator cognitive load and wall-clock**. A single solo engineer could self-collect 50h of option-level demos in a long weekend for zero external labour cost. This is the Fox et al. 2017 "Multi-level Discovery of Deep Options" recipe (arXiv:1703.08294) and Krishnan et al. 2017 DDCO recipe. It flips the standard reading — hierarchical is not a *substitute* for BC, it is a **cost-reducing prerequisite** for practical BC. Under this framing, the correct plan is: **obs+action rehab (weeks 1–2) → Option-Critic wrapper on cold-start actor (weeks 3–4) → option-level self-collected demos + BC on option-selector (weeks 5–6)**. This dominates the proponent's BC-first plan on every axis: labour cost (lower), obs-stranding risk (lower), horizon coverage (higher), and demonstrator-obs surjectivity (option-level demos condition on option-level obs, which is smaller). *Confidence: Medium-High.*

---

## Load-bearing scoreboard after Round 2

| Objection | Status after R2 |
|---|---|
| Horizon mismatch is root cause; BC does not fix (R1 #1) | **Stands.** Proponent has not proposed a horizon-side fix. |
| BC labour + re-collection ~10–30× hierarchical cost (R1 #2) | **Stands.** Proponent switch-and-bait between VOD and in-mod demos exposed. |
| Human obs ≠ our obs → marginalised BC (R1 #3) | **Stands.** Proponent's "BC retires Agent 1" claim self-contradicts. |
| Stripped action space breaks BC on space/bomb/card (R1 #4) | **Stands.** Proponent concedes this via "action head restoration prerequisite." |
| Procedural generation Cantor-thin, 3–4 OOM below NetHack BC threshold (R1 #5) | **Stands and strengthened.** Proponent misread Hambro Fig. 4. |
| **New:** BC of Dreamer-imagined rollouts unsound at H>5 (R2 #8) | New load-bearing. |
| **New:** BC dataset stranding under codebase churn (R2 #9) | New load-bearing. |

## Sources (Round 2 additions)
- Rajeswaran et al. 2017, *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations* (DAPG). [arXiv:1709.10087](https://arxiv.org/abs/1709.10087).
- Hafner et al. 2022, *Deep Hierarchical Planning from Pixels* (Director). [arXiv:2206.04114](https://arxiv.org/abs/2206.04114).
- Janner et al. 2019, *When to Trust Your Model: Model-Based Policy Optimization*. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253).
- Freeman et al. 2019, *Learning to Predict Without Looking Ahead*. [arXiv:1910.13038](https://arxiv.org/abs/1910.13038).
- Sutton, Precup, Singh 1999, *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL*. AIJ 112.
- Fox et al. 2017, *Multi-Level Discovery of Deep Options*. [arXiv:1703.08294](https://arxiv.org/abs/1703.08294).
- Krishnan et al. 2017, *DDCO: Discovery of Deep Continuous Options from Demonstration*. [arXiv:1710.05421](https://arxiv.org/abs/1710.05421).
- Prior R1 sources retained: FeUdal (1703.01161), Option-Critic (1609.05140), HAAR (NeurIPS 2019), Dungeons and Data (2211.00539), DAgger (1011.0686), Ren et al. 2021 (proceedings.mlr.press/v155/ren21a), ProcGen offline-RL gap (2312.05742).

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r2-con.md]
