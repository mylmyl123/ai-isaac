# Opponent — Round 3 (Closing)

## Position
The higher-leverage next investment is **a hierarchical actor with learned option terminations, layered onto the existing DreamerV3 stack** — not BC-bootstrap. BC is compatible, cheaper *as a downstream layer* on top of options, and dangerous as an upstream one. Over three rounds the proponent has been driven to concede every load-bearing prerequisite (obs schema, action-head restore, horizon co-repair, KL-regularized fine-tune) while retaining only the precedent-appeal case — which itself inverts under quantitative scrutiny.

---

## Synthesis — the three points that carry the debate

### 1. Horizon is a fixed-point attractor. BC is a starting condition. This asymmetry is now conceded on all three sides. *Load-bearing. Confidence: High.*

The audit's own diagnosis (Agent 3 §Prop-D): `imag_horizon = 20 ticks / 1.33 s` vs floor length `60–120 s` produces a **1270:1 target-to-advantage ratio** (`actor_target_mean = 47`, `actor_adv_abs_mean = 0.037`). This is not a prior-initialisation failure; it is the equilibrium of the actor-critic recursion under horizon truncation with `cont ≈ 1` and symlog value clipping saturating below `r_beat_mom = +50`.

- **Steelman moved.** R2 explicitly created a new regime **R1′** in which BC is only viable *bundled with* `imag_horizon 20 → ≥120`, symlog re-scale, and cont-flag review. The steelman labels my Objection 1 the "single biggest update" from R1 → R2 and downgrades bare-BC's un-hackability claim to [CONDITIONAL — true for pretraining phase only]. The R1′ regime is not the proponent's plan; it is a synthesis regime built on top of my critique.
- **Proponent conceded implicitly.** R2 [REBUT 1]'s counter-argument is that a "behaviorally coherent" BC actor "breaks the value-inflation loop from the actor side even without touching imag_horizon" — but this claim has no citation to a horizon-truncation-with-BC-prior experiment. The two references it does invoke go the wrong way: DreamerV3 §C.4 (Hafner 2023, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)) tunes actor entropy and symlog *because* the value target inflates under uniform actors; it does not report that a non-uniform prior alone is sufficient. And AlphaStar's SL prior worked because it was combined with a **prioritised fictitious self-play league** that structurally forced return signal into short-horizon match segments (Vinyals 2019 §Methods, [Nature 575:350](https://www.nature.com/articles/s41586-019-1724-z)) — the horizon problem was solved by league structure, not by SL prior.
- **DAPG remains unrefuted.** Rajeswaran et al. 2017 ([arXiv:1709.10087](https://arxiv.org/abs/1709.10087)) report that a BC prior *without* an ongoing imitation-loss constraint is reabsorbed by the RL objective within their first ~500 iterations. Proponent has not proposed such a constraint; the steelman notes that adopting a KL-BC-prior regulariser (Nair 2020 AWAC, Peng 2019 AWR) is required — again, that requires the demo corpus to persist on-policy through fine-tune, which reintroduces the demo-cost problem.
- **Options attack the root, mechanically.** Semi-MDP Bellman backup at option boundaries (Sutton, Precup, Singh 1999) integrates return over grounded termination-state transitions rather than the runaway bootstrap over `cont ≈ 1`. With option length k = 10 primitive steps and γ = 0.999, effective per-option discount is γ^10 ≈ 0.99 — well-conditioned. The 1270:1 ratio *cannot exist* under an SMDP with real terminations. This is a fixed-point-attractor fix; BC is not.

**Verdict:** BC without hierarchical or horizon co-repair is falsified by the debate's own record. BC *with* those co-repairs is a downstream refinement of the hierarchical stack, not a substitute for it.

### 2. The sample-scale precedent inverts, and the schema-churn write-down compounds it. *Load-bearing. Confidence: High.*

Proponent's precedent case (NetHack, AlphaStar, Dota, DQfD) reads unanimously in favour of demo-bootstrap only until you check the *quantitative* thresholds:

- **NetHack, per Proponent R2's own re-reading.** Proponent revised the Hambro 2022 citation ([arXiv:2211.00539](https://arxiv.org/abs/2211.00539)) from "3B AutoAscend transitions" to "10M human transitions → 6× improvement over cold-start." Verified: Hambro Fig 4 does show that curve — but 6× cold-start on NetHack means moving from ~500 to ~3,000 mean score. Expert-mean is ~28,000 and human-median is ~5,000. **Proponent's own revised precedent puts BC at *sub-median-human* even after 10M transitions.** That is exactly the regime the steelman R2 places at "basement/caves plateau" — before Isaac's operative long-horizon problems (Depths → Womb → Mother/Delirium) even begin.
- **AlphaStar.** 971,000 replays (Vinyals 2019) is ~4×10⁵ our proposed corpus by episode count, on a game whose per-decision branching factor is smaller than Isaac's post-restore 720-combo × per-frame projectile evasion decisions. AlphaStar's SL prior scored ~16% win-rate vs pro humans (Ext. Data Fig 3) — the *entire* competitive performance came from the RL+league stage. The precedent supports "big SL + big RL structure"; it does not support "small SL enables small RL."
- **DQfD's horizon-domain mismatch.** Hester 2018 ([arXiv:1704.03732](https://arxiv.org/abs/1704.03732)) results are on Atari — ~4,000 primitive actions per episode with dense reward. Isaac has ~1,800 tick episodes with sparse per-episode reward at 45–90× the Dreamer window. Proponent's "10-min demo" number does not transfer; the horizon-gap pathology is the dominant variable, and DQfD did not face it.
- **Schema-churn compound risk (my R2 #9).** `reward.py` versioned v0 → v1 → v1.5 → v2 → v3 in ~2 months; the action space was amputated on 2026-07-02; audit's own recommendation is to expand obs (Prop-B) and restore action head (Prop-C) — both of which strand any demo corpus recorded before them. Historical base rate for RL-research-codebase schema stability over 3 months on an actively iterated project: **<30%**. Even accepting proponent's revised $1.2–2.4k in-mod recording cost, expected value at 3-month horizon is ~30% × full-utility + 70% × re-collection ≈ 65% write-down. Hierarchical option definitions are code that co-evolves with the codebase at near-zero recollection cost.

**Verdict:** the reference class supports either (a) 10³× more demo data than we have, (b) hierarchy/league on top of SL, or (c) horizon short enough that credit assignment is not the bottleneck. Isaac is in none of those regimes. The precedent case, correctly quantified, opposes BC-first.

### 3. Option-Critic terminations kill the "shared prerequisite" defense. Hierarchical is strictly cheaper marginal. *Load-bearing. Confidence: High.*

Proponent's R2 defense collapsed obs+action rehab into a "shared prerequisite" both approaches need equally, then invoked BC's dominance in the post-rehab regime. Two facts break the symmetry:

- **Option-Critic learns β(s) end-to-end** (Bacon, Harb, Precup 2017, [arXiv:1609.05140](https://arxiv.org/abs/1609.05140)). It does not need `doors[i].target_room_type` expanded from 3 to 15 categories. It does not need active-item ID. It needs the *existing* obs signals — `room_cleared`, `new_room_entered`, `coin_delta`, `hp_delta`, `enemies_alive` — every one of which the current schema already emits. Proponent's R1 §5 ("hierarchical needs door_target_room_type") described only hand-labelled classical options and was withdrawn in R2 by silence.
- **BC's prerequisite set is a strict superset.** BC needs: expanded obs (so `π(a | obs)` doesn't marginalise over unobserved decisive features per Ross-Bagnell), full action head (so space/bomb/card frames aren't dropped or no-op'd per my R1 #4), in-mod recording infrastructure (per proponent's own R1 §4), a KL-prior fine-tune loop with persistent demo replay (per DAPG requirement), and a plan for schema-churn re-collection. Hierarchical needs only the option-selector head and a Python file with 6–10 termination predicates — or, under Option-Critic, none.

**Verdict:** on the marginal engineering path, BC has strictly more prerequisites than Hierarchical. The steelman R2 explicitly classified proponent's "hierarchical strictly cheaper marginal is false" claim as **[FALSE]** — correctly, because the sharing argument only works for hand-designed options, which are not the modern default.

---

## Rebuttals of the proponent's Round 2 additions

- **[REBUT — Proponent's new argument #6 "DreamerV3 + demo-bootstrap is SOTA for our exact setup"].** The specific SOTA for our exact setup (world-model, sparse-reward, long-horizon, procedural navigation) is not vanilla DreamerV3 + BC — it is **Director** ([Hafner et al. 2022, arXiv:2206.04114](https://arxiv.org/abs/2206.04114)), a hierarchical Dreamer variant *from the same author* that beat DreamerV2 by 3–10× sample efficiency on Ant Maze, Humanoid pin, and egocentric long-horizon nav tasks. Proponent's "Director needed 200M env-steps on 2D Atari" is a category error — Director's 200M budget was for benchmarks strictly harder than ours (Egocentric Ant Maze has ~5×10⁴ step horizons; Isaac is 1.8×10³). At sample efficiency 3–10× DreamerV2's, Director on Isaac fits within the same compute envelope proponent claims for BC-fine-tune. *Confidence: High.*
- **[REBUT — Proponent's new argument #7 "Demo-mined option discovery is a free byproduct of BC"].** True in principle; false in practice at our data scale. Fox 2017 ([arXiv:1703.08294](https://arxiv.org/abs/1703.08294)) and Krishnan 2017 DDCO used 10⁵+ trajectories to cluster stable option boundaries. Our 40–80h × 4 runs/hour ≈ 200–320 trajectories is 3 OOM below the demonstrated threshold. Demo-mined options on our corpus will be underpowered clustering artefacts, not learned skills. Hand-designed options (or Option-Critic learned β) are the operative alternative — and both are BC-independent. *Confidence: Medium-High.*
- **[REBUT — Proponent's new argument #8 "Time-to-first-signal favours BC ~10×"].** Proponent's own R2 timeline shows this backwards. BC critical path: obs-fix (2 wk) + action-fix (0.5 wk) + demo-recording infrastructure (1 wk) + demo collection (1–2 wk wall-clock at 0.6× real-time × 3 humans coordination) + BC train (day) + KL-regularised RL fine-tune (weeks, unknown convergence) ≈ **6–10 weeks to first meaningful signal**. Hierarchical critical path: obs-fix (2 wk, shared with BC) + option-selector head (0.5 wk) + Option-Critic β wiring (1 wk) + SMDP-aware actor loss patch (0.5 wk) ≈ **4 weeks to first signal**, with the horizon fix landing *inside* the same 4 weeks. Proponent's 10× ratio charges BC nothing for the demo-collection wall-clock and charges Hierarchical for a from-scratch Dreamer retrain that is not actually required (Director is a local wrapper).

---

## Novel ideas surfaced by the debate

### Novel Idea A — Composition inversion: build the hierarchy first, then let BC ride on top at 50–100× lower demo cost.

Both sides implicitly accepted "BC → RL fine-tune → maybe options later" as the composition ordering. That ordering is backwards under the demo-cost accounting the debate has surfaced.

**Standard ordering (proponent's):** primitive BC on ~1,800 decisions per episode × ~200 episodes = ~360,000 primitive demonstrations. Wall-clock 40–80h × 3 humans × in-mod re-recording. Cost $1.2–2.4k (proponent's revised) to $3–18k (my R1). Stranded on obs/action churn.

**Inverted ordering:** 
1. Ship R0 (obs + action rehab). Shared with all downstream paths.
2. Wrap the existing DreamerV3 actor as an Option-Critic **intra-option** controller and add a manager head (Director-style, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)). Learned β(s); no `door.target_room_type` expansion needed.
3. Bump `imag_horizon` at the manager level to 20 options × ~1 s each = 20 s, comfortably inside a room clear. Value target grounds on real terminations, not `cont≈1`.
4. *Now* collect BC — but only at the **option-selector granularity**. A human plays and marks "take the north door / clear this room / buy the pedestal / use active" = **10–30 macro-decisions per episode**, 50–100× fewer than primitive BC. A single engineer self-collects 200 episodes in a weekend. Zero external labour. Zero coordination overhead. Zero schema-churn stranding, because the option interface is stable code even when obs fields churn.
5. KL-regularised RL fine-tune of the option-selector against the option-level BC prior. Because the demo is at option-granularity, DAPG's wash-out timescale of ~500 iterations becomes ~500 option-decisions, which is ~25–50 episodes — trivially within a persistent-demo-replay buffer.

**Why this dominates:** primitive-level BC struggles because human `π(a|s)` marginalises over unobserved decisive features (Ross-Bagnell). Option-level BC is defensible in the same setup because option-level decisions ("take Devil deal or not," "use D6 here or save it") condition on features the *option-level* obs (room-type, active-charge readiness, HP, floor-number) already exposes. The obs-conditioning gap collapses at the abstraction level where the human's cognitive representation and the agent's state representation *actually match*. The un-hackability of CE loss (proponent's mechanical property) is preserved. The horizon fix (my mechanical property) is preserved. Labour cost is 50–100× lower. Schema-churn risk is lower. Time-to-first-signal is *faster* than primitive BC because the demo-collection step compresses from weeks to a weekend.

This is not a compromise — it is a strictly Pareto-superior plan to either pure side's proposal, and it emerged only because the debate forced both sides to quantify prerequisites they preferred to hand-wave.

### Novel Idea B — Cheap falsification test before committing to either investment.

Both proposals cost 4–10 engineer-weeks. Neither is falsifiable by argument at this stage. A cheap experiment ends the debate empirically:

- **Test:** in a ~2-LOC PR, bump `imag_horizon: 20 → 120` in the current Dreamer config, leave everything else including the uniform-random actor prior untouched, and re-run 40h.
- **Prediction under my thesis:** value-inflation loop breaks or substantially attenuates. `actor_target_mean` drops from 47 toward the r_beat_mom scale (~5–10 with proper λ). `actor_entropy` decays from 3.804 measurably. `floors_reached_max` moves off zero. Cost: 40h of GPU time already budgeted, no new labour.
- **Prediction under proponent's thesis:** nothing changes. The prior is the bottleneck, and horizon extension without a coherent actor produces the same 1270:1 target-advantage ratio at longer imagined-rollout timescale (or diverges).
- **Decision rule:** if horizon-bump alone unblocks the run, BC's remaining marginal value is at most "improve rooms_visited from N to 1.5N" — a modest polish that would need much stronger justification than the $1.2–2.4k demo bill. If horizon-bump does *not* unblock, my Objection 1 is falsified as sole root cause and the debate shifts to the compose-both regime (Novel Idea A), where the ordering is Director → option-BC anyway.

Either outcome dominates spending 6–10 weeks on primitive-level BC before running a 2-LOC controlled test. **Neither side proposed this in R1 or R2.** It is the single-highest-EV action the debate has surfaced.

---

## Load-bearing scoreboard after Round 3

| Claim (mine) | Status | Steelman classification |
|---|---|---|
| Horizon is fixed-point attractor; BC does not fix (R1 #1) | **STANDS.** Steelman created regime R1′ to accommodate. | [VALID] |
| BC labour + schema-churn write-down (R1 #2 + R2 #9) | Stands as expected-value argument. | [CONDITIONAL] — but conditional on assumptions BC also needs |
| Human obs ≠ agent obs → marginalised BC (R1 #3) | Partially defused by in-mod recording. Residual on minimap/intuition. | [CONDITIONAL] |
| Stripped action space → BC drops key frames (R1 #4) | Conceded prerequisite for BC. Shared, but Hierarchical needs less. | [VALID] |
| Sample-scale gap vs NetHack (R1 #5) | **STANDS and strengthened.** Proponent's revised numbers put BC below human-median. | [CONDITIONAL — true if BC terminal] |
| BC ceiling < late-floor objectives (R1 #6) | Deferred to when BC becomes terminal. | [CONDITIONAL] |
| Reward-alignment via SMDP option termination (R1 #7) | Genuine synergy. Adopted by steelman into R3 mechanism. | [VALID] |
| BC of Dreamer-imagined rollouts unsound at H>5 (R2 #8) | Unrefuted. | (Not classified) |
| Option-Critic learns β(s) end-to-end — kills "shared prerequisite" defense (R2 REBUT 5) | **STANDS.** Steelman explicitly marked proponent's "strictly cheaper marginal" claim as **[FALSE]**. | [VALID] |
| Options break value-inflation loop mechanically via SMDP (R2 #10) | **STANDS.** SMDP theory settled. | (Aligned with steelman [VALID]) |
| Director is strictly local upgrade to Dreamer (R2 #11) | **STANDS.** Proponent's counter (200M env-steps) was a category error. | (Not classified) |

---

## Final recommendation for the audit team

Not "BC vs Hierarchical." The dominant plan the debate has produced is:

1. **Week 0 (2 LOC):** Bump `imag_horizon 20 → 120` and re-run 40h. Novel Idea B. If value-inflation loop breaks, most of the BC case evaporates and the horizon fix alone gets us to R1′.
2. **Weeks 1–2:** R0 obs + action rehab. Restore `use_active / drop_bomb / pill_card` heads. Expand `doors[i].target_room_type` from 3 to 15 categories. Add active-charge and transformation counters. This is shared prerequisite work every plan agrees on.
3. **Weeks 3–4:** Wrap the existing DreamerV3 actor as an Option-Critic intra-option controller with learned β(s). Add manager head. Bump manager-level `imag_horizon` to option-timescale. This is a Director-style local upgrade, not a rewrite.
4. **Weeks 5–6:** Solo-engineer collects **option-level** BC demos in a weekend (Novel Idea A). Wire KL-regularised RL fine-tune of the option-selector. Retire the reward-shaping treadmill because CE-on-options is un-hackable *and* option-terminal rewards are grounded in real game events (best of both mechanical properties).
5. **Week 7+:** Iterate on option-selection reward and character conditioning. Character conditioning is cheap here because it's a manager-level embedding, not a per-character demo recollection.

Total cost: ~7 weeks, ~$0 external labour, ~40h additional GPU (Week 0 test). Compare to proponent's 6–10 weeks + $1.2–2.4k best-case demo bill for a plan whose weakest link (horizon fix) has been conceded across all three positions.

## Sources cited across Rounds 1–3
- Bacon, Harb, Precup 2017. *Option-Critic Architecture.* [arXiv:1609.05140](https://arxiv.org/abs/1609.05140)
- Hafner et al. 2022. *Deep Hierarchical Planning from Pixels (Director).* [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)
- Hafner et al. 2023. *DreamerV3.* [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Vezhnevets et al. 2017. *FeUdal Networks.* [arXiv:1703.01161](https://arxiv.org/abs/1703.01161)
- Li et al. NeurIPS 2019. *HAAR.* [nips.cc](https://papers.nips.cc/paper_files/paper/2019/file/81e74d678581a3bb7a720b019f4f1a93-Paper.pdf)
- Sutton, Precup, Singh 1999. *Between MDPs and Semi-MDPs.* AIJ 112.
- Rajeswaran et al. 2017. *DAPG.* [arXiv:1709.10087](https://arxiv.org/abs/1709.10087)
- Hambro et al. 2022. *Dungeons and Data.* [arXiv:2211.00539](https://arxiv.org/abs/2211.00539)
- Vinyals et al. 2019. *AlphaStar.* [Nature 575:350](https://www.nature.com/articles/s41586-019-1724-z)
- Hester et al. 2018. *DQfD.* [arXiv:1704.03732](https://arxiv.org/abs/1704.03732)
- Ross, Gordon, Bagnell 2011. *DAgger.* [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Ren et al. 2021. *Generalization Guarantees for Imitation Learning.* [proceedings.mlr.press](https://proceedings.mlr.press/v155/ren21a/ren21a.pdf)
- Janner et al. 2019. *When to Trust Your Model.* [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Freeman et al. 2019. *Learning to Predict Without Looking Ahead.* [arXiv:1910.13038](https://arxiv.org/abs/1910.13038)
- Fox et al. 2017. *Multi-Level Discovery of Deep Options.* [arXiv:1703.08294](https://arxiv.org/abs/1703.08294)
- Krishnan et al. 2017. *DDCO.* [arXiv:1710.05421](https://arxiv.org/abs/1710.05421)
- Cobbe et al. 2020. *ProcGen benchmark.*
- Local: `isaac-swarm/agent1-item-economy.md`, `agent2-character-strategy.md`, `agent3-adversarial-audit.md`.
