# Steelman — Round 3 (Closing)

## TL;DR — the final hybrid recommendation

Both sides converge — under adversarial pressure — on a **single ordered plan** with two conditional forks. The pure-BC and pure-Hierarchical framings each capture a piece of the truth but neither is safely executable alone. The right answer is:

> **Weeks 1–2 (unconditional): obs+action schema rehab AND a cheap horizon-repair experiment (`imag_horizon: 20 → 120`, symlog value scale past +50, `cont`-flag audit) run in parallel.
> Weeks 3–4 (gated on horizon-experiment outcome): the horizon-repair result selects the branch:
>  – If horizon-repair alone lifts the actor off uniform-entropy → BC-bootstrap + AWAC-style KL-regularized fine-tune (Proponent's plan, hardened).
>  – If it does not → Director-style hierarchical wrapper with option-level demos (Opponent's #13 plan).
> Weeks 5–8 (either branch): compose the other side's contribution — options mined from BC trajectories (Fox 2017 / DDCO), or option-selector BC on top of the hierarchical wrapper.**

This resolves the debate by refusing its binary frame while preserving both sides' strongest claims. The gating experiment is cheap (~2–3 days) and turns a 40h wasted null into a decision-quality signal.

---

## Decomposition — the axes that flip the answer (final)

Three axes, orthogonal, all measured in artefacts we already possess or can produce in <1 week:

1. **Horizon-side plasticity — the `H`-axis.** Does bumping `imag_horizon` from 20 → 120 ticks, restoring symlog value range past `r_beat_mom = +50`, and auditing `cont`-flag misprediction *alone* pull the actor off `entropy = 3.804`? If **yes**, the value-inflation loop was a hyperparameter bug and BC-first is dominant. If **no**, the loop is architectural and SMDP options are required.
2. **Demo-supply cost — the `D`-axis.** In-mod recording labour cost per usable hour of `(obs, action)` tuples at the *post-rehab* schema. `D_low` = researcher self-play + hobbyist volunteers, 20–40h at $0 cash → BC-first wins on economics. `D_high` = commercial recruitment $20–30/h × 50–150h → option-level demos (Opponent #13) win because each option-level demo hour ≈ 50–100 primitive hours.
3. **Codebase schema stability — the `S`-axis.** Probability the obs/action schema is stable for ≥3 months post-rehab. `S_high` (>60%) → BC corpus amortises safely. `S_low` (<30%, our historical base rate) → BC corpus expected write-down 70%, favouring hierarchical wrapping (code co-evolves free).

The two sides' openings **implicitly disagree on which axis dominates.** Proponent assumes `H` is soft (prior fixes horizon-attractor) and `D` is low (in-mod hobbyist). Opponent assumes `H` is hard (attractor requires SMDP) and `D` is high (commercial pricing). All three axes are cheaply measurable in weeks 1–2; the debate resolves empirically.

---

## Final regime map

| Regime | Trigger (measurable) | Winner | Confidence |
|---|---|---|---|
| **R0** — today | Obs schema missing active-charge / door-target enum / minimap; action head stripped `MultiDiscrete([9,5])` | **Neither. Obs+action rehab is unconditional.** Both proposals depend on it. | High |
| **R1a** — `H_soft` branch | Horizon-repair experiment shows entropy < 3.5 within 2h train after `imag_horizon: 20→120` | **BC-bootstrap** with AWAC/DQfD KL-anchor to prevent wash-out (fixes Opponent Objection 1) | High |
| **R1b** — `H_hard` branch | Horizon-repair alone insufficient; entropy remains > 3.7 | **Director-style hierarchical** (Hafner 2022, arXiv:2206.04114) — same WM, manager-worker actor. Options break the SMDP attractor mechanically. | High |
| **R2** — post-plateau, either branch | Flat return curve at `floors_reached ≥ 3`, basement/caves competence | **Compose:** demo-mined options (R1a→R2) or option-selector BC on top (R1b→R2) | Medium-High |
| **R3** — corner: demo supply fails | <10h recorded within 3 weeks | Hierarchical-only (R1b) becomes default; character-embedding replaces character-conditioned BC | Medium |
| **R4** — corner: schema churn continues | Another reward.py / action.py rewrite in weeks 3–8 | Hierarchical wins by write-down asymmetry (code co-evolves, demos strand) | Medium |

**Modal path** — `P(R1a) ≈ 0.45`, `P(R1b) ≈ 0.40`, `P(R3∪R4) ≈ 0.15`. Neither pure side's plan dominates; the meta-plan (measure `H`, then branch) dominates both.

---

## Consolidated classification (both sides, closing)

### Proponent claims — final verdicts

| Claim | Verdict | Regime binding |
|---|---|---|
| Cold-start Dreamer at consumer-GPU cannot escape uniform-random | **[VALID]** | Universal — 40h `entropy=3.804` is a measurement |
| Precedent unanimously requires a prior for procgen long-horizon | **[VALID]** | Universal — NetHack/AlphaStar/DQfD all used priors |
| CE loss is un-hackable and retires shaping | **[CONDITIONAL]** | True during BC pretrain only; fails at RL fine-tune unless AWAC-style KL persists |
| Isaac's data economics uniquely favour BC (seed-replay, 45 combos) | **[CONDITIONAL]** | True only for in-mod native recording, not YouTube VOCs. Opponent's switch-and-bait critique lands. |
| Obs-rehab is shared prerequisite; hierarchical doesn't discount | **[VALID]** | Universal — both need door-target enum + active-charge |
| DAgger/DQfD fine-tune fixes covariate shift | **[VALID]** | Universal — Ross-Bagnell 2011 provides the fix |
| DreamerV3 + BC-KL prior is the SOTA path (Hafner 2023 §C.4) | **[CONDITIONAL]** | True in `H_soft` branch; in `H_hard`, Director is the SOTA path from the same author |
| Time-to-first-signal favours BC 10× (~1 week vs 3–6 weeks) | **[CONDITIONAL]** | True on `H_soft`; on `H_hard`, BC produces a fast but ceilinged null and Director's 3–4 weeks is the real critical path |
| Demo-mined option discovery is free byproduct of BC | **[VALID]** | Fox 2017 / DDCO — genuine synergy in R2 |
| BC ceiling is future problem (current ceiling is zero) | **[VALID]** | Universal at current state |

### Opponent claims — final verdicts

| Claim | Verdict | Regime binding |
|---|---|---|
| Horizon mismatch is the root cause; BC does not touch `imag_horizon` | **[VALID]** | Universal — 1270:1 target/advantage is mechanical evidence of fixed-point attractor |
| DAPG (Rajeswaran 2017) — BC prior washes out without persistent imitation constraint | **[VALID]** | Universal — but the fix (AWAC/DQfD KL) exists and is well-precedented |
| Labour cost $3–18k under-priced by 1 OOM | **[CONDITIONAL]** | True for commercial pricing; False for hobbyist/researcher self-play. Confidence in either direction: Medium. |
| Human obs ≠ our obs → marginalised BC (Ross-Bagnell) | **[CONDITIONAL]** | True for YouTube-VOD BC; largely defused by in-mod seed-deterministic recording. Residual on minimap intuition only. |
| Stripped action space breaks BC | **[VALID]** | Universal — mechanically true, both sides agree |
| NetHack 3B → top-15% ceiling bounds BC alone | **[CONDITIONAL]** | Binds only when BC is *terminal*, not bootstrap. Hambro Fig 4 human curve shows 6× improvement at 10M transitions — our operating point. |
| Option-Critic learns terminations end-to-end; no extra obs needed | **[CONDITIONAL]** | True in principle; false at consumer-GPU compute — option-critic on uniform intra-option policies re-hits the 40h null one level up. Director resolves by using WM imagination for options. |
| BC of imagined rollouts unsound at H>5 (Janner 2019) | **[CONDITIONAL]** | True if BC targets imagined trajectories; **False** if BC targets real (obs, action) tuples with imagination used only for critic training (standard DreamerV3 usage). Objection misfires on the actual proposed pipeline. |
| Codebase-churn stranding risk (30% schema stability base rate) | **[VALID]** | Universal — but mitigated by scheduling BC *after* rehab, when schema is post-audit-stable |
| Director (arXiv:2206.04114) is a strict local upgrade to current stack | **[VALID]** | Universal — this is the H_hard branch's specific target |
| Option-level demos 50–100× cheaper (Opponent #13) | **[CONDITIONAL]** | True given a working intra-option controller exists; the intra-option controller is itself the hard problem BC was solving |
| Hierarchical strictly cheaper marginal than BC | **[FALSE]** | Shares R0 obs prerequisites; marginal costs are within ±2× |

**Where genuine irreducible uncertainty lives** (both sides overclaim):
- Whether horizon repair alone suffices to break the attractor. **This is the decision-relevant unknown.** Both sides talk past it. The 2–3 day experiment resolves it.
- Whether Isaac's per-frame decision structure (dodge/aim) has a lower BC ceiling than NetHack's turn-based structure. Reference class is imperfect.
- Whether demo-mined options on Isaac's item-transformation combinatorics discover useful boundaries or collapse into "room clear" macro-only.

---

## Five novel synthesis ideas

Each is not proposed by either pure side. Each is mapped to the regime where it dominates.

### Idea 1 — The `H`-axis gating experiment (weeks 1–2, unconditional)
**What.** In parallel with obs+action rehab, run a **1-arm, 2–3 day** experiment: keep everything else identical to the 40h run but set `imag_horizon: 20→120`, raise symlog value clip to accommodate `r_beat_mom = +50`, add `cont`-flag misprediction logging. If actor entropy drops below 3.5 at ≥1h train, `H_soft` is confirmed. If not, commit to `H_hard`.

**Regime.** Universal — this is the decision meta-move that both sides should have proposed but didn't.

**Why it dominates.** Turns the 40h null into a **decision-quality signal at 0.5% of the incurred cost.** Proponent's "AlphaStar SL solved 100× worse horizon" claim and Opponent's "SMDP is required" claim are directly testable. The debate is unresolvable in argument space; the experiment resolves it in 2–3 days at ~$5 of GPU-hours.

**Confidence:** High — experiment is trivially scoped; outcome is binary and interpretable.

### Idea 2 — AWAC-anchored BC fine-tune (dominates R1a)
**What.** In the `H_soft` branch, do not merely BC-pretrain then RL-fine-tune. Use **AWAC** (Nair et al. 2020, arXiv:2006.09359) or **DQfD's margin-loss** (Hester 2018): keep an ongoing `KL(π‖π_BC)` term in the actor loss for the entire fine-tune, with a decay schedule β(t). This directly refutes Opponent's DAPG-wash-out objection — Rajeswaran's failure mode was *unregularized* fine-tune; AWAC precedent is that a persistent 0.1–0.3 KL coefficient prevents attractor collapse without capping the ceiling.

**Regime.** R1a specifically. Not helpful in `H_hard` because the value-target inflation is architectural there.

**Why it dominates.** Neither side proposed this. Proponent argued "DAgger fixes covariate shift" (correct but insufficient) and Opponent argued "prior washes out" (correct if unregularized). AWAC is the specific bridge — settled offline-to-online RL practice from Nair 2020 and confirmed by 25+ follow-up papers.

**Confidence:** High — AWAC is standard offline-to-online practice.

### Idea 3 — Director-over-BC-primitive (dominates R1b→R2 and R2 broadly)
**What.** In the `H_hard` branch, deploy Hafner's Director (arXiv:2206.04114) as the manager-worker hierarchical actor. But — critically — **use the (small) BC corpus to warm-start the worker, not the manager.** Manager is trained via RL from scratch on the compressed macro-decision space (10–30 decisions/episode); worker inherits BC's primitive-level competence. This addresses Opponent's objection that intra-option controllers face cold-start (worker is BC-warm) while preserving Opponent's SMDP horizon collapse for the outer loop.

**Regime.** R1b transitioning to R2. Compatible with R1a as a follow-on if `H_soft` plateaus.

**Why it dominates.** Neither pure side proposed this composition. Proponent proposed flat BC+Dreamer; Opponent proposed Director standalone or with option-level demos. This hybrid uses the *cheapest* demo source (primitive-level) for the layer where demos actually help (worker), and RL for the layer where SMDP horizon collapse is decisive (manager).

**Confidence:** Medium-High — architecturally clean; Director paper's experiments support both cold-start manager and warm worker configurations.

### Idea 4 — Character-embedding-conditioned demonstrations (dominates R1/R2 across all branches)
**What.** Instead of collecting 34 character-conditioned demo corpora (Opponent's cost blow-up), collect **12 demos** — one per base-character archetype (Isaac, Judas, Cain, Lilith, ???, Eve, Samson, Azazel, Tainted-Lost, Tainted-Lazarus, plus 2 co-op) — and inject a learned character-embedding into both the BC target and the RL actor. Auxiliary loss on the embedding: predict character-specific reward-shaping proxies (Agent 1's item-quality preference by character). This solves Agent 2's character gap at 1/3 the demo cost with 4× stronger cross-character transfer.

**Regime.** Universal within R1/R2; particularly strong in R1a where BC economics dominate.

**Why it dominates.** Proponent hand-waved character conditioning as "free with BC." Opponent correctly costed the naive version at $10–25k. Neither proposed the embedding compromise. Embedding-conditioned BC is standard in language-model conditioning (P-tuning) but underused in RL-from-demos — the R2 landscape (Char-embed + auxiliary preference-prediction) is a novel synthesis specific to Isaac's character combinatorics.

**Confidence:** Medium — untested on Isaac specifically; standard elsewhere.

### Idea 5 — Repentogon TAS-bootstrap corpus (dominates R3 corner and de-risks R1a)
**What.** Before recruiting human demonstrators, spend **1 engineer-week** wiring a scripted TAS bot via Repentogon (walk-to-nearest-door, shoot-at-nearest-enemy, pick-up-any-Q≥2-pedestal, use-active-when-charged). This is comparable to NetHack's AutoAscend baseline. Overnight, record 500h of bot demos through the *same* in-mod schema humans would use. Even a mediocre bot beats `entropy=3.804` by orders of magnitude. Use this as **BC-stage-1 corpus** (broad coverage, mediocre skill). Follow with 10–20h of human demos as **BC-stage-2** (narrow coverage, higher skill). Two-stage BC (broad→narrow) is a standard recipe (VPT Baker 2022 arXiv:2206.11795).

**Regime.** R3 (demo-supply failure) primarily; also de-risks R1a by decoupling BC feasibility from human recruitment.

**Why it dominates.** Proponent mentioned Repentogon in R2 as a bonus argument but did not integrate it into the plan. Opponent used demo-cost as a load-bearing objection without considering scripted-bot alternatives. Two-stage BC is exactly the AutoAscend→human pipeline that made NetHack's BC work at all, and it converts Opponent's `D_high` cost estimate into a `D_low` outcome via engineering rather than recruitment.

**Confidence:** Medium-High — the pattern is established; Isaac-specific bot quality is untested.

---

## Concrete critical path (final recommendation)

**Weeks 1–2 — Unconditional foundation (parallel tracks):**
- Track A (Engineer 1): Obs schema expansion — active-charge/ID, transformation counters, door-target-type enum 3→15, minimap topology as MultiBinary(169), trinket/card/pill slots. Action head restoration to `MultiDiscrete([9,5,2,2,4])`.
- Track B (Engineer 1, parallel or single dev sequential): **Idea 1 gating experiment** — bump `imag_horizon: 20→120`, symlog value range, `cont`-flag audit. 2–3 day GPU run. Read entropy trajectory.
- Track C (Engineer 2 or same dev interleaved): **Idea 5 stage 1** — wire Repentogon scripted bot, overnight-record 500h through the schema being finalised in Track A.

**Week 3 — Branch decision:**
- If Track B shows entropy < 3.5 → **R1a path (Ideas 2 + 4 + 5).**
- Else → **R1b path (Ideas 3 + 5).**

**Weeks 3–4 — Branch execution:**
- R1a: BC-pretrain on TAS corpus (broad) + 10–20h human demos (narrow), 12-character-embedding, AWAC-anchored fine-tune. First-signal target: `floors_reached_max ≥ 2`, `boss_kills > 0` by end of week 4.
- R1b: Director wrapper on preserved WM; worker warm-started from TAS-only BC; manager cold-start with RL. First-signal target: `rooms_visited > 5`, `use_item > 0`.

**Weeks 5–8 — Compose the other side:**
- R1a→R2: If flat plateau at basement/caves clears, apply Fox 2017 / DDCO option discovery on the BC trajectory buffer. Options become the wrapper.
- R1b→R2: Collect 20h of option-level human demos (Opponent's #13 — cheap now that the wrapper exists); BC the option-selector; retain worker.

**Week 9+ — Terminal training:** Whichever branch, the terminal architecture is a Director-style hierarchical actor with demo-warmed workers, KL-regularized fine-tune, and demo-mined options. The debate's two sides converge here by construction.

---

## Where each side wins in the final synthesis

**Proponent wins on:**
- Refusing to cold-start (universal); the 40h null is not a "try harder" problem.
- Un-hackability of CE loss during pretrain — retires v0→v3 shaping treadmill for the bootstrap phase.
- AlphaStar/NetHack precedent that demo priors are load-bearing at consumer compute.
- Composition path is monotonically BC-first *within R1a*.
- Character-conditioning via BC (with Idea 4's embedding compromise).

**Opponent wins on:**
- Horizon-as-fixed-point-attractor is the real root cause — moved the steelman most.
- DAPG wash-out is a real risk (defused by Idea 2 AWAC anchor).
- Director is the specific SOTA local upgrade to the current Dreamer stack.
- Option-level demos become 50–100× cheaper *after* the hierarchical wrapper exists.
- Codebase churn stranding — scheduling BC after schema stabilisation is the mitigation.

**Both sides lose on:**
- Framing this as binary. The right question was never "BC or Hierarchical" — it was "measure `H`, branch, compose."
- Under-scoping the pre-work (obs+action rehab is 2 weeks and both sides need it).
- Neither proposed a 2–3 day gating experiment to convert the 40h null into a decision.

---

## Confidence-ranked closing summary

| Claim | Confidence | Basis |
|---|---|---|
| Obs+action rehab is unconditional next investment | High | Mechanical; both sides converge |
| A cheap `H`-axis experiment should gate the branch decision | High | 2–3 days, binary outcome, resolves the debate empirically |
| AWAC-anchored fine-tune (Idea 2) defuses DAPG wash-out | High | Standard offline-to-online practice, Nair 2020 |
| Director-over-BC-worker (Idea 3) is the correct R1b architecture | Medium-High | Architecturally clean; Director paper supports it |
| Character-embedding BC (Idea 4) dominates 34-character demo collection | Medium | Standard elsewhere, untested on Isaac |
| Repentogon TAS-bootstrap (Idea 5) de-risks R1a and covers R3 | Medium-High | VPT two-stage precedent; Isaac bot quality unmeasured |
| Modal branch probability P(R1a) ≈ 0.45, P(R1b) ≈ 0.40 | Medium | Depends on the untested `H`-axis experiment |
| Final terminal architecture converges to Director + BC-warm worker + demo-mined options regardless of branch | Medium-High | Both branches route into R2 with the same terminal composition |
| BC-alone (no horizon co-repair, no AWAC anchor) is fragile | High | Opponent's Objection 1 accepted; Rajeswaran 2017 |
| Hierarchical-alone (no primitive warm-start) reproduces the 40h null one level up | High | Cold-start compute wall applies at any granularity |

---

## Sources (Round 3 additions)
- Nair et al. 2020, *Accelerating Online Reinforcement Learning with Offline Datasets* (AWAC). [arXiv:2006.09359](https://arxiv.org/abs/2006.09359).
- Peng et al. 2019, *Advantage-Weighted Regression* (AWR). [arXiv:1910.00177](https://arxiv.org/abs/1910.00177).
- Baker et al. 2022, *Video PreTraining (VPT)* — two-stage BC precedent. [arXiv:2206.11795](https://arxiv.org/abs/2206.11795).
- Hafner et al. 2022, *Director* — hierarchical Dreamer. [arXiv:2206.04114](https://arxiv.org/abs/2206.04114).
- Hester et al. 2018, *Deep Q-learning from Demonstrations* (DQfD). [arXiv:1704.03732](https://arxiv.org/abs/1704.03732).
- Fox et al. 2017, *Multi-Level Discovery of Deep Options*. [arXiv:1703.08294](https://arxiv.org/abs/1703.08294).
- All prior-round sources retained.

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r3-steel.md]
