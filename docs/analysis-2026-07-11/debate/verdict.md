# Verdict: BC-bootstrap vs Hierarchical-actor as the next Isaac RL investment

## Summary
The debate resolved into a **conditional win for BC-bootstrap with important caveats**, not the clean victory the thesis implies. Both sides — under pressure — converged on three points neither started with: (i) obs+action rehab is a shared, unconditional prerequisite that beats both proposals as the literal *next* action; (ii) BC without horizon co-repair (`imag_horizon 20→120`, symlog value scale, `cont`-flag audit) is fragile; (iii) a 2–3 day gating experiment on the existing checkpoint resolves the load-bearing disagreement more cheaply than either full proposal. Given those caveats, BC still comes out ahead of a pure Hierarchical-actor investment because every viable composition path in the debate (including the Opponent's own R2 novel idea #13 and the Steelman's R3 hybrid) routes through a BC layer, whereas the reverse is not true.

---

## Claim ledger

| # | Claim | Source persona / round | Evidence quality | Status |
|---|-------|------------------------|------------------|--------|
| 1 | Cold-start Dreamer at consumer GPU cannot escape uniform entropy after 40h (`actor_entropy = 3.804/3.807`, `floors_reached_max = 0`) | Pro R1 §2; conceded Opp R1 §steelman; Steel R1 R0 | High — direct instrumented measurement from `tb_dreamer_stage1_20260709-174421.json` | **CONFIRMED** |
| 2 | On procedural/sparse/long-horizon games at consumer-GPU compute, all published successes used a demo prior (NetHack, AlphaStar, DQfD; Dota's alternative is compute we lack) | Pro R1 §1, R3 §1 | High — 4 peer-reviewed references | **CONFIRMED** as a reference-class fact; **CONTESTED** on whether it implies "BC-first at *our* scale" |
| 3 | NetHack's Hambro 2022 human-BC curve at 10M transitions gives ~6× improvement over cold-start (500→3000 mean score) | Pro R2 revised, R3 §1 | High — Hambro 2022 Fig 4 | **CONFIRMED** as a data point; **CONTESTED** on interpretation (Opp R3: 3000 is sub-median-human) |
| 4 | Value-target inflation (`actor_target_mean=47` vs `actor_adv_abs_mean=0.037`, 1270:1 ratio) is a fixed-point attractor of the actor-critic dynamics under horizon truncation, not a bad prior | Opp R1 §1, R2 REBUT1, R3 §1; Steel R2 explicitly adopted this as "biggest update R1→R2" | High — audit measurement + SMDP Bellman theory (Sutton/Precup/Singh 1999) | **CONFIRMED** |
| 5 | Therefore a BC prior alone (no horizon co-repair) is washed out by RL fine-tune within ~5–10k gradient steps (DAPG precedent, Rajeswaran 2017) | Opp R1 §1, R2 REBUT1 | Medium-High — DAPG is direct analogue but on different domain; Steelman graded [VALID] | **CONFIRMED** (weight: BC needs AWAC/DQfD-style KL anchor, not standalone) |
| 6 | Pro's counter that AlphaStar solved a 100× worse horizon with SL alone (no hierarchy) | Pro R2 REBUT1, R3 §1 | Medium — AlphaStar used SL+league+auto-regressive factored action head, which Opp R3 argues is structurally hierarchical | **CONTESTED** — Opp's counter (league solved horizon via match-segment structure, not SL prior alone) is credible; neither cite settles this |
| 7 | Obs schema is broken for both proposals: missing active-charge/ID, transformation counters, door-target enum (3/15), minimap, trinket/card/pill; only 3/15 room types entered in 40h | Steel R1 R0; Opp R1 §3; Pro R1 §5 concedes | High — grep-able from code + audit | **CONFIRMED** (shared prerequisite) |
| 8 | Action head `MultiDiscrete([9,5])` cannot represent SPACE / bomb / card; `use_item = 0`, `bombs_used_max = 1` in 40h | Opp R1 §4; Steel R1; Pro R2 REBUT4 concedes | High — code diff on 2026-07-02 | **CONFIRMED** (shared prerequisite) |
| 9 | Human obs ≠ agent obs → BC converges to marginalised policy `E_missing[π_H(a|obs,missing)]` with Ross-Bagnell `O(H²)` compounding error | Opp R1 §3 | Medium-High for the theory; Medium for magnitude on Isaac specifically | **CONTESTED** — partially defused by (a) in-mod native recording (Pro R1 §4), (b) DAgger/AWAC fine-tune (Pro R2 REBUT3, Steel [VALID]). Residual risk on minimap/HUD intuition remains. |
| 10 | Demo labour cost estimate | Opp R1 §2: $3–18k commercial; Pro R2 REBUT2: $1.2–2.4k or $0 self-play; Steel R2 [CONDITIONAL] | Medium — no direct measurement; depends on unstated project economics | **CONTESTED** — 5–10× spread between reasonable estimates |
| 11 | Codebase schema stability base rate is <30% over 3 months (evidence: reward.py v0→v3 in 2 months; action head amputated 2026-07-02) | Opp R2 §9 | Medium — plausible base rate, uncited external comparison | **CONFIRMED** as a risk factor; mitigation (schedule BC *after* rehab) reduces severity |
| 12 | BC-of-imagined-rollouts is unsound at H>5 (Janner 2019 model-based bound) | Opp R2 §8 | Low as applied — Steelman R3 marked this a category error: BC targets real (obs,action) pairs, not imagined states; imagination is only in critic training | **UNSOURCED as applied** (misfires on proposed pipeline) |
| 13 | Option-Critic learns β(s) end-to-end, so the "hierarchical needs expanded door-target enum" objection is defused | Opp R1 §5, R2 REBUT5, R3 §3 | High — Bacon 2017 architecture is settled | **CONFIRMED** (undercuts one of Pro's arguments, but see #14) |
| 14 | Option-critic on uniform-random intra-option policies reproduces the 40h null at option granularity ("credit assignment relocates, not collapses") | Pro R2 REBUT1; Steel R1 R1 last bullet; Steel R3 | Medium-High — mechanical argument, no direct experiment | **CONFIRMED** — Opp did not refute; instead pivoted to Director (WM-based imagination at option scale) |
| 15 | Director (Hafner 2022) is a strict local upgrade to the existing DreamerV3 stack (manager+worker over same WM), demonstrated 3–10× sample efficiency on sparse-reward long-horizon nav | Opp R2 §11, R3 REBUT | High — peer-reviewed, same author as our stack | **CONFIRMED** — this is the strongest architectural counter Pro did not fully address |
| 16 | Director's own paper reports 200M env-steps to converge on 2D Atari; at our 2.36 sps = 24,000 wall-clock hours | Pro R3 §3 | Medium — Opp R3 counters this is category error (Director's 200M was on harder benchmarks) | **CONTESTED** — extrapolation uncertain |
| 17 | WM losses are converging (`enemies_mask 11.8→0.006`, `doors 9.9→0.006`); only the actor is broken. BC = targeted actor-swap preserving 40h of WM training | Pro R1 §implication 3, R2 §6, R3 §3 | High — direct measurement | **CONFIRMED** — Steelman R2 adopted this as an update; Opp did not directly refute (only noted WM would be usable in either scheme) |
| 18 | BC is the earliest common ancestor of every viable composition path in this debate (Opp R2 idea #13 has BC on option-selector; Steel R3 has BC-warm worker; Pro's plan is BC-then-options) | Pro R3 §2 | High — verifiable from the transcripts themselves | **CONFIRMED** — even the Opponent's proposed inverted-ordering plan still uses BC, just at a different granularity |
| 19 | Time-to-first-signal favours BC ~5–10× (~1 week vs 3–6 weeks) | Pro R2 §8, R3 §3 | Medium — both timelines are estimates, not measurements; Opp R3 gives BC 6–10 wk vs Hierarchical 4 wk, contradicting Pro's estimate | **CONTESTED** — depends on whether Director wrapping is treated as "1–2 wk local upgrade" or "full retrain" |
| 20 | Demo-mined option discovery (Fox 2017 / DDCO) is a free byproduct of BC | Pro R2 §7 | Medium — Opp R3 REBUT7 correctly notes Fox/Krishnan used 10⁵+ trajectories; our 200–320 trajectories is 3 OOM below | **CONTESTED** — Pro overstates; useful only as a bonus, not load-bearing |
| 21 | The 2–3 day `imag_horizon: 20→120` gating experiment resolves the load-bearing disagreement at ~0.5% of the cost of either full proposal | Opp R3 Novel B; Pro R3 Novel B; Steel R3 Idea 1 (all three converged in R3) | High — cheap, binary, decision-relevant | **CONFIRMED** — highest-EV action surfaced by the debate |

---

## Score

**58/100** in favour of the thesis.

### Reasoning (weighted)

Points that move toward the thesis (BC > Hierarchical as next investment):
- Claim 1 (cold-start is broken, +10): unanimous.
- Claim 2 (reference-class demo priors, +5): confirmed but bounded by scale critique.
- Claim 17 (WM preservation via actor-swap, +8): strong and largely unrefuted.
- Claim 18 (BC is earliest common ancestor, +12): strongest single move; the Opponent's own R2 novel idea and the Steelman's R3 both route through BC.
- Claim 19 (time-to-first-signal, +3): contested but plausible in the modal `H_soft` branch.
- Claim 3 (NetHack curve supports us at operating point, +3): confirmed at the small-data end even if Opp is right about the ceiling.

Points that move away from the thesis:
- Claim 4 (horizon-attractor is root cause, −8): confirmed; forced steelman to create R1′ (BC + horizon co-repair as co-requisite). Bare BC would fail.
- Claim 5 (DAPG wash-out, −4): confirmed; requires AWAC/DQfD anchor, not proposed by Pro until R3.
- Claim 7+8 (shared prerequisites, −5): the literal *next* investment is neither proposal; it is obs+action rehab. Thesis is technically false at t=0.
- Claim 15 (Director is strict local upgrade to Dreamer, −6): Pro's strongest gap; the "Hierarchical requires full retrain" straw-man was refuted.
- Claim 11 (schema-churn write-down, −2): real but mitigable.
- Claim 20 (demo-mined options overstated, −1): minor.

Net: 58 — a modest, conditional win for BC over Hierarchical, contingent on (a) horizon `H_soft` branch of the gating experiment, (b) AWAC-anchored fine-tune bundled with BC, (c) obs+action rehab completed first. Under `H_hard` the balance flips toward Director; the Steelman's ~0.45/0.40/0.15 branch probabilities put the expected value at roughly BC-favoured but not decisively.

Confidence in the 58 score: **Medium**. The number would move ±15 points depending on how the untested gating experiment resolves. This is the load-bearing unknown.

---

## Top novel ideas (ranked by expected value = payoff × feasibility ÷ cost)

1. **`imag_horizon` gating experiment (2–3 days on existing checkpoint)** — proposed by Opp R3 Novel B, Pro R3 Novel B, and Steel R3 Idea 1 (all three converged on this in the closing round; nobody proposed it in R1 or R2). Payoff: **very high** (resolves the load-bearing debate empirically). Cost: **~$5 of GPU-hours**. Novelty: **5/5** (emerged from synthesis; no persona held it in R1). Why it matters: converts a 40h wasted run into a decision-quality signal at 0.5% of the cost of either full proposal. This is the single highest-leverage action surfaced.

2. **AWAC-anchored BC fine-tune** — Steelman R3 Idea 2. Payoff: **high** (defuses DAPG wash-out, which was the strongest Opp objection). Cost: **~10 LOC + KL coefficient schedule**. Novelty: **4/5** (Nair 2020 is standard offline-to-online, but neither Pro nor Opp named it until Steel R3). Why it matters: preserves BC's leverage through fine-tune without capping the ceiling; refutes Rajeswaran 2017 objection directly.

3. **Director-over-BC-warm-worker composition** — Steelman R3 Idea 3. Payoff: **high** (both branches route into this terminal architecture; combines SMDP horizon-collapse with demo-warmed primitives). Cost: **medium** (~2 wk to wire Director wrapper on existing DreamerV3). Novelty: **4/5** (specific hybrid neither pure side proposed). Why it matters: this is the terminal architecture the debate converges on regardless of which branch wins in Week 3.

4. **Option-level demos are 50–100× cheaper than primitive-level** — Opp R2 Novel #13. Payoff: **high** (compresses demo collection from weeks to a weekend). Cost: **requires a working intra-option controller as prerequisite** (which is the hard part). Novelty: **4/5**. Why it matters: inverts the standard "primitive BC → options later" ordering into "options first → option-level BC on top." Depends on Director wrapper working before option-level demos have anything to attach to.

5. **BC as imagination-distribution shaper (not just policy prior)** — Pro R3 Novel A. Payoff: **medium-high** (persistent, compounding win rather than one-shot). Cost: **zero marginal on BC investment**. Novelty: **4/5** (compositional argument no persona explicitly made in R1/R2). Why it matters: reframes the 40h WM training from "wasted" to "trained on the wrong action distribution — swap the actor and the WM continues learning the right on-policy distribution for free." Hierarchical wrappers over uniform primitives cannot claim this.

6. **Repentogon TAS two-stage BC (bot corpus → human corpus, VPT-style)** — Steel R3 Idea 5. Payoff: **medium-high** (decouples BC feasibility from human recruitment; covers R3 corner). Cost: **~1 engineer-week to wire bot**. Novelty: **3/5** (Pro mentioned Repentogon in R2 as a bonus but did not integrate; Steel R3 formalised the two-stage pipeline). Why it matters: converts Opp's `D_high` cost concern into `D_low` via engineering rather than recruitment.

7. **Character-embedding-conditioned BC (12 demos not 34)** — Steel R3 Idea 4. Payoff: **medium**. Cost: **low**. Novelty: **3/5**. Why it matters: solves Agent 2's character gap at 1/3 the demo cost by trading breadth for parameter-shared conditioning.

---

## Open questions (must resolve before committing)

1. **Does `imag_horizon: 20→120` alone lift the actor off `entropy = 3.804` within ~2h of additional training on the existing checkpoint?** — Resolves whether horizon is `H_soft` (hyperparameter bug, favours BC-first) or `H_hard` (architectural attractor, favours Director-first). The 2–3 day experiment answers this. **This is the single most important unknown.**

2. **What is the realistic demo-supply cost curve for our project?** — Opp assumes $20–30/h commercial × 3 humans × 50h. Pro assumes $0 researcher self-play or Repentogon bot. Steelman assumes hobbyist $0. Resolves the `D`-axis. Would be resolved by: 1-week recruitment pilot (r/isaac, Isaac speedrun Discord) OR wiring the Repentogon bot and measuring corpus quality.

3. **What is Isaac's BC ceiling per unit of demo data, empirically?** — NetHack's Hambro Fig 4 is the closest reference class but Isaac has more per-frame decision structure (real-time dodge/aim) and less strategic depth. Resolves whether the "10M transitions → 6× cold-start" extrapolation holds. Would be resolved by: BC pilot on 5h of demos and measuring `rooms_visited` / `use_item` deltas.

4. **Does Director's 3–10× sample efficiency claim hold on Isaac's specific state distribution?** — Opp's strongest architectural claim; Pro's "200M env-steps" counter is disputed as category-error. Would be resolved by: implement Director wrapper (~2 wk) and measure sample efficiency on a truncated task (Basement I clear).

5. **Is the codebase schema actually stable post-audit-rehab, or will further churn strand any BC corpus?** — Base rate <30%; audit itself recommends significant changes. Would be resolved by: freeze schema for 4 weeks after obs+action rehab, then record demos.

---

## Recommended next action

**Do not commit to either BC-bootstrap or Hierarchical-actor yet. Ship the shared-prerequisite obs+action rehab (weeks 1–2) in parallel with the 2–3 day `imag_horizon: 20→120` gating experiment on the existing checkpoint, and let the empirical result select the branch — BC + AWAC anchor if entropy drops (`H_soft`), Director wrapper + BC-warm worker if it does not (`H_hard`). Both branches terminate at the same composed architecture, so the ordering decision is worth resolving with data rather than debate.**
