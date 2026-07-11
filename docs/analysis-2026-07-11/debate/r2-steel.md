# Steelman — Round 2

## Executive shift from R1 → R2

R1's structure (**R0 obs/action rehab → R1 BC → R2 Hierarchical → R3 compose**) survives contact with both openings, but two arguments force sharpening:

1. **Opponent Objection 1 (horizon-as-fixed-point-attractor) is load-bearing and I under-priced it.** BC alone is a *one-shot* policy improvement; if `imag_horizon`, symlog value clipping (±20), and `cont≈1` are not co-repaired, the RL fine-tune phase will wash out the BC prior within ~10–20k gradient steps. My R1 treated the horizon fix as a "post-plateau R2 concern" — that was wrong. **The horizon fix is a co-requisite of BC, not a successor to it.** This creates a new regime **R1′ = obs+action rehab + BC + horizon repairs (imag_horizon, value scale, cont), applied together.** In R1′, BC wins; in bare R1 (BC without horizon fix), BC's advantage decays.

2. **Proponent Claim 4 (in-mod native-obs demo recording via seed-determinism) partially defuses Opponent Objection 3 (Ross-Bagnell covariate shift).** If the human plays *inside the RL bridge at the mod's observation resolution* — optionally with a training-HUD that renders only the obs vector — then `d_H(π_learned, π_expert)` collapses from Ω(1) to something bounded by the surjectivity gap of the *revealed* obs schema. This is not a full defuse (humans still use minimap intuition even when the HUD is stripped), but it converts Opponent's Objection 3 from [VALID] to [CONDITIONAL on demo recording method].

Neither side moved me on the core structural claim: **the "BC vs Hierarchical" framing is a category error.** Both openings, when you strip the advocacy, agree on the audit's own priority stack (obs first, BC second, options third). Their disagreement is about *ordering under compute pressure*, not about *inclusion*.

---

## Classification of Proponent claims

- **[VALID]** *Precedent is unanimous that procedural/long-horizon games at consumer-GPU compute require a demo prior.* NetHack (Hambro 2022, arXiv:2211.00539), AlphaStar (Vinyals 2019 Nature), OpenAI Five (arXiv:1912.06680), DQfD (Hester 2018) — five references, zero cold-start counter-examples at our compute tier. Opponent does not contest this, only its sufficiency.
- **[VALID]** *The 40h run is empirical proof cold-start Dreamer cannot escape uniform-random.* `actor_entropy = 3.804 / 3.807` is a mechanical measurement. Opponent concedes this in his own steelman ("the 40h Dreamer run is provably learning nothing").
- **[CONDITIONAL — true only after horizon repairs]** *BC's CE loss is un-hackable.* True for the BC pretraining phase (mechanical property of supervised loss). **False for the subsequent RL fine-tune** if horizon remains broken — the RL phase re-imports the exact v0→v3 hack cycle. Sharpening: the un-hackability claim covers the pretraining window only, not the fine-tune equilibrium. Opponent's Objection 1 lands here.
- **[CONDITIONAL — true only for in-mod recording]** *Isaac's data economics uniquely favor BC.* True if we record through `obs.lua` at the mod's 15 Hz control rate. False if we try to OCR YouTube VODs — those give us action distributions but not obs-conditioned actions, and Opponent's covariate-shift argument then binds fully. Northernlion's 10k+ runs are *supply*, not directly usable as `(obs, action)` pairs; the seed-replay claim is the actual killer feature, and it needs one human to re-play seeds inside the mod.
- **[VALID]** *Hierarchical actor's option termination predicates require obs the current schema lacks (door target-room type covers 3/15).* Both sides agree on this diagnosis; it is a mechanical property of the code. This is exactly why R0 (obs rehab) is prerequisite for both approaches — a point Opponent implicitly concedes when acknowledging obs schema stability is <30%.

## Classification of Opponent claims

- **[VALID]** *Horizon mismatch is the root cause of the value-inflation loop and BC does not touch `imag_horizon`.* The 1270:1 target-to-advantage ratio (`actor_target_mean=47` vs `actor_adv_abs_mean=0.037`) is a horizon-truncation artifact, not a bad prior. This is the strongest single claim in either opening. It moves me toward **R1′** — BC must be shipped together with horizon repairs, not as a standalone.
- **[CONDITIONAL — true for hand-designed options only]** *Options collapse the horizon 200× and this dominates a policy-prior fix.* True structurally. False *as a next investment* because the intra-option controllers still face the cold-start problem Proponent Claim 2 documented. An option-critic architecture with uniform-random intra-option policies will produce option-level returns dominated by primitive noise — the credit-assignment problem *relocates*, does not collapse. Opponent's counter-move (option-critic can learn intra-option policies) is true in principle, but re-hits the compute wall that produced the 40h uniform run.
- **[CONDITIONAL — true for commercial-team pricing, false for hobbyist/researcher pricing]** *Demo labor cost $3–18k under-priced by 1 OOM.* True if we hire external skilled players at $20/h. **False** if the researcher-of-record plays 20–40h through the mod (0 marginal cost) or recruits 1–2 volunteers via the community. Repentance has an active speedrun/mod community; 20h of hobbyist-recorded demo through our schema is realistic at $0 cash outlay. Opponent's cost estimate assumes an economic model that doesn't match our project structure.
- **[CONDITIONAL — true for YouTube-scraped BC, false for in-mod native BC]** *Human obs ≠ our obs → BC converges to marginalised-over-hidden-state noise.* Ross-Bagnell compounding-error is real (`T²=3.24×10⁶`), but the bound applies to the surjectivity gap between what the *human* saw and what the *agent* sees. If humans play through the mod at obs-resolution (Proponent Claim 4's seed-determinism move), the gap is bounded by whatever residual features remain unshared. Opponent has a real point that humans still use *implicit* minimap intuition even when the HUD is stripped — this is a partial residual, not a categorical block.
- **[VALID]** *Stripped action space MultiDiscrete([9,5]) cannot represent space-bar/bomb/card actions; BC on this space is provably broken.* This is exactly R0. Both sides agree — Proponent lists it as a shaping-retirement side-effect, Opponent lists it as an anti-BC objection, but the mechanical fix is the same: restore the action head before recording demos.
- **[CONDITIONAL — true if BC is the terminal design; false if BC is bootstrap for hierarchical]** *NetHack's 3B transitions produced only top-15% BC and our budget is 3-4 OOM below.* Load-bearing against **BC-only** as a solution, not against **BC-as-bootstrap for hierarchical fine-tune**. AlphaStar Unplugged (arXiv:2308.03526) showed offline-RL from the *same* demo corpus recovers most of RL-trained performance — but Vinyals 2019 explicitly required hierarchy+league to *exceed* the SL prior. Reference class supports "BC gets you to top-15% then you need something else"; that something else is R2/R3 (hierarchical + option discovery).
- **[FALSE]** *"The Bayesian dominance is one-sided" — hierarchical is strictly cheaper marginal than hierarchical+obs+BC.* This is wrong on two counts. First, hierarchical *also* requires the obs expansion (both sides agree door-target-type must cover 15 room types; both sides agree active-charge readiness is needed for `use_active_when_ready` options). Second, obs-expansion is a strict prerequisite of hierarchical too, so it doesn't discount. The marginal-cost comparison is BC-effort vs option-definition-effort *given* the obs+action fix is already paid — and that comparison is genuinely close (Opponent's own estimate: 1-2 engineer-weeks for options; Proponent's estimate: 1-2 weeks for BC pipeline + 20-40h demo recording).
- **[CONDITIONAL]** *BC ceiling caps late-floor performance below Mother/Delirium.* True when BC is terminal (agrees with R4). False when BC is bootstrap for RL fine-tune + hierarchical — the ceiling is imposed only by the *terminal* training method, not the *bootstrap* method.
- **[VALID]** *Hierarchical simplifies reward via option-termination boundaries aligning with natural game events (Sutton/Precup/Singh SMDP).* Genuine synergy claim. But this is a reason to *include hierarchical eventually*, not a reason to skip BC. Compatible with R3.

---

## Unified model (sharpened)

Five regimes, four axis-values, one modal path.

### Regime R0 — TODAY. **Neither wins; obs+action rehab is the correct next investment.**
Both sides converge here mechanically. Action head lacks space-bar/bomb/card; obs schema lacks active-charge, transformation counters, door-target enum beyond 3/15, minimap. Any policy — BC-cloned, hand-coded, or hierarchical — that must condition on these signals is capacity-limited by the obs. Effort estimate: 1–2 engineer-weeks. **Confidence: High** (mechanical, both sides agree).

### Regime R1′ — POST-OBS + HORIZON CO-REPAIR (weeks 3–8). **BC-bootstrap wins.**
Preconditions: (a) obs+action rehab shipped; (b) **horizon repairs shipped together with BC**: `imag_horizon` raised from 20 → ≥120 ticks (8s → covers a room clear), value-head symlog scale raised past `r_beat_mom = +50`, `gamma` re-examined given new horizon; (c) ≥20h in-mod demo recorded in the new schema.

This is the update Opponent's Objection 1 forced. Under R1′, BC has three roles:
- **Warm-start the actor** — takes the policy off the 3.804-uniform basin in one supervised pass.
- **Serve as a *behavioral regularizer* during RL fine-tune** — KL(π‖π_BC) in the actor loss prevents the horizon-fixed-point attractor from washing out the prior. This is DQfD-style (Hester 2018) or CQL-style regularization, standard in offline-to-online RL.
- **Retire the shaping treadmill for the pretraining phase** — CE loss is un-hackable; sparse reward becomes viable during fine-tune.

Why not hierarchical here: an option-level RL over uniform-random primitive controllers re-inherits cold-start pathology at option granularity. Options need a competent primitive layer to point at.

**Confidence: High** on BC-plus-horizon-repair dominance; **Medium** on the exact weeks.

### Regime R2 — POST-BC PLATEAU (weeks 8+). **Hierarchical wins.**
Sharpened boundary: **plateau reached at basement/caves clears, not at Mom/Womb** (NetHack top-15% analogue per Hambro 2022). Opponent Objection 6 correctly bounds this — BC + flat RL will not reach the game's terminal objectives via demonstration prior alone. Once the flat BC-fine-tuned actor stops improving on `floors_reached ≥ 3`, the temporal-horizon problem re-binds at *multi-floor* scale (2–4 minute macro-actions), which even a bumped `imag_horizon=120` cannot cover.

At this point options become dominant. Termination predicates now instantiable because R0 shipped the missing obs fields. **Confidence: Medium-High.**

### Regime R3 — COMPOSE (dominant strategy once demos ≥50h). **BC-mined option discovery.**
Proponent Claim 5 (BC as prerequisite for hierarchical anyway) and my R1 R3 converge. The specific mechanism sharpened by both openings:
- Cluster demonstration trajectories on behavior segments → options.
- Intra-option controller = BC actor conditioned on option-label.
- Option-level policy = BC-trained on option-labels, then RL fine-tune.
- Option termination boundaries defuse the value-inflation loop because option returns terminate at real events, not `cont≈1` forever (Opponent Objection 1's structural fix).

This is Fox et al. 2017 / Krishnan et al. 2017 option discovery from demonstrations. **Confidence: Medium** for demo-mined-over-hand-designed on Isaac specifically.

### Regime R4 — CORNER: no demos available. **Hierarchical by default.**
If demo recruitment fails (<10h, no skilled player available), BC ceiling drops below cold-start floor. Given Isaac's community and Proponent's seed-replay argument, probability of falling into R4: **Low**, but should be verified in week 1 before committing to R1′.

---

## Where each side moved me

**Proponent moved me on:**
1. The un-hackability of CE loss for the pretraining phase is stronger than my R1 acknowledged. Retires ~15 shaping terms mechanically. Adopted into R1′.
2. WM losses are converging (`enemies_mask 11.8 → 0.006`, `doors 9.9 → 0.006`). Only the actor is broken. BC replaces just the actor. This is an **argument for preserving the Dreamer stack, not throwing it away** — my R1 was silent on this and it's important for the R1′ → R2 transition (imagination-based fine-tune over BC actor is exactly the AlphaStar recipe).
3. Seed-deterministic in-mod recording partially defuses Ross-Bagnell covariate shift. Adopted as [CONDITIONAL] boundary on Opponent Objection 3.

**Proponent failed to move me on:**
1. The claim that BC alone solves the failure. Opponent's horizon-fixed-point argument is real. BC needs horizon co-repairs, not just precedent-appeal.
2. The reference-class strength claim. NetHack's 3B-transition-to-top-15% result bounds where BC lands; it doesn't fail the whole strategy but it does fix the ceiling.

**Opponent moved me on:**
1. Horizon-as-fixed-point-attractor. BC's advantage decays without horizon co-repairs. Created R1′ regime.
2. NetHack quantitative precedent (3B → top-15%). Sharpens R2 boundary — plateau kicks in earlier than "at Mom kill." R2 begins around basement/caves clears.
3. Reward-alignment via option termination boundaries (SMDP). Real synergy with Agents 1/2's reward gaps. Adopted into R3 mechanism.

**Opponent failed to move me on:**
1. Labor cost. $3-18k is commercial-team pricing; our project pays 0 marginal cost for researcher self-play + hobbyist volunteers. Downgraded to [CONDITIONAL].
2. Covariate-shift-kills-BC. Partially defused by in-mod recording. Downgraded to [CONDITIONAL].
3. "Hierarchical is strictly cheaper marginal." False — hierarchical shares R0's obs prerequisites, so it doesn't dominate on cost.
4. BC ceiling as terminal argument. Only binds if BC is terminal, not as bootstrap. Downgraded.

---

## Where evidence is genuinely uncertain (both sides overclaim)

- **Wall-clock ratios.** Proponent's "4h train + 1-2 weeks pipeline" and Opponent's "1-2 engineer-weeks for options" are both plausible and neither is measured on our specific stack. Uncertainty: ±2× either direction.
- **BC ceiling on Isaac specifically.** NetHack's 3B-to-top-15% is the closest reference class but Isaac has more per-frame decision structure (dodge, aim) and less strategic depth than NetHack. Ceiling could be higher or lower than the NetHack extrapolation.
- **Whether horizon repair alone (imag_horizon 20→120) is sufficient or whether hierarchical is required.** Genuinely open. Cheap to test: bump imag_horizon in a controlled experiment before committing to options.
- **Demo-mined options vs hand-designed options on Isaac.** Mixed empirical record in the literature; Isaac's item-and-transformation combinatorics may make hand-designed options underspecified.

---

## Preview of R3 novel synthesis (headline only)

Five candidate hybrids that neither pure side proposed:
1. **BC + horizon repair as a bundled co-requisite (R1′)** — the single highest-value change.
2. **KL-regularized RL fine-tune with BC as behavior prior** — prevents fine-tune from washing out BC weights (Nair 2020 AWAC, Peng 2019 AWR).
3. **Demo-mined options with BC intra-option controllers** — R3, but with the specific mechanism: cluster on room-clear boundaries because obs already emits `enemies_alive` and `room_cleared`.
4. **Character-conditioned BC via one demo per character** — combines Proponent Claim + Agent 2's character gap into a single 12-demo dataset design.
5. **Dreamer WM preservation + BC actor swap** — the WM is converging; only replace the actor, keep the imagination loop. Turns the "40h run wasted" narrative into "40h run trained the WM, now warm-start the actor."

All five to be developed in Round 3.

---

## Confidence summary

| Claim | Confidence | Basis |
|---|---|---|
| R0 dominates today (neither BC nor Hierarchical) | High | Mechanical; both sides converge |
| R1′ (BC + horizon co-repair) dominates weeks 3-8 | High | Precedent + fixed-point argument accepted |
| R2 boundary at basement/caves clears (not Mom) | Medium-High | NetHack extrapolation |
| R3 compose strategy dominates given ≥50h demos | Medium | Mixed empirical record on demo-mined options |
| R4 corner-case unlikely | Medium | Depends on unmeasured demo supply |
| BC alone (without horizon repair) is fragile | High | Opponent's Objection 1 accepted |
| BC labor cost estimate at $3-18k | Low (context-dependent) | True for commercial, false for hobbyist |

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r2-steel.md]
