# Proponent — Round 3 (Closing)

## Position
BC-bootstrap is the higher-leverage next investment. The debate has moved the *specification* of the plan (BC must ship with horizon co-repair, and the demo pipeline must be in-mod against a rehabbed obs/action schema) but has not moved the *ordering*: BC is monotonically first in every viable path — including every hybrid the opponent and steelman proposed.

---

## Synthesis — the three points that survive rebuttal

### 1. The reference class is unanimous, and opponent's counter-cites reinforce it.
On sparse-reward procedural long-horizon games at any compute tier, every published success used a demonstration prior before or instead of hierarchy. AlphaStar (Vinyals 2019, *Nature* 575:350) operated on horizons **~100× worse than Isaac** (~34,000 primitive steps per episode) with a rollout window of ~64 — a horizon ratio far more extreme than our 20/1800 — and solved it with SL pretrain plus league, **no hierarchical actor**. NetHack (Hambro 2022, [arXiv:2211.00539](https://arxiv.org/abs/2211.00539)) top submissions were BC-first, unanimously. DQfD (Hester 2018) beat DQN on 41/42 Atari games with <10 min of demo. Opponent's Hambro headline number (3B AutoAscend transitions → top-15%) is the *saturation* point of the human-BC curve; at our operating point of 5–10M transitions Fig. 4 shows a 6× improvement over cold-start — which is precisely the regime we need to leave `floors_reached_max=0`. There is no cold-start-plus-hierarchy counter-example at consumer-GPU compute in the entire literature. *Confidence: High.*

### 2. Every "hierarchical is cheaper" argument, when audited, requires BC anyway.
- Opponent's **Objection 4** (stripped action space): a shared prerequisite. Both proposals need `use_active / drop_bomb / pill_card` restored.
- Opponent's **Objection 5** (obs coverage of door target-room type, active charge, transformation counters): a shared prerequisite. `move_to_door(devil_deal)` and `use_active_when_ready` cannot fire without these obs fields either.
- Opponent's **novel Round-2 idea #13** ("hierarchical makes demos 50–100× cheaper because option-level macro-decisions"): the option-level policy is still BC. The opponent's own R3-composition path is `obs rehab → Option-Critic wrapper → option-level BC`. That is *BC-second, not BC-never*. It also silently assumes an intra-option controller that can execute "clear this room" — which is exactly what our 40h cold-start Dreamer cannot do, and which BC on primitive actions is the cheapest way to install.
- Steelman's **R3 composition path**: `BC → BC+RL-flat → BC+RL-flat + option-wrapper`. Monotonic superset. Every step starts with BC.

Across the opponent's, the steelman's, and my own composition paths, **the earliest common ancestor of every viable plan is a BC pretrain**. That is definitionally the higher-leverage next investment. *Confidence: High.*

### 3. Time-to-first-signal and stack preservation dominate under project reality.
The 40h Dreamer run's world-model losses are converging — `enemies_mask 11.8 → 0.006`, `doors 9.9 → 0.006` (Agent 3 audit). Only the actor is broken. **BC is a targeted actor-swap that preserves the entire trained WM.** Wall-clock cost: 1 week for pipeline + demo record + train, per DreamerV3 + BC-prior published precedent ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104)) and VPT ([Baker 2022, arXiv:2206.11795](https://arxiv.org/abs/2206.11795)). Hierarchical (Director, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)) requires: SMDP-aware actor-loss rewrite, manager-worker split, option-critic head, retrain WM under new imagination semantics — Director's own paper reports **200M env-steps** to converge on 2D Atari; at our 2.36 sps that is 24,000 hours of wall-clock, versus our 40h checkpoint. In a project already 40h into a null result, **iteration speed dominates leverage**. BC delivers a testable hypothesis 5–10× faster and preserves the existing WM investment. *Confidence: High.*

---

## Rebuttals — where opponent's R2 arguments landed and where they didn't

**Landed (steelman-adopted, plan updated):**
- Horizon co-repair (imag_horizon 20→120, value-scale past `r_beat_mom = +50`) must ship *with* BC, not after it. This is the R1′ update. It does not change the ordering — it changes the bundle. BC-plus-horizon-repair is still BC-first.

**Did not land:**
- **[REBUT §1, R2]** "1270:1 target-to-advantage ratio is a fixed-point attractor immune to prior fixes" — falsified by AlphaStar's 100×-worse horizon solved by SL alone, and by DreamerV3's own C.4 ablation showing coherent-actor rollouts break the value inflation loop from the actor side.
- **[REBUT §3, R2]** "Ross-Bagnell covariate shift kills BC" — Ross-Bagnell 2011 *is* the DAgger paper; its remedy (KL-regularised fine-tune, DQfD-style margin loss, AWAC/AWR) is the standard second stage of BC-bootstrap and provably restores `O(H)` regret.
- **[REBUT §8, R2]** "BC of Dreamer-imagined rollouts unsound at H>5 (Janner 2019)" — Janner's bound applies to *model-based rollouts trained on the BC-derived data distribution*, not to a BC loss computed on real (obs, action) pairs. The BC gradient never touches imagined states; it is an actor-space regulariser. This objection is category-confused.
- **[REBUT §9, R2]** "Schema stability <30% → 70% BC-corpus write-down" — mitigated by the shared prerequisite: the obs+action rehab is the schema change opponent fears. Once R0 lands (agreed by both sides as a next-step precondition), the corpus is recorded against the final schema. The 70% write-down assumes we deliberately record on a known-broken schema, which nobody proposes.
- **[REBUT §5, R2]** "Option-Critic learns β(s) end-to-end so obs gap is a non-issue" — true for β(s), but the *option-selector* still needs semantic obs to select `move_to_door(devil_deal)` vs `move_to_door(treasure)`. If the selector cannot distinguish room types via `door.target_room_type`, it cannot express the strategy that motivated hierarchy in the first place. Opponent has argued option-critic away from the failure mode by removing the strategic content it was supposed to enable.

---

## Novel ideas (surfaced during the debate)

### Novel Idea A — BC as an *imagination-distribution* shaper, not just a policy prior.
Neither side explicitly named this. The 40h WM has been trained on rollouts from a **uniform-random actor** (entropy 3.804 / 3.807). That means the WM is well-fit near states the uniform actor visits (early rooms, low HP, no items) and progressively worse-fit elsewhere. Every imagined rollout the actor loss consumes is a rollout in this narrow, uniform-actor-shaped manifold. A BC actor immediately shifts the on-policy rollout distribution to human-visited states (deeper floors, item pickups, boss rooms). **The WM then continues training on rollouts that are pre-selected for strategic relevance.** This is a compositional argument that hierarchical cannot claim: an option-critic wrapper over a uniform primitive still generates uniform-primitive rollouts inside each option, so the WM sees no distribution shift. BC uniquely repositions *both* the actor and the data distribution the WM continues to learn from. This turns "we wasted 40h" into "we trained the WM on the wrong distribution — swap the actor and the WM starts learning the right one for free."

Why it strengthens the case: it moves BC from "one-shot prior injection" to "a persistent data-distribution intervention that keeps compounding across every subsequent imagination step" — which is the specific property the Dreamer stack rewards most.

### Novel Idea B — A one-day falsification experiment that resolves the whole debate before committing.
Opponent's central load-bearing claim is that horizon truncation, not policy prior, is the root cause. This is directly testable at near-zero cost: **on the existing 40h checkpoint, bump `imag_horizon` 20 → 120, raise the symlog value-head range past `r_beat_mom = 50`, and set `cont` to terminate on episode end. Run 5–10k additional gradient steps.** If opponent is right, actor_entropy should drop off 3.804 as the corrected value targets propagate — no BC needed. If actor_entropy stays pinned near-uniform, opponent's horizon claim is empirically falsified and the prior-fix path (BC) is the confirmed lever. Cost: <24 hours of GPU. **In neither outcome is a 3–6-week hierarchical rewrite the correct next investment** — either horizon-repair-alone works (skip both), or BC is the confirmed lever (do BC). Hierarchical is not on the decision tree until at least one of these branches is executed.

Why it strengthens the case: it converts the debate from "which precedent is right?" to a Popperian experiment with an outcome that either fully vindicates the proponent or renders the whole comparison moot. Hierarchical never wins this branch.

---

## Closing summary

The opponent's strongest move — reframing the 40h stall as a horizon problem rather than a prior problem — sharpened the plan (BC must ship with horizon co-repair) but did not displace the ordering. The steelman's decomposition confirmed the thesis is TRUE in the modal near-term regime (R1′) and that every "compose-both" path is BC-first, not hierarchy-first.

Recommended next investment, integrating all three positions:

1. **Week 1** — obs+action rehab (Agent 3 §Prop-B, restore `use_active`/`bomb`/`card` heads). Shared prerequisite. Not contested.
2. **Week 1 (parallel)** — the falsification experiment in Novel Idea B on the current checkpoint. <24 GPU-hours.
3. **Weeks 2–3** — record 20–40h of in-mod human demos against the finalised schema. Cost: ~$1.2–2.4k or free via Repentogon scripted bot / researcher self-play.
4. **Week 3** — BC pretrain the actor (~4h wall-clock). Ship horizon co-repairs (imag_horizon 20→120, value scale, cont).
5. **Weeks 4–6** — RL fine-tune with KL-to-π_BC prior (DQfD / AWAC style). Measure plateau.
6. **Week 7+** — *only if* the fine-tuned flat actor plateaus below `floors_reached ≥ 3`, then wrap in hierarchical (Director-style or demo-mined options via Fox 2017 / Krishnan 2017 on the BC corpus we already collected).

Hierarchical stays on the roadmap. It just stays where the evidence puts it: after BC, not before.

## Confidence-ranked closing scoreboard

| Claim | Confidence | One-line rationale |
|---|---|---|
| BC is the earliest common ancestor of every viable composition path | High | Steelman R3, opponent R2 #13, and audit ordering agree |
| Reference class (AlphaStar, NetHack, DQfD) supports BC-first at consumer-GPU tier | High | Zero cold-start-plus-hierarchy counter-examples |
| BC preserves the converged WM; hierarchical requires retraining under new SMDP semantics | High | Direct from `enemies_mask 11.8 → 0.006` and Director's 200M-step figure |
| Time-to-first-signal ~5–10× in BC's favour | High | 1 week vs 3–6 weeks engineering critical path |
| Horizon co-repair is a co-requisite of BC (steelman-adopted) | High | Fair concession; does not change ordering |
| Novel: BC shifts imagination distribution — a persistent, compositional win | Medium-High | Compositional argument hierarchical cannot claim |
| Novel: 24-hour falsification test resolves the whole debate before commit | High | Cheap, decisive, unavoidably informative |

## Sources cited across debate
- Hambro et al. 2022, *Dungeons and Data*, [arXiv:2211.00539](https://arxiv.org/abs/2211.00539).
- Vinyals et al. 2019, *AlphaStar*, Nature 575:350. [nature.com/articles/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z).
- Mathieu et al. 2023, *AlphaStar Unplugged*, [arXiv:2308.03526](https://arxiv.org/abs/2308.03526).
- Hester et al. 2018, *DQfD*, [arXiv:1704.03732](https://arxiv.org/abs/1704.03732).
- Hafner et al. 2023, *DreamerV3*, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).
- Hafner et al. 2022, *Director*, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114).
- Baker et al. 2022, *VPT*, [arXiv:2206.11795](https://arxiv.org/abs/2206.11795).
- Fox et al. 2017, *Multi-Level Discovery of Deep Options*, [arXiv:1703.08294](https://arxiv.org/abs/1703.08294).
- Krishnan et al. 2017, *DDCO*, [arXiv:1710.05421](https://arxiv.org/abs/1710.05421).
- Nair et al. 2020, *AWAC*, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359).
- Ross, Gordon, Bagnell 2011, *DAgger*, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686).
- Internal audits: `agent1-item-economy.md`, `agent2-character-strategy.md`, `agent3-adversarial-audit.md`.

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r3-pro.md]
