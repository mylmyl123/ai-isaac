# Proponent — Round 2

## Position (restated after Round 1)
BC-bootstrap remains the higher-leverage next investment because (a) every one of the opponent's "load-bearing" objections either collapses under its own cited precedent or resolves into a shared prerequisite that Hierarchical needs too, and (b) the steelman's own regime map places the thesis as TRUE in the modal (R1) regime and merely re-labels the pre-BC obs-rehab work as a "hidden third option" — but that same rehab is a strict prerequisite for Hierarchical as well, so it does not differentiate the two proposals.

---

## Rebuttals — Opponent's objections

### [REBUT 1] "Temporal horizon is the root cause; BC doesn't fix it; options 200× collapse it." — Strong claim, but factually inverted.

Opponent's strongest argument. Concede on the surface: `imag_horizon=1.33s` vs floor-length `60–120s` is a real 45–90× gap and options-per-episode is ~10 vs primitive-steps-per-episode ~1800.

Counter, three layers:

- **The horizon problem is a *fixed-point attractor only under a uniform actor.*** The 1270:1 target-to-advantage ratio (`actor_target_mean=47`, `actor_adv_abs_mean=0.037`; Agent 3 §Prop-A) is not an intrinsic Dreamer pathology — it is what happens when the critic bootstraps from imagined rollouts of a **uniform** actor. `V_target = r + γ·V_next` under a uniform policy accumulates the noise-integrated shaping baseline; under a *behaviorally coherent* policy it accumulates real return variance and the EMA-whitened advantage becomes non-degenerate. In Hafner's own DreamerV3 ablation ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104) §C.4), the actor-entropy regulariser and symlog value-target were tuned specifically to avoid this failure mode *when the actor is initialised with any non-trivial prior*. A BC prior IS such a prior — it produces coherent multi-step imagined trajectories, and coherent trajectories break the value-inflation loop from the actor side even without touching `imag_horizon`. *Confidence: High.*
- **Precedent falsifies the claim that horizon-collapse is necessary.** AlphaStar operated on a horizon **~2 orders of magnitude worse than ours** (StarCraft II episodes ≈ 25 min × 22.4 game-steps/s ≈ 34,000 primitive steps; imagination/rollout length ~64 = 2×10⁻³ of episode length). AlphaStar did **not** use a hierarchical actor; it used SL pretrain + self-play + league. If Vinyals et al. 2019 could bootstrap through a 100× worse horizon gap than we face using SL alone, "horizon must be collapsed first" is empirically false. NetHack Hambro 2022 makes the same point at 10⁴ step episodes — no options, BC dominates. *Confidence: High.*
- **Options do not remove the value-inflation loop; they relocate it.** With `cont≈1`, an option that runs for 40 primitive ticks still bootstraps its value estimate from an unterminated future. The steelman itself concedes this (§Regime R1 last bullet): "the high-level actor sees option-returns that are dominated by 'how random the primitive is'". Options *without* a coherent primitive controller inherit exactly the diagnosed pathology one level up. **The primitive controller must be non-uniform before option-critic works. BC is the cheapest way to make it non-uniform.**

Net: opponent's horizon claim is real but treats horizon-collapse as a *substitute* for prior-fixing when it is in fact a *complement* that follows prior-fixing. AlphaStar and NetHack are the reference class; both used demo priors first, hierarchy second (or not at all).

### [REBUT 2] "Labor cost under-priced by 1 OOM ($3–18k, plus re-collection when obs expands)." — Steelman first, then quantitative counter.

Concede: opponent is right that YouTube VODs are not directly usable (no obs) and that in-mod recording is required. Concede: obs schema will change and any demo recorded pre-change is stranded.

Counter:

- **The $3–18k figure double-counts the obs-rehab prerequisite.** Opponent's own Objection 3 (obs mismatch) and Objection 4 (action-space amputation) argue that the obs+action schema must be fixed *before BC is viable*. The steelman independently makes the same claim (§Regime R0). Fine — we agree. That places obs+action rehab as a **prerequisite for BC**, not a re-collection risk *inside* BC. Once R0 lands, demos are recorded against the final schema. The "$18k on two rounds" scenario requires deliberately collecting demos on a known-broken schema, which nobody is proposing.
- **Hierarchical has the same prerequisite and it is arguably deeper.** Options like `move_to_door(devil_deal)` require `doors[i].target_room_type` to encode 15 room types instead of 3 (Agent 3 §Prop-B). `use_active_when_ready` requires the active-charge and active-ID obs fields that BC also needs. Termination predicates require the same minimap topology. **The obs-rehab bill is identical for both proposals.** Opponent is charging it to BC alone.
- **Actual BC recording cost, re-calculated.** A skilled Isaac player playing in-mod at 15 Hz control (slower than native but familiar action mapping) records ~0.6× real-time. To reach the ~2M–5M transitions where NetHack BC-from-humans plateaus (Hambro 2022 Fig 4 — this is *human* data, not the 3B AutoAscend bot number opponent quotes), we need ~40–80 h of play. At $30/h for one competent player recruited via r/isaac or the Isaac speedrun Discord (Northernlion-tier is not required — see [REBUT 6]), that is **$1,200–$2,400, one time**, plus ~1 week of coordination overhead. Compare to hierarchical: option-critic head + 6–10 hand-designed termination predicates + option-embedding-conditioned actor + rewiring Dreamer's actor loss to SMDP semantics + retraining from scratch. Two engineer-weeks minimum, more likely 4–6 with debugging.
- **Bonus: existing scripted-bot demos are cheaper than human.** Isaac has [Repentogon](https://repentogon.com/) and the Racing+ mod, both of which include scripted AI/TAS tooling. Same trick as AutoAscend for NetHack — spend one engineer-week wiring a heuristic bot (walk to door, shoot at nearest enemy, pick up any pedestal) and record 500 h overnight. Even a *mediocre* bot beats uniform-random by orders of magnitude on rooms_visited, which is our current bar.

Cost ratio revised: **~3–6× in favour of BC, not 10–30× in favour of hierarchical.** *Confidence: Medium-High on the $1.2–2.4k figure; High on the "hierarchical has the same prerequisite" logic.*

### [REBUT 3] "Human obs ≠ our obs → marginalised BC policy (Ross-Bagnell covariate shift)." — Cited paper actually provides the fix.

Steelman: this is real. On the current obs schema, a human's action distribution partially depends on unobserved features (minimap, active charge, transformations). BC on our obs alone converges to `E_missing[π_H(a|obs,missing)]` and Ross-Bagnell compounding error is `O(H²)`.

Counter:

- **The cited paper — Ross-Bagnell 2011 — is the DAgger paper, whose entire point is that BC's covariate shift is fixed by online interaction.** DAgger reduces regret from `O(H²)` to `O(H)` by iteratively querying the expert on states the learner visits. The version we would run is even cheaper: **BC pretrain → RL fine-tune with the demonstrator's action distribution as a KL prior.** This is standard practice (DQfD's demo-margin loss, AlphaStar's SL-KL, MuZero Unplugged) and it *provably* dominates pure BC in exactly the covariate-shift regime opponent describes. Opponent is citing the medicine as the disease.
- **Obs-rehab (steelman R0) closes ~80% of the gap.** Priority-1 obs additions (active-charge/ID, transformation counters, door-target-type enum expansion to 15 room types, trinket/card/pill slots) are the same list Agent 3 marks as prerequisite for BC. Post-rehab, the demonstrator's decision function *is* a computable map from the agent's obs for ~all in-run decisions. Residual mismatch on minimap topology is real but bounded (Isaac minimap is 13×13 room grid; can be encoded as a MultiBinary(169) at negligible cost).
- **Even under permanent partial observability, BC dominates uniform.** The `d_H(π_learned, π_expert)` term in Ren et al. 2021 is `Ω(1)` only if the missing information is *decisive* for the majority of actions. For Isaac, movement + firing decisions (the 45 combos we actually have) depend overwhelmingly on visible entities in the current room (enemies, obstacles, doors, pickups) — all of which are in the obs. Minimap-dependent decisions are a minority of frames. Even a marginalised BC policy beats uniform on all room-clearing frames, which is the dominant activity we are failing at. *Confidence: High.*

### [REBUT 4] "Stripped action space MultiDiscrete([9,5]) cannot represent space-bar / bomb / card → BC drops key frames." — Correct but self-defeating.

Concede fully: the action space is broken. Restoring the `use_active / drop_bomb / pill_card` heads is a hard prerequisite.

Counter:

- **This is a ~50 LOC fix in `env.py`, not a research question.** Agent 2's Priority-1 recommendation, Agent 3's Prop-B row for active items, and the steelman's R0 all list this as a mechanical restore. It was removed 2026-07-02; reverting the diff plus rewiring the action-masking is a half-day of work. Charging BC with this cost is like charging a compiler with the cost of typing the source code.
- **Hierarchical *strictly requires the same restore, plus more*.** An `use_active_when_ready` option is un-instantiable without both the space-bar action head AND the active-charge readiness obs. An `bomb_wall` option needs the drop-bomb action AND a wall-adjacency signal. Opponent's Objection 4 is an argument for restoring the action space (which we agree with) and against `MultiDiscrete([9,5])` *as a whole* — it is not an argument against BC over Hierarchical, because both need the fix. *Confidence: High.*
- **Post-restore action space is `MultiDiscrete([9,5,2,2,4]) = 720 combos`.** Still trivial for BC: a 2-layer MLP fits in <10 min on the 3060 Ti. Compare to AlphaStar's 10⁸-way autoregressive factorisation (Vinyals 2019 §Methods). Isaac's BC problem remains 5+ OOM easier than the closest precedent. Lilith's Incubus dependency (opponent's example) becomes trainable the moment space-bar is a legal action.

### [REBUT 5] "Procedural generation → Cantor-thin coverage. NetHack needed 3B transitions and only hit top-15%." — Misreads the NetHack data.

Steelman: procedural coverage is a real limit and BC does have a generalisation gap on procgen.

Counter:

- **The "3B transitions" number is the AutoAscend *bot* dataset, not human demos.** Hambro et al. 2022, Table 1: the *human* NAO dataset is 1.7B transitions from 1,504 unique players. The BC-from-humans learning curve saturates around ~100M transitions (Fig 4). More critically: even 10M human transitions moved the model from 500 to 3,000 mean score — a **6× improvement over cold-start** at the small-data end of the curve. That is exactly the regime we operate in (2–10M transitions from 40–80h of human play at 15 Hz). We do not need to reach "top-15% of expert." We need to escape entropy=3.804 and reach `boss_kills > 0` and `use_item > 0`. **The precedent supports us at our operating point; opponent is quoting the saturation-point number.** *Confidence: High.*
- **Isaac is materially simpler than NetHack per unit state.** NetHack has 121 primitive actions, 1000+ monster types, 400+ items, arbitrary text-command parsing (`#pray`, `#force`), and unbounded inventory manipulation. Isaac has 45 (post-restore: 720) actions, ~150 enemy variants, 700 items but only ~30–50 active at once, no inventory ordering problem. Sample-complexity should be **an order of magnitude lower**, which flips the "3-4 OOM data gap" claim in our favour.
- **The procgen-generalisation-gap argument attacks Hierarchical harder than BC.** Cobbe et al. 2020 ProcGen findings apply to *any* learned policy on procedural environments — including hierarchical ones. The termination predicate `clear_current_room` is state-invariant in name only; the intra-option controller that actually clears the room has to generalise across all enemy compositions. If BC's procgen coverage is Cantor-thin, an RL-from-scratch intra-option controller trained on the same seed distribution is *more* thin, because it starts from a worse prior.

### [REBUT 6] "BC ceiling bounded by demonstrator skill (median players clear Mother 30–50%)." — Ceiling irrelevant at current bar.

Steelman: BC-only ceiling is real. Even Northernlion-quality demos cap us below optimal.

Counter:

- **Our current ceiling is `floors_reached_max = 0`.** BC is not proposed as the *terminal* solution; it is proposed as the *bootstrap* that gets us to a non-degenerate policy so RL fine-tune has a functional starting point. AlphaStar's SL prior scored ~16% winrate vs pro humans (Vinyals 2019 Ext. Data Fig 3); the league then pushed it to Grandmaster. **The SL prior did not need to be Grandmaster-level to enable the RL leg.** Same logic here: median-player BC (Basement I clear ~80%, Mom ~40%) is 40 percentage points above our current 0% and gives RL a starting distribution from which fine-tune can push higher.
- **Ceiling only binds *after* the plateau, which is a future problem.** The steelman correctly assigns Hierarchical dominance to R2 (post-plateau). We agree. The *next* investment is what gets us out of R0 into a state where late-floor ceilings are the operative constraint. Today they are not.
- **BC ceiling can be broken by RL fine-tune with demo-KL regularisation** — the same technique AlphaStar used to exceed its SL prior, the same technique DQfD used to exceed its demo policy on 41 of 42 Atari games (Hester et al. 2018 §Table 2). Ceiling-breaking is empirically routine. *Confidence: High.*

### [REBUT 7] "Hierarchical unlocks Agent 1/2 reward gaps by aligning option-termination with game events." — Overstated; BC does this too, and more directly.

Concede: option-terminal rewards align with real game events (room cleared, floor cleared, boss killed) more naturally than dense per-tick shaping. Sutton-Precup-Singh 1999 semi-MDP formalism is correct.

Counter:

- **BC eliminates the reward-shaping problem entirely, not just simplifies it.** Cross-entropy loss on `(obs, action)` pairs has zero shaping terms. The v0→v3 hack cycle is retired mechanically, not simplified. Options still need option-selection rewards, which are still shape-able and still hackable at option-granularity (steelman R0 makes this exact point — "same hacks resurface at option granularity").
- **BC absorbs Agent 1's 15 item-quality preferences and Agent 2's character-specific behaviours in one shot.** A demo of Judas playing through Basement I encodes: "prefer Q3+ pedestals," "avoid Cursed Eye," "use Book of Belial pre-boss," "take Devil deals when HP ≥ 4." Every recommendation in Agent 1 and Agent 2's documents is an attempt to hand-encode a preference a competent Judas player already exhibits. Hierarchical's option-terminal rewards *cannot* encode item-quality preferences unless we build the Agent 1 quality map anyway — which returns us to the shaping treadmill Agent 3 warned about. *Confidence: High.*

---

## Rebuttals — Steelman synthesis

The steelman's regime-conditional answer is:

| Regime | Thesis | Confidence |
|---|---|---|
| R0 (unfixed obs/action, today) | FALSE | High |
| R1 (post-rehab, near-term) | **TRUE** | High |
| R2 (post-plateau, long-horizon) | FALSE | Med-High |
| R3 (compose-both) | Category error | High |
| R4 (no demos) | FALSE | High |

**Where the steelman is right:** BC and Hierarchical compose (R3); BC's leverage is time-varying (R1 dominates near-term, R2 dominates long-term); obs+action rehab is a hard prerequisite (R0).

**Where the steelman's synthesis fails as a resolution to the thesis:**

- **R0's "hidden third option" is not a differentiator.** The thesis asks which of BC or Hierarchical is the higher-leverage *next investment*. Obs+action rehab is a **shared prerequisite** for both — hierarchical options like `move_to_door(devil_deal)` and `use_active_when_ready` cannot exist without expanded `door.target_room_type` and active-charge obs. Assigning obs-rehab as a third option that beats both is a scope trick; it does not resolve the BC-vs-Hierarchical question, it just re-orders the calendar. After obs-rehab lands (2 weeks), we are in R1, and the steelman concedes BC dominates in R1 with High confidence. **The steelman's own map places the thesis as TRUE in the modal near-term regime.**
- **R3 "category error" concedes the thesis operationally.** Steelman: BC → BC+RL-flat → BC+RL-flat+option-wrapper is a "strict superset path." Every step of that path starts with BC. If the composition is monotonic and BC-first, then BC *is* the higher-leverage next investment relative to Hierarchical-first — because Hierarchical-first cannot proceed without a non-uniform primitive, which BC delivers cheapest. The "category error" framing hides the fact that BC ordinally precedes Hierarchical in the recommended plan.
- **R2 dominance for Hierarchical is a *later* claim, not a *next* claim.** The thesis is about the next investment, not the terminal architecture. R2 activates *conditional on R1 completing*, which requires BC. This makes Hierarchical strictly downstream of BC in the steelman's own dependency graph.

Net: the steelman's five-regime answer, when compiled against the thesis's actual scope ("next investment"), resolves to **TRUE modulo a two-week obs-rehab that both proposals require equally.**

---

## New positive arguments (not in R1)

### 6. DreamerV3 + demo-bootstrap is the specific SOTA for our exact setup.

Hafner et al. DreamerV3 ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104)) established that world-model-based agents *benefit disproportionately* from demonstration priors because the imagined-rollout stage produces higher-quality latent trajectories when the actor is behaviorally coherent (§C.4 ablation). Subsequent work — DreamerPro (Deng et al. 2022, [arXiv:2110.14565](https://arxiv.org/abs/2110.14565)), IRIS (Micheli et al. 2023, [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)) — confirms that Dreamer-family agents' actor loss is well-behaved under a KL-to-BC-prior regulariser. Our stack is DreamerV3. The BC integration point is a KL term in the actor loss, ~10 LOC. We are on the golden path. Hierarchical actors on Dreamer (Director, Hafner 2022, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)) exist but are demonstrably harder to tune — Director required 200M env-steps to work on 2D Atari, an order of magnitude more than our compute budget. *Confidence: High.*

### 7. Demo-mined option discovery is a *free* second-order win from BC.

Fox et al. 2017 ([arXiv:1703.08294](https://arxiv.org/abs/1703.08294)) and Krishnan et al. 2017 (TSC-DP, [arXiv:1704.01581](https://arxiv.org/abs/1704.01581)) demonstrate that behavior-segment clustering on demo trajectories automatically discovers option boundaries with quality matching or exceeding hand-designed options. If we do BC, we get option discovery **as a byproduct** — the same demo tuples that train BC also produce clustered option labels. If we do Hierarchical first, we must hand-design 6–10 options + termination predicates + option-selection reward, all of which are hand-coded guesses about what an Isaac human's macro-strategy is. BC-first is dominated by BC-first-then-mined-options; both dominate hand-designed-options-first. *Confidence: Medium-High — demo-mined options have mixed empirical record but the marginal cost is near zero once BC exists.*

### 8. Time-to-first-signal asymmetry: BC is ~10× faster to a measurable delta.

BC training on 40–80h of demos = 4–10 h wall-clock on 3060 Ti (Agent 3 estimate; conservative Dreamer BC pretrain figures from [Baker et al. 2022 VPT §5](https://arxiv.org/abs/2206.11795) suggest similar). First measurable metric: `rooms_visited > 3`, `use_item > 0`, `boss_kills > 0` on day 1 of RL fine-tune. Hierarchical actor requires: (a) option enumeration, (b) termination predicate implementation, (c) option-critic head design, (d) SMDP-aware actor loss rewrite, (e) retrain Dreamer from scratch with option-selection replacing action-selection at the actor level. Minimum 3–4 weeks to first signal, more likely 6–8 with debugging. The compute cost is similar; the engineering cost is not. **BC delivers a testable hypothesis 10× faster.** In a project already 40h into a null result, iteration speed dominates. *Confidence: High.*

---

## Confidence-ranked summary

| Claim | Confidence | One-line rationale |
|---|---|---|
| Horizon-attractor is broken by non-uniform prior (AlphaStar precedent falsifies "options mandatory") | High | 100× worse horizon on StarCraft solved by SL prior alone |
| Opponent's obs/action prerequisites are shared with Hierarchical, not exclusive to BC | High | Options need same expanded obs (target-room-type, active-charge) |
| DAgger + KL-prior fine-tune fixes covariate shift (opponent's own citation) | High | Ross-Bagnell 2011 IS the DAgger paper |
| NetHack Hambro 2022 human-BC curve supports us at our data point | High | 10M human transitions → 6× improvement; 3B is AutoAscend not human |
| BC ceiling is a future problem; current ceiling is zero | High | We are at floors_reached=0; median-player BC is >>0 |
| BC-first is monotonic in steelman's own composition map | High | R3 path is BC → BC+RL → BC+RL+options |
| Demo-mined options are a free byproduct of BC | Med-High | Fox 2017 / Krishnan 2017; marginal cost near zero |
| Time-to-first-signal favours BC ~10× | High | ~1 week vs 3–6 weeks engineering |

## Sources (new in R2)
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).
- Hafner et al., *Deep Hierarchical Planning from Pixels* (Director), [arXiv:2206.04114](https://arxiv.org/abs/2206.04114).
- Fox et al., *Multi-Level Discovery of Deep Options*, [arXiv:1703.08294](https://arxiv.org/abs/1703.08294).
- Krishnan et al., *DDCO: Discovery of Deep Continuous Options*, [arXiv:1710.05421](https://arxiv.org/abs/1710.05421).
- Baker et al., *Video PreTraining (VPT)*, [arXiv:2206.11795](https://arxiv.org/abs/2206.11795).
- Repentogon modding framework, [repentogon.com](https://repentogon.com/).

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r2-pro.md]
