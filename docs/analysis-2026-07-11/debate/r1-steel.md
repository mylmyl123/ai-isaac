# Steelman — Round 1

## Thesis under debate
"BC-bootstrap is a higher-leverage next investment for our Isaac RL project than Hierarchical-actor."

## Position (headline)
**Reject the binary. Neither BC nor Hierarchical is the correct *next* investment given the current codebase state.** The dominant regime today is one where **both** proposals fail on the same prerequisite: the obs and action space are amputated relative to what any demonstrator (human) or any option (macro-action) needs to condition on. Once that prerequisite is repaired, **BC dominates in the near-term (Weeks 1–4)** and **Hierarchical dominates in the long-term (post-plateau)** — and, critically, **they compose**: BC produces an intra-option controller for a Hierarchical actor, so the "vs" framing is false.

Strong hybrid position: **Obs+Action rehab → BC bootstrap → Hierarchical wrapper on the BC-warmed actor.** Confidence: High that this ordering strictly dominates either pure alternative under our compute budget; Medium on the exact wall-clock ratios.

---

## Decomposition — axes that flip the answer

**Axis A — Obs/Action completeness (binary: broken vs. sufficient).**
The current MultiDiscrete([9,5]) action head has **no space-bar action** (Agent 2 §"Action space has no active-item"; Agent 3 §Prop-B row "Active item charge"). `use_item` fired **0 times in 346k env-steps** (Agent 3 §Prop-C). Obs is missing active-charge, transformation counters, minimap, door-target-type enum, item stack counts, trinket/card/pill slots. This axis is the master switch: any policy — cloned, hierarchical, or hand-scripted — that must decide "press space now" cannot do so because (i) the human demonstrator's `SPACE` presses have no valid target action head to clone into, and (ii) the current agent has no `is_ready` signal to condition option-termination on. **Below the "sufficient" threshold on this axis, BC and Hierarchical are both bottlenecked by the same information leak.**

**Axis B — Compute budget (consumer GPU, n_envs=2, ~2.4 sps).**
The 40.8h Dreamer run moved policy entropy 0.08% off uniform (3.804/3.807; Agent 3 §Exec-summary). Precedent (NetHack — Hambro 2022; AlphaStar; OpenAI Five) is unanimous: **cold-start RL on procedural-roguelike-scale complexity requires either scale we do not have, or a demonstration prior.** This axis makes BC dominant in the near term because a 2-layer BC actor fits in <5 min on the 3060 Ti (Agent 3 §Prop-C), while a Hierarchical actor over a still-uniform primitive policy trains the same primitive from scratch — one hierarchy level up but with the same compute floor.

**Axis C — Temporal horizon of the target behavior.**
`imag_horizon=20 ticks = 1.33s`; room-clear = 5–20 s; floor traversal = 60–120 s; boss = 30–60 s (Agent 3 §Prop-D). Flat actors — including a BC-cloned one — cannot bridge this 20–90× gap via critic bootstrap because `loss/actor_target_mean=47` is a runaway inflation, not a value estimate. **This axis makes Hierarchical dominant *for the long-horizon objectives* (beat_mom, floor progression) but not for the immediate objectives (room-clear, dodge, door choice) that BC already covers.**

**Axis D — Demonstrator ↔ obs surjectivity.**
BC works if and only if the demonstrator's decision function is a computable map from what the *agent* observes. Right now that map is broken (humans use minimap, trinket HUD, transformation icons; agent gets none). Fixing Axis A closes this. Alternately, one could ask humans to play with a "training HUD" that renders only the obs vector (Agent 3 §Prop-C rebuttal) — cheap and possible.

**Axis E — Reward-hack surface area.**
Every past cycle produced a new shaping hack (v0 idle → v1 stationary → v1.5 backtrack → v2 clear_idle → v2 seek_door → v3 cap; Agent 3 §Prop-C). BC bypasses this entirely — the demonstrator's actions define the loss, and shaping becomes a fine-tune bonus, not the training signal. Hierarchical does *not* bypass it: option-selection needs its own reward, and if the primitive policy is still shaping-fed, the same hacks resurface at option granularity (agent picks the option that maximizes shaping accumulation).

---

## Regime map

### Regime R0 — **CURRENT** (Axis A = broken). Neither wins; **obs+action rehab is the correct next investment.**
Preconditions: no space-bar action; passives is presence-only MultiBinary(256) with unknown-collectible drop bug; no transformation counters; no minimap; no active-charge; door-target enum covers 3/15 room types; only 3 room types ever entered in 40h (Agent 3 §Non-obvious #7); reward-shaping cycle unfixed.

Under these conditions:
- **BC fails deterministically.** Cloned actions target a MultiDiscrete([9,5]) space that literally cannot represent SPACE, drop-bomb, or use-card. Agent 1's entire premise (item quality, transformations, trap-item avoidance) is *unlearnable-from-demonstration* because the demonstrator's item-pickup decision depends on unobserved features (quality, pool, transformation-progress). Even if the human is asked to play with a training-HUD, ~30% of skilled Isaac decisions still fall outside the obs (minimap layout, card/pill preview, door-target). BC would clone a shadow policy that ignores half the game.
- **Hierarchical fails less-deterministically but still.** Option termination predicates (e.g., `move_to_door_slot_k terminates on new-room`) need obs signals that mostly exist. But options like `purchase_pedestal_j`, `use_active`, `use_card_i` are impossible because (i) no space-bar / drop / card action heads exist, (ii) no active-charge readiness signal exists, (iii) pedestal price is not read into the encoder (Agent 1 §Gap 8). ~40% of the useful option set is un-instantiable.

**Correct move for R0:** Ship Agent 3's Prop-B priority 1–4 (active-charge/id, transformation counters, passives→count vector, trinket/card/pill slots), Agent 2's Priority-1 change #3 (restore `use_active` action head with masking), Agent 1's Priority-1 R1+R3+R5 (quality-weighted pickup, transformation bonus, trap-item override). This is 1–2 weeks of engineering, unblocks both BC and Hierarchical, and produces immediate quality-of-training gains before we spend a single GPU-hour on either method.

Confidence: **High.** The evidence is mechanical, not empirical — the action space demonstrably lacks the required heads and the obs demonstrably lacks the required fields. This is the "hidden third option" the framing invited.

### Regime R1 — Post-obs-rehab, near-term (Weeks 3–8). **BC-bootstrap wins.**
Preconditions: Axis A repaired; ≥20h of skilled human demo recorded in the new obs+action schema; compute unchanged.

- Sample-efficiency argument: BC actor on ~20h of demo produces a policy that (per NetHack Hambro 2022 precedent) beats the current 346k-step agent on rooms_visited, boss_kills, use_item, keys_used by ≥10×. Wall-clock: 4h train.
- Reward-hack argument: BC replaces the shaping cycle. The v0→v3 hack sequence stops because the training signal is imitation loss, not shaping accumulation. This alone justifies the investment.
- Composition argument: A BC actor *is* the intra-option controller that Hierarchical needs in R2. Doing BC now is not a detour from Hierarchical — it is the first stage of it.
- **Why not Hierarchical here?** Because a Hierarchical actor trained via RL over an untrained primitive controller re-inherits all the "primitive policy is uniform-random" pathologies at option granularity. The high-level actor sees option-returns that are dominated by "how random the primitive is" — the credit assignment problem doesn't collapse, it relocates.

Confidence: **High** (BC dominance in R1). The precedents (NetHack, StarCraft, Dota) are unanimous and Isaac's demo-data availability (Northernlion ~10k runs; deterministic seed replay; Agent 3 §Prop-C) is on the strong-BC side of the reference class.

### Regime R2 — Post-BC-plateau, long-horizon objectives (Weeks 8+). **Hierarchical wins.**
Preconditions: BC actor exists and beats current agent by ≥10× on short-horizon metrics; RL fine-tune is plateauing on `floors_reached`, `boss_kills`, `beat_mom` because of the temporal-horizon problem.

- The imagination-horizon-vs-macro-action-timescale gap (20 ticks vs 60–120s; Agent 3 §Prop-D) is a fundamental flat-actor limitation. Bumping `seq_len` and `imag_horizon` is the cheap fix but caps at ~4s; a full run is 20–40 min. Only option-level imagination collapses this to a tractable planning horizon.
- The option set becomes definable *because* the obs was fixed in R0: `{move_to_door_slot_k, clear_current_room, purchase_pedestal_j, use_active_when_ready, use_card_i, bomb_wall, wait}` each has a well-defined termination predicate now.
- BC actor as intra-option controller: the "move_to_door_slot_k" option can be executed by the BC actor conditioned on a goal-embedding (target door slot), inheriting all the dodge/aim/navigation competence BC already learned.
- **Why not BC here?** BC does not solve credit assignment for "was this Devil deal worth it 3 floors later?" The demonstration prior encodes *what a human does*, not *why*. For post-plateau novelty (routes a human wouldn't take), we need search or hierarchical exploration.

Confidence: **Medium-High** (Hierarchical dominance in R2). Depends on whether the BC-warmed flat actor actually plateaus vs. continuing to climb — this is genuinely uncertain until measured.

### Regime R3 — **Compose-both** (the strongest position). BC provides primitives; Hierarchical structure emerges via demo-mined options.
This is the specific composition the user asked about, and it's the target design after R0.

Mechanism: cluster demonstration trajectories on **behavior segments** (e.g., "human left room via NORTH door within 5s of clear"). Each cluster becomes an option. The high-level policy is BC-trained on option-labels (which cluster did the human enter next?); the low-level policy is BC-trained on primitive actions conditioned on the option-label. Then RL fine-tune both levels. This is Option-Discovery-from-Demonstrations (Fox et al. 2017; Krishnan et al. 2017); it eliminates the "define the option set by hand" risk in Agent 3's Prop-A alternative.

**This regime dominates R1 and R2 whenever demo data is sufficient (~≥50h)** because:
- Option boundaries are learned from data rather than hand-coded, avoiding the "wrong options → bad action space" failure mode Agent 3 flagged.
- The primitive controller is BC-warmed, so option-level RL sees meaningful returns from the first update.
- The value-inflation loop (Agent 3 §Prop-A point 2) is broken at option boundaries because option-returns terminate at real termination predicates, not `cont≈1` forever.

Confidence: **Medium** for the specific claim that demo-mined options beat hand-designed options on Isaac. Demo-mined option discovery has a mixed empirical record; on the other hand, hand-designed options on Isaac's 15-room-type / 200-active / 700-item state space are hard to enumerate.

### Regime R4 — **BC ineffective corner case.** Demonstrator quality is too low or demo count <10h.
If we cannot recruit ≥1 skilled Isaac player for ≥20h, BC's ceiling drops below the current agent's floor. In this regime Hierarchical becomes correct-by-default because we cannot bootstrap. Confidence: **High** (mechanical), but Agent 3 argues (correctly) that Isaac has abundant public demo data (Northernlion, Hutts, ThePlushGiant, etc.). We should verify demo availability *before* committing.

---

## The composition claim (why "BC vs Hierarchical" is a false binary)

The strongest claim in this steelman: **there is no design in which choosing BC excludes Hierarchical, or vice versa.** They occupy orthogonal layers:

- BC operates on the **actor-parameterization** layer: it initializes the policy weights from imitation loss instead of RL loss.
- Hierarchical operates on the **temporal-abstraction** layer: it collapses N primitive steps into 1 option-step for credit assignment.

Doing BC now produces an actor that is *strictly useful* as either (a) a flat actor for RL fine-tune, or (b) an intra-option controller inside a later Hierarchical wrapper. The path is monotonic: BC → BC+RL-flat → BC+RL-flat + option-wrapper. Each step is a strict superset of the previous.

Conversely, doing Hierarchical *first* (without BC) means the intra-option policies are trained by RL from scratch under the same compute+obs conditions that produced the 40h uniform-entropy run. The high-level policy sees option-outcomes dominated by primitive-policy noise. Agent 3's Prop-A ranks Options-HRL as best-fit *architecturally* but does not claim it is the cheapest-first move — that would contradict Agent 3's own Prop-C ranking of BC as #1.

**Therefore:** the correct read of Agent 3's audit is **BC before Hierarchical, both after Obs**. This is not a compromise between two audits — it is the audit's own priority ordering (Prop-C #1, Prop-B #2, Prop-A #3).

---

## Preliminary regime-conditional answer to the thesis

- **In R0 (today, unfixed obs/action):** Thesis is **FALSE**. Neither BC nor Hierarchical is the correct next investment. Obs+action rehab dominates both. Confidence: High.
- **In R1 (post-obs rehab, near-term):** Thesis is **TRUE**. BC has strictly higher leverage per GPU-hour. Confidence: High.
- **In R2 (post-BC plateau, long-horizon):** Thesis is **FALSE**. Hierarchical has higher leverage on the long-horizon objectives that remain unlearned. Confidence: Medium-High.
- **In R3 (compose-both):** Thesis is a **CATEGORY ERROR**. BC and Hierarchical are complementary layers; you do both in sequence, not one vs the other. Confidence: High.
- **In R4 (no demo data available):** Thesis is **FALSE** for reasons unrelated to Hierarchical's merits. Confidence: High.

The *modal* answer — the one that applies to the next 4–6 weeks of work under realistic conditions — is: **fix obs first (2 weeks), then BC (4 weeks), then wrap in Hierarchical (post-plateau).** The thesis as literally stated ("BC > Hierarchical") is true in the middle segment of that plan and false at both ends.

---

## Evidence quality flags

- **Settled:** Actor entropy 3.804/3.807 after 40h (Agent 3 §Exec-summary — mechanical). `use_item=0` in 346k steps (Agent 3 §Prop-B). Action space lacks space-bar (Agent 2 §Reward-function §item 9). Value head clips r_beat_mom at ±20 (Agent 3 §Non-obvious #4). Only 3 room types entered in 40h (Agent 3 §Non-obvious #7). All directly grep-able from code.
- **Well-supported:** BC-bootstrap dominance on procedural-roguelike + demo-abundant games. Reference class: NetHack (Hambro 2022 "Dungeons and Data"), AlphaStar, OpenAI Five. Confidence High that it transfers to Isaac.
- **Uncertain:** Exact wall-clock for BC training on the 3060 Ti in *our* schema (Agent 3 estimates 4h; not measured). Whether demo-mined options beat hand-designed options on Isaac specifically. Whether Hierarchical fine-tune post-BC extracts meaningful additional gains vs. simply pumping seq_len/imag_horizon (cheap flat-actor fix).
- **Genuinely open:** Whether the current mod's `restart` behavior + Eden character randomization would corrupt BC obs recording — need to force character determinism (Agent 2 §Priority-1 #1) before recording demos.

---

## Sources

- `agent1-item-economy.md` — item-quality/pool/transformation obs gaps; trap-item enumeration.
- `agent2-character-strategy.md` — action-space amputation (removed use_active on 2026-07-02); Isaac-only training defense; obs schema gaps for coin/bone/rotten hearts.
- `agent3-adversarial-audit.md` §Prop-A (Dreamer configuration); §Prop-B (obs gaps enumeration); §Prop-C (BC-bootstrap rationale + NetHack/AlphaStar precedent); §Prop-D (temporal horizon 20 ticks vs 60–120s macro-action); §Prop-E (curriculum-of-one, 3-of-12 skills trained); §Non-obvious #1–8 (uniform-entropy deception, critic hallucination, RND no-op, value clipping, room-type coverage).
- External references implied by audits: [Item Quality wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Item_Quality); [Transformations — tboi.com](https://tboi.com/transformations); Hambro et al. 2022 "Dungeons and Data" (NetHack BC); Vezhnevets 2017 (Feudal Networks); Bacon 2017 (Option-Critic); Fox et al. 2017 / Krishnan et al. 2017 (option discovery from demos).

[WROTE: /var/folders/fn/rb1gwy910bjcy80d31y8gy700000gn/T/pi-subagents-uid-501/chain-runs/4cffab9f/argue/r1-steel.md]
