# Isaac RL — 2026-07-11 Deep Analysis + Debate Verdict

Two parallel investigations were run this session to reset the project's strategic direction after the 40h `stage1_single_room_xs` run stalled at policy-uniform (actor_entropy 3.804/3.807 max).

## Files

### Swarm audits (parallel deep-dive research)

- **[agent1-item-economy.md](agent1-item-economy.md)** — Isaac item economy gaps: 700-item quality field (0-4) unused, item pools uncollapsed, transformations invisible, trap items unpunished (Plan C = auto-lose currently rewards +2 use_item), 30 game-defining synergies unrewarded.
- **[agent2-character-strategy.md](agent2-character-strategy.md)** — Character-specific playstyle differences. Mod uses `restart` bare (not `restart 0`), so character isn't guaranteed Isaac. `max_hp = 6.0` hardcoded. Space-bar action removed 2026-07-02 → Isaac's D6 unreachable, Lilith unplayable (0 base damage without Incubus).
- **[agent3-adversarial-audit.md](agent3-adversarial-audit.md)** — Adversarial system audit. Most damning finding: **the 40h run wasn't learning at all** — actor_entropy=3.804 vs max 3.807 (99.92% uniform), ep_reward gains came from reward-function pumping across restarts, not policy learning. Value head clips `r_beat_mom=+50` at `v_max=+20` (mechanical bug). RND is silent (predictor_loss=6.8e-6). Obs missing critical info (no active-charge, no transformations, no minimap, only 3/15 door target types). Curriculum-of-one trains 4 of 12 required skills.

### Structured debate — BC vs Hierarchical

`debate/` contains the full 3-round proponent/opponent/steelman transcripts + judge verdict on the thesis:

> "BC-bootstrap (record human demos, pretrain actor, RL fine-tune) is a higher-leverage next investment than Hierarchical-actor (options + termination predicates + high-level policy)."

- **[debate/verdict.md](debate/verdict.md)** — Judge's adjudication. **58/100 conditional win for BC**, medium confidence. Score would swing ±15 depending on the gating experiment result (see below).

## The unanimous novel finding

All three debate personas — proponent, opponent, AND steelman — independently converged in Round 3 on the same recommendation:

> **Run an `imag_horizon: 20→120` experiment on the existing 40h checkpoint. 2-3 days GPU, ~$5. Resolves the load-bearing question empirically.**

The load-bearing question: is the actor-entropy collapse `H_soft` (fixable hyperparameter — truncated planning horizon) or `H_hard` (architectural attractor — critic hallucinating `V≈47` everywhere with `cont≈1` and inflated targets)?

- If `actor_entropy` drops below 3.5 within ~2h of additional training → **`H_soft`** → BC-bootstrap wins → proceed to Track A2 (BC + AWAC anchor)
- If entropy stays pinned near 3.804 → **`H_hard`** → Hierarchical wins → proceed to Track A3 (Director wrapper over DreamerV3)

Neither committing to a 6-week BC pipeline nor a 4-week hierarchical rewrite is justified without this data.

## Recommended critical path (from Steelman R3)

### Weeks 1-2 — Unconditional foundation (3 parallel tracks)

**Track A — Obs + action rehab.** Shared prerequisite for both branches:
- Value head: `value_v_max: 20 → 100` (unclip `r_beat_mom=+50`)
- Restore `use_active` action head with masking: `MultiDiscrete([9,5,2])` where factor 3 = press-space, masked when `active_charge == 0` OR `active_item_id == 0`
- Add active-item obs: `player.active_item_id`, `active_charge`, `active_max_charge` (mod-side `player:GetActiveItem()`, `player:GetActiveCharge()`)
- Add transformation counters: `player.transformations[15]` from `player:GetPlayerFormCounter(0..14)`
- Ship item quality on `pickup_collectible` event (mod caches `pickup.SubType` pre-frame, reads `Isaac.GetItemConfig():GetCollectible(id).Quality`)
- Replace flat `r_pickup_collectible=2.0` with `r_pickup_collectible_by_quality = (0.5, 1.0, 2.0, 3.5, 6.0)` per Q0..Q4
- Trap-item override table with Plan C (id 475) = -20 on `use_item`
- Expand `PASSIVES_K` beyond 256 (or use learned embedding for unknown IDs)
- Expand `doors[i].target_room_type` from 3 (BOSS/TREASURE/SECRET) to all 15 room types
- Force `restart 0` (Isaac) in mod boot; add `player.player_type` to obs; assert Isaac in env reset

**Track B — Gating experiment.** Highest-leverage, cheapest, first.
- Load existing 40h checkpoint
- Config override: `imag_horizon: 20 → 120`, `value_v_max: 20 → 100`, `rnd_intrinsic_scale: 0.1 → 100.0` (retune after v3 real-features fix — current predictor_loss is 6.8e-6 not 1e-3)
- Add `cont`-flag misprediction logging (detect if imagined `cont≈1` diverges from replay-observed episode ends)
- Train 2-3 days
- Decision criterion: `actor_entropy < 3.5` within 20k updates AND `actor_target_mean` doesn't inflate proportionally

**Track C — Repentogon TAS bot wire.** De-risk demo supply.
- 1 engineer-week to wire a scripted bot: walk-to-nearest-door, shoot-at-nearest-enemy, pickup-any-Q≥2-pedestal, use-active-when-charged
- Overnight-record 500h through the same in-mod schema (frozen from Track A)
- BC-stage-1 corpus (broad coverage, mediocre skill)
- Optional: recruit 10-20h of human demos as BC-stage-2 (narrow, higher skill) — VPT-style two-stage BC

### Week 3 — Branch decision (based on Track B result)

- **`H_soft` → R1a path**: BC-warm actor on TAS corpus (+ optional human demos) → AWAC-anchored fine-tune with persistent `KL(π‖π_BC)` term decaying via β(t). Target: `floors_reached ≥ 2`, `boss_kills > 0` by end of week 4.
- **`H_hard` → R1b path**: Director wrapper on preserved WM (existing DreamerV3, same author, strict local upgrade). Worker warm-started from TAS-only BC. Manager cold-start RL on option-scale imagination. Target: `rooms_visited > 5`, `use_item > 0`.

### Weeks 5-8 — Compose the other side

- R1a → R2: If flat plateau, apply Fox 2017 / DDCO option discovery on the BC trajectory buffer. Options wrap around.
- R1b → R2: Collect 20h option-level human demos (10-30 macro-decisions/episode, 50-100× cheaper than primitive-level). BC the option-selector, retain worker.

### Week 9+ — Terminal training

Both branches converge to: **Director-style hierarchical actor + BC-warm workers + AWAC anchor + option-level demos + character-embedding conditioning**. Same architecture, different ordering.

## Sources cited across the analysis

- Hambro et al. 2022, *Dungeons and Data*, [arXiv:2211.00539](https://arxiv.org/abs/2211.00539) — NetHack BC+RL precedent
- Baker et al. 2022, *VPT*, [arXiv:2206.11795](https://arxiv.org/abs/2206.11795) — two-stage BC (broad bot → narrow human)
- Vinyals et al. 2019, *AlphaStar*, [Nature 575:350](https://www.nature.com/articles/s41586-019-1724-z)
- Hafner et al. 2023, *DreamerV3*, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Hafner et al. 2022, *Director*, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114) — hierarchical over Dreamer WM, strict local upgrade
- Nair et al. 2020, *AWAC*, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359) — offline-to-online RL with KL anchor
- Ross et al. 2011, *DAgger*, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) — covariate shift in BC
- Rajeswaran et al. 2017, *DAPG* — BC wash-out precedent
- Fox et al. 2017, *Multi-Level Discovery of Deep Options*, [arXiv:1703.08294](https://arxiv.org/abs/1703.08294)
- Krishnan et al. 2017, *DDCO*, [arXiv:1710.05421](https://arxiv.org/abs/1710.05421)

## Immediate next actions

**In this session (before compact):** all analysis artifacts have been preserved to this directory. Nothing more required.

**Next session priorities:**
1. Ship Track B (gating experiment) config override — smallest PR, highest leverage. See `python/isaac_rl/dreamer/configs/stage1_single_room_xs.yaml` for the imag_horizon / value_v_max / rnd_intrinsic_scale bumps.
2. Add `cont`-flag misprediction logging to `dreamer/isaac_models.py::_train_step_inner`.
3. Verify checkpoint resume works with the config override (may need to force actor reset if the checkpoint has stale optimizer state).
4. User runs experiment on Windows box.
5. After experiment: analyze result, decide branch, proceed to Track A obs+action rehab (per debate plan).
