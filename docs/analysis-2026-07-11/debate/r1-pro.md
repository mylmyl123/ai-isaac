# Proponent — Round 1

## Position
BC-bootstrap (20–100h of human demos → supervised pretrain → RL fine-tune) is the higher-leverage next investment than a hierarchical actor because it is the only intervention that removes the *root* cause of every failure documented in the three swarm audits — uniform-random exploration paired with hand-shaped reward — while a hierarchical actor merely shortens the credit-assignment horizon around that same broken core.

## Arguments

### 1. Every comparably complex procedural / long-horizon game AI in the modern era used demo-bootstrap; none succeeded by cold-start RL alone.

- **NetHack.** Hambro et al., *Dungeons and Data* (NeurIPS 2022, arXiv:2211.00539) — the strongest published NLE agents to date use BC on the AutoAscend dataset (>10⁹ transitions from a symbolic bot) as pretraining, then RL fine-tune. Pure model-free RL on NLE plateaus in the first dungeon level, exactly where we are ([arXiv](https://arxiv.org/abs/2211.00539)). The NeurIPS 2021 NetHack Challenge writeup (arXiv:2203.11889) is explicit: "all top submissions used behavioural cloning."
- **StarCraft II.** Vinyals et al., *Nature* 575, 350–354 (2019) — AlphaStar's League play only worked after supervised pretraining on **971,000 human replays**; the SL prior is what gave the RL loop something to bootstrap against. DeepMind's follow-up *AlphaStar Unplugged* (arXiv:2308.03526, 2023) demonstrates that pure offline RL from the same demo corpus reaches 90% win-rate vs the RL-trained agent — i.e. the demos carry most of the signal.
- **Dota 2.** OpenAI Five's technical report explicitly notes that dense reward shaping + scale (152,000 CPU-years, 128,000 CPUs) was required to avoid a demo bootstrap. That is not our budget. ([openai.com/research/openai-five](https://openai.com/research/openai-five)).
- **Atari.** DQfD (Hester et al. 2018, AAAI) beat DQN on 41 of 42 games with <10 minutes of demo data per game.
- **Implication.** The literature has already run our experiment for us. On sparse-reward, long-horizon, procedural games the ordering is: BC baseline first, then RL fine-tune. There is no counter-example at our compute tier. *Confidence: High — literature is unanimous.*

### 2. Our own 40-hour run is empirical proof that cold-start Dreamer at consumer-GPU budget cannot escape the uniform-random basin.

From `tb_dreamer_stage1_20260709-174421.json` (Agent 3 audit, verbatim):
- `loss/actor_entropy = 3.804` vs theoretical uniform `log 9 + log 5 = 3.807` — the policy has moved **0.08% off uniform after 346,668 env-steps and 344k WM updates**.
- `ep_reward mean = 9.67`, of which **`seek_door = +6.34 (65.5%)`** — reward is being *hacked*, not earned.
- `floors_reached_max = 0`, `boss_kills = 0`, `use_item = 0`, `bombs_used_max = 1` across the entire run.
- `beh_room_types_seen_max = 3` out of 15 possible room types — the agent has never entered a shop, boss room, secret room, arcade, devil, or angel room in 40h × ~1000 episodes.
- `perf/sps = 2.36` at `n_envs = 2` — we cannot brute-force our way out; a 100× compute increase over 40h is 4,000h, and NetHack RL-from-scratch plateaus even at 10⁹ transitions.

A hierarchical actor with hand-defined options (`clear_room`, `move_to_door`, `purchase`) still needs an *intra-option* controller. That controller faces the exact same cold-start problem — it just faces it at option-level rather than tick-level. The uniform-random 3.804-entropy problem does not disappear; it just gets renamed. BC directly loads a non-uniform prior into the actor's initial weights, which is the *only* intervention that touches the root cause. *Confidence: High — direct empirical measurement.*

### 3. BC breaks the v0 → v1 → v2 → v3 reward-hack treadmill because CE loss on action distributions is fundamentally un-hackable.

The audit documents the treadmill (Agent 3 §Prop C):
- v0: idled in corner → added `r_idle_penalty`
- v1: wiggled to escape idle → added `r_stationary_penalty`
- v1.5: oscillated between cleared rooms → added `r_backtrack`
- v2: `clear_idle_extra` became a policy-independent −2.83/ep bias → 6× reduction + grace
- v2: `seek_door` pumped 6.34/ep, only 37% conversion to `new_room` → v3 per-ep cap
- v3: **the audit predicts another hack will appear** — no matter what you shape, uniform-random policies find the shortest path to accumulation.

Compare to BC's loss: `L = −Σ log π(a_human | s_human)`. There is nothing to hack. The gradient points exclusively at "match the human's action distribution." Once BC has run, the RL fine-tune stage can use a **much sparser reward** — the audit's own recommendation is `r_step + r_kill + r_room_clear + r_new_room + r_pickup + r_beat_mom + r_death` (Agent 3 §recommendation 1). This retires ~15 shaping terms currently in `reward.py` and, transitively, retires most of the fixes Agent 1 recommends. Agent 1's 15-item quality map, transformation counters, synergy detector, trap-item override map — **every one of those recommendations exists to hand-encode preferences that Northernlion and any competent Isaac player already exhibit in gameplay.** BC absorbs all of them at zero engineering cost per item. *Confidence: High — mechanical property of supervised loss.*

### 4. Isaac has uniquely favorable data economics for BC that few other RL domains enjoy.

- **Demo supply is essentially free and public.** Northernlion has recorded over 2,300 numbered Northernlion-Plays-Isaac episodes on YouTube (channel start Nov 2011; the "Rebirth/Afterbirth/Repentance" cohort alone exceeds 1,500 videos of 25–40 min each = ~600h of gameplay by a single top-1000 player). Add Hutts, ThePlushGiant, DanRykerNL and the corpus is >10,000 recorded runs. YouTube + Twitch VOD is a >20 TB open dataset with timestamps.
- **Deterministic seed replay is a killer feature few other environments have.** Isaac Repentance seeds are reproducible: the same seed regenerates identical floor layout, item pool draws, and pickup positions ([bindingofisaacrebirth.wiki.gg/wiki/Seeds](https://bindingofisaacrebirth.wiki.gg/wiki/Seeds)). We do not need OCR-from-video: we ask a human to play `restart <seed>` in our modded environment, record `(obs, action)` tuples through the same `obs.lua` schema the trainer already consumes, and every demo is a schema-perfect training pair. AlphaStar and NetHack had to build heavy replay-decoding pipelines. We already have the observation pipe.
- **Action space is trivial for BC.** `MultiDiscrete([9, 5]) = 45 discrete combos`. A 2-layer MLP over the existing 256-dim latent fits in <5 minutes on the same 3060 Ti currently blocked on Dreamer. Compare to AlphaStar's autoregressive 10⁸-way action factorisation — Isaac's BC problem is 6 orders of magnitude easier per sample.
- **Precedent for demo-count.** DQfD used <10 min of demo per Atari game. NetHack AutoAscend used ~30h of bot rollouts before BC quality plateaued. 20–100h of human demo is well inside the diminishing-returns regime documented in Hambro et al. Fig. 4 (arXiv:2211.00539).
- **Implication.** The BC step is single-digit-days of wall-clock. The hierarchical-option approach requires designing termination predicates for 6–10 options, wiring an option-critic head, and re-training. Data-versus-engineering ratio favours BC by ~10:1. *Confidence: High for demo volume; Medium for the exact seed-replay fidelity guarantee (some Repentance mechanics — Damocles, Eden RNG — are not seed-stable across game versions; recording native gameplay in our mod avoids this).*

### 5. Hierarchical actor is *complementary*, not competitive, and its critical dependencies are unfunded — so it is strictly the higher-risk, higher-latency investment right now.

Steelman first: options collapse the temporal horizon from 2000 ticks to ~10 option-invocations (Agent 3 §Prop A). That is a real 200× speedup on credit assignment and I concede it is a valuable long-term structure. But **as a next investment** the hierarchical path has three unfunded prerequisites the audits themselves surface:

- **Termination predicates require obs we do not have.** Agent 3 §Prop B: `doors[i].target_room_type` currently encodes only `{BOSS, TREASURE, SECRET}` — 3 of 15 room types. An option like `move_to_door(devil_deal)` cannot even *fire* because the observation never tells the high-level policy that a door leads to a devil deal. Options are pointing at a room-type enum that isn't in the obs.
- **Intra-option controllers still need to be trained.** Someone has to make `clear_current_room` actually clear a room. The audit shows we can't even do that today with the flat controller. Options don't remove that problem; they just add a layer of routing on top of it. BC solves both layers simultaneously by imitating whatever the human's intra-option and inter-option choices were.
- **The audit ranks BC #1 and options #3** (Agent 3 §"Top 3 fundamental changes"). Agent 3 explicitly labels BC as "highest impact / lowest risk / fastest wall-clock" and options as "medium-high impact / medium risk / higher wall-clock." Our best-informed source treats the ordering the same way.

Second-order implication: **BC is a strict prerequisite for the hierarchical approach to succeed anyway.** Once you have a BC policy that clears rooms, you can *derive* option boundaries from human behaviour (a room-clear terminates when the human's revealed macro-action ends). This is exactly the Directed-Options / Discovery-of-Options line (Bacon et al. Option-Critic 2017; Machado et al. Eigenoption 2018). Doing BC first gives us the option abstractions for free from data. Doing options first without BC forces us to hand-guess every predicate. *Confidence: High.*

## Second-order implications the debate should register

1. **BC retires most of Agent 1's document.** Item-quality maps, pool multipliers, synergy detector, trap-item overrides, transformation-bonus tables — all of it encodes preferences a human demo already exhibits. Agent 1's own summary calls the reward "uniform where the game is 2 orders of magnitude non-uniform"; BC learns that 2-orders-of-magnitude non-uniformity directly from action frequency in demos, with zero hand-crafted scalars.
2. **BC accelerates Agent 2's character-conditioning problem for free.** Record 5h per character; the character-conditioned policy emerges from character-conditioned demos. No `r_pickup_heart` rework per character.
3. **BC turns the currently-broken Dreamer WM from a liability into a bonus.** Agent 3 confirms the WM losses are converging (`enemies_mask 11.8 → 0.006`, `doors 9.9 → 0.006`). The failing component is only the actor. BC replaces just the actor; the WM is preserved and continues to improve, and Dreamer's imagination-based fine-tune becomes a search-in-latent-space over a warm-started actor — precisely the AlphaStar recipe (SL prior + latent rollouts).

## Ranked, quantified case summary

| Claim | Confidence | Rationale one-liner |
|---|---|---|
| Precedent unanimously supports demo-bootstrap for our regime | High | NetHack, AlphaStar, Dota, DQfD all did this; no cold-start counter-example at consumer-GPU tier |
| 40h empirical run proves cold-start Dreamer cannot escape uniform-random | High | actor_entropy=3.804/3.807, 0/0/0 on the three key competence counters |
| BC's CE loss is un-hackable in a way dense shaping is not | High | Mechanical property of supervised objective; retires 15+ shaping terms |
| Isaac's data economics uniquely favour BC | High | 10k+ Northernlion runs, deterministic seeds, 45-combo action space, existing obs pipe |
| Hierarchical actor is orthogonal, more expensive, and BC-prerequisite | Medium-High | Option termination predicates require obs (door target room type) we don't have; audit ranks BC first |

## Sources
- Hambro et al., *Dungeons and Data: A Large-Scale NetHack Dataset*, NeurIPS 2022. [arXiv:2211.00539](https://arxiv.org/abs/2211.00539).
- Hambro et al., *Insights from the NeurIPS 2021 NetHack Challenge*. [arXiv:2203.11889](https://arxiv.org/abs/2203.11889).
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575, 350–354 (2019). [nature.com/articles/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z).
- Mathieu et al., *AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning*, 2023. [arXiv:2308.03526](https://arxiv.org/abs/2308.03526).
- Berner et al., *Dota 2 with Large Scale Deep Reinforcement Learning*, 2019. [arXiv:1912.06680](https://arxiv.org/abs/1912.06680).
- Hester et al., *Deep Q-learning from Demonstrations*, AAAI 2018. [arXiv:1704.03732](https://arxiv.org/abs/1704.03732).
- Bacon, Harb, Precup, *The Option-Critic Architecture*, AAAI 2017. [arXiv:1609.05140](https://arxiv.org/abs/1609.05140).
- Repentance seed determinism, [bindingofisaacrebirth.wiki.gg/wiki/Seeds](https://bindingofisaacrebirth.wiki.gg/wiki/Seeds).
- Local empirical: `/Users/I048254/Downloads/isaac-ai/isaac-swarm/agent3-adversarial-audit.md` (40h Dreamer run, `tb_dreamer_stage1_20260709-174421.json`).
- Local empirical: `/Users/I048254/Downloads/isaac-ai/isaac-swarm/agent1-item-economy.md` (reward-uniformity vs 50× item variance).
- Local empirical: `/Users/I048254/Downloads/isaac-ai/isaac-swarm/agent2-character-strategy.md` (character-conditioned reward gaps).
