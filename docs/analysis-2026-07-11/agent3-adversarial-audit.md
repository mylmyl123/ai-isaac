# Isaac RL — Adversarial Audit
_Agent 3 / Devil's Advocate. Reads: `spaces.py`, `reward.py`, `obs.lua`, `isaac_models.py`, `stage1_single_room_xs.yaml`, and the last 10 TB dumps (esp. `tb_dreamer_stage1_20260709-174421.json`, 40.8h / 346,668 env-steps, the run where the v2 reward hack was diagnosed)._

---

## Executive summary

The v3 patch will paper over the specific `seek_door` hack but the design is deeper-broken than that. The 40h run is a **learning illusion**: `loss/actor_entropy = 3.804` versus theoretical max **3.807** — after 346k env-steps and 344k WM updates the actor has moved **0.08%** off uniform-random, yet `rollout/ep_reward_best` shows a monotone climb to 76.7 and `rollout/ep_reward` mean of 9.67. Reason: **65% of episode reward comes from `seek_door` alone (6.34/9.67 mean)** on what is essentially a uniform-random controller — the "improvement" curve is measuring reward-function coefficient tuning across restarts, not policy learning. The three fundamental design errors are (1) **hand-shaped reward + no imitation bootstrap** in a game with abundant human demo data, (2) **an obs that is missing every strategically relevant piece of Isaac state** (active-item charge, transformations, minimap, card/pill/trinket inventory, item stack counts, door target-type coverage), and (3) **a temporal horizon that is 20-100× too short for the game's natural macro-actions** (imag_horizon=20 ticks ≈ 1.3s of game time vs 60-120s for a floor). The v3 rework does none of these. `perf/sps=2.36` at `n_envs=2` also means the wall-clock budget will run out long before the current formulation can converge even to Stage-1 room clearing, let alone Mom.

---

## Proposition A — Dreamer architecture

**Position: MILD DISAGREE that Dreamer is *fundamentally* wrong; STRONG AGREE that the current Dreamer *configuration* is a bad fit and that MuZero / Options-HRL would be strictly better.**

### Evidence Dreamer is not fatally wrong here
- The world-model losses show it *is* learning obs structure: `loss/enemies_mask 11.8 → 0.006`, `loss/global 63.7 → 0.06`, `loss/doors 9.9 → 0.006`, `loss/enemies_feats 0.56 → 0.004`. Reconstruction is fine; the WM is not the failing component.
- Isaac's transition dynamics are near-deterministic given full state (procedural but seed-stable within a run). WMs shine when dynamics are learnable; they are here.
- `loss/reward` low = the reward head predicts our shaped reward well, which is precisely what we asked it to.

### Evidence the current Dreamer *configuration* is broken
1. **Symlog-disc value head range is [-20, +20] with 255 atoms** (`value_v_min=-20, value_v_max=20`, `isaac_models.py:reward_head`). But `r_beat_mom=+50`, `r_boss_kill=+20`, `r_death=-3` after prior policies used `-10`. **The single largest reward the agent could ever earn is clipped by 60%** at the value head. The critic literally cannot represent "beating Mom is worth twice as much as clearing a floor."
2. **imag_horizon = 20 ticks × 1/15 Hz = 1.33 seconds of game time**. Actor training rolls out 1.3s of imagination. Room-clear takes 5–20s; door traversal is 60–90s at level start; boss fights 30–60s. The imagined trajectory *never reaches a termination or a floor-clear*. This is trivially observable: `loss/cont ≈ 3.5e-4` (imagined continue-prob ≈ 1.0 for every step), so imagination is a semi-infinite series discounted by `gamma=0.999` — with 6-per-step reward from `seek_door` and cont≈1, the λ-return blows up. **Confirmed**: `loss/actor_target_mean=47, std=25` — the critic thinks every state is worth ~47 future reward. This is a value-inflation loop, not learning.
3. **Reward EMA whitening kills the useful gradient.** `loss/actor_adv_abs_mean = 0.037` vs `loss/actor_entropy_bonus_mag = 0.011`. Nominally the reinforce signal should dominate 3:1 — but the entropy is stuck at max (`3.804 / 3.807`). The 5–95 percentile EMA is being computed over imagined targets that already include the pumped `seek_door`, so useful advantages compress toward zero relative to the noise.
4. **Batch size 8 × seq_len 32 = 256 samples/update**. This is *fine* per Danijar's ablation, but combined with `n_envs=2` and 15 Hz, useful diversity per replay window is tiny.
5. **`compile_rssm=false`** in the actual config despite the codepath — so RSSM step runs eager at seq_len=32 iterations. `time_pct/wm_rssm_observe` dominates.

### Alternatives, ranked

- **Options / Feudal-HRL (best fit).** Isaac's macro-actions are a discrete, well-defined set: `{clear_current_room, exit_via_door_slot_k, purchase_pedestal_j, use_active_item, use_card_i, bomb_wall}`. Each has a natural termination predicate (room cleared, entered new room, coin count decreased, active-charge reset, etc.). A high-level policy selecting among 6–10 options + low-level per-option controllers reduces the credit-assignment horizon from ~2000 ticks/episode to ~10 option-invocations. This is a 200× reduction. Feudal Networks (Vezhnevets 2017), HAAR, or Option-Critic all apply — Option-Critic is closest to the current PPO/Dreamer stack.
- **MuZero.** Better than Dreamer for Isaac because: (i) MuZero plans in latent space with **explicit MCTS**, which handles Isaac's item-choice branching directly; (ii) it uses **categorical value** with a learned rescaling (`h(x) = sign(x)(sqrt(|x|+1)-1) + eps·x`) that natively handles the r_beat_mom-scale extremes without clipping; (iii) MuZero's value bootstrap is via search, not a critic that hallucinates the same reward forever. Downside: MCTS at 15 Hz is more expensive; unclear it fits the 3060 Ti.
- **DreamerV3 with an options head on top.** Cheapest change: keep the WM (it works), replace the flat MultiDiscrete(9,5) actor with a two-level actor `(option ∈ {6-10}, primitive ∈ 45)` and terminate imagination on option-termination predicates. `imag_horizon` becomes option-count, not tick-count.
- **MCTS + heuristic scoring on the WM latent.** Since dynamics are learnable, use the WM as a simulator and search primitive-action sequences with a heuristic scoring (e.g., predicted `new_room` + predicted `+kill`). Essentially "Dreamer as world simulator, not as actor."

### One thing to keep from Dreamer

The reconstruction-based WM is genuinely useful for Isaac: enemies + projectiles + player state all have structure, and self-supervised prediction pretrains a representation without reward. Don't throw the WM out — throw the actor out and replace with search or options.

---

## Proposition B — Obs gaps

**Position: STRONG AGREE. The obs is missing so much state that a human could not clear Basement I with only what the WM sees. The agent is playing Isaac through a keyhole.**

Grepped in `spaces.py`, `obs.lua`, and the `_PLAYER_FIELDS`/`_GLOBAL_FIELDS` tuples. Categorised missing state:

### Missing player-side state
| Field | Impact | Where it should live |
|-------|--------|---------------------|
| **Active item charge** (0/N, N/N ready) | Agent has `use_item` reward but no visibility of whether the space bar will work. **`use_item` fired 0 times in the 40h run.** Not surprising: agent has no `is_ready` signal to condition on. | `player.active_charge`, `player.active_max_charge` |
| **Active item ID** | Isaac has 200+ actives; each triggers vastly different behavior. Agent has no idea which one it holds. | `player.active_item_id` (one-hot into ACTIVES_K table) |
| **Trinket ID + gulped trinkets** | Trinkets modify combat/economy heavily (Mysterious Paper, Cursed Skull, Small Rock, etc.). Not observed at all. | new field |
| **Card / pill slots** | 2 slots (or 8 with Starter Deck), each with distinct effects. Not observed. | new fields |
| **Transformation counters** (Guppy, Beelzebub, Fun Guy, Leviathan, Bookworm, Adult, Spun, Mom, Yes Mother, Conjoined, Seraphim, Bob, Spider Baby) | 12+ transformations, each triggered at 3 items of a class. Currently the agent cannot see "you are 2/3 toward Guppy" — a decision that a human uses to decide whether to take a fly item over a spider item. | Level-1 API: `player:GetPlayerFormCounter(form_id)` for each form |
| **Item stack counts** | `passives` is a **presence-only** MultiBinary(256). It can't distinguish "1 copy of 20/20" (tears-only) from "2 copies of 20/20" (no stack — needs Ludo). It also can't distinguish "Sad Onion picked up 1×" from "Sad Onion + Blood of Martyr + Cricket's Head" (same shape, different effects). | Replace MultiBinary(256) with `int8[256]` count vector |
| **Passives table only covers 256 of ~730 collectibles** | `tables.lua` curates "top 256" for basement/caves. Anything outside collapses to bucket-0. Once past floor 2, most items are invisible. | Expand to ~730 collectibles, or use a learned embedding indexed by ID |
| **Effects / temporary buffs** (Book of Belial active, tarot card effect duration, Damocles sword, Mama Mega, etc.) | `player.Damage` reflects some but not effect duration/timers. | new fields |
| **Currently-held pickup queue** (crown/wisp/orbital positions) | Occupy space, deal damage, block projectiles. Currently the mod filters entities via `IsVulnerableEnemy()` so allies are dropped entirely. | Add `familiars` entity group |

### Missing global / meta state
| Field | Impact |
|-------|--------|
| **Minimap** | Only `visited_rooms` (int scalar). No layout topology, no "boss room location," no "how many unexplored rooms." Humans use minimap to *plan* movement — agent literally cannot. |
| **Room-clear count / total** | `is_clear` (bool) but no "rooms cleared this floor" or "rooms remaining." Agent cannot compute floor-progression fraction. |
| **Boss room found flag** | Isaac's minimap reveals boss room only after adjacent-room entry. Agent has no equivalent signal. |
| **Item pool state** | Treasure-pool draws without replacement. Once taken, the pool shrinks. Not observable; nothing tracks "shop pool empty." |
| **Damocles / Curse of Blind / Curse of Maze active** | `curses` is a scalar bitmap. Individual curse effects (Blind hides item pedestals) are strategic — agent gets a bit, not a semantic feature. |
| **Devil deal / Angel deal availability** | Post-boss room has a chance of Devil door based on hidden state (deals-since-last, damage-taken flags). Agent can't predict door type. |
| **Door target room type** — mod only encodes {BOSS, TREASURE, SECRET} at `doors` (obs.lua build_doors, 6-slot vector). Missing: shop, arcade, curse, sacrifice, devil, angel, chest, dice. Room-choice signal is 3-of-15 room types. |
| **Boss HP / phase** | `enemies_feats[6]` is per-entity `hp/max_hp`, up to MAX_ENEMIES=24. Multi-entity bosses (Larry Jr = 4 segments, Chub = 3 segments, Blastocyst = split cascade) don't show "phase 1/3." |

### Missing entity-side state
- `ENEMY_FEATS=16` but only 15 fields written in `build_enemies`; slot 16 is "reserved." Real Isaac attributes that would help: `EntityCollisionClass`, `HitFrame`, `ProjectileParams` for shooters, target-of-attack, is-in-attack-animation-state. Currently agent sees position + velocity + hp + type + state-int but not "is this Gaper about to lunge?"
- **Projectiles are typed by `Variant`** (numeric int) but not by ProjectileType (BLOOD, TEAR, FIRE, BONE — different behaviors). The variant is exposed at index 8; the model must learn a numeric-int-to-behavior mapping from scratch with no priors.
- **Laser lifetime** — `FrameCount` is included but no `Timeout` or "will this laser disappear in N frames." Agent must learn duration statistically.

### Could a human play with just our obs?
**No.** A skilled human relies on: (i) minimap layout, (ii) active-item charge, (iii) trinket ID on HUD, (iv) card/pill preview, (v) transformation-progress icons above the health bar, (vi) door-target-room icons (mapped from adjacent-room visits), (vii) damage-taken feedback ring. **We provide (vii) only, partially.** For empirical evidence: with the current obs a *human* watching only the numeric state vector cannot decide whether to take a Devil deal — the deal's terms (item quality, HP cost) aren't observable. If the human can't play, the agent can't either.

### Minor concern: obs decode bug
`_decode_passives` reads a 1-based sparse index list, decrements, and clips to `PASSIVES_K=256`. Any collectible whose `Tables.COLLECTIBLES` dense index is 0 (e.g., unrecognised items — the mod maps unknown IDs to 0) *never* fires the bitmap. Confidence: 90%. See `obs.lua build_passives` — it emits `Tables.COLLECTIBLES[cid]`; if that returns 0 for unknown collectibles, `_decode_passives` does `int(0) - 1 = -1`, fails the `0 <= i < PASSIVES_K` guard, and silently drops it. So "unknown collectible" isn't even represented as an "other" bucket — it's invisible.

---

## Proposition C — Reward shaping vs Imitation Learning

**Position: STRONG AGREE. Continuing the reward-shaping treadmill on Isaac at consumer-GPU compute is a dead-end. Bootstrap with BC on human demos.**

### The evidence that shaping is failing on its own terms

From the 40.8h run (v2 shaping, the run that motivated v3):
- `loss/actor_entropy = 3.804` versus theoretical uniform of `log(9)+log(5) = 3.807`. **Policy is 99.92% uniform after 40 hours.**
- `rollout/ep_reward mean = 9.673`. Of which:
  - `seek_door` mean = **+6.338 (65.5%)**  — the reward-hack the postmortem named.
  - `new_room` mean = +2.361 (24%) — pumped as byproduct.
  - Everything else = ~+1 combined.
  - **80% of episode reward flows from two shaping terms** — and neither is a real objective.
- `behavior/floors_reached_max = 0` — never left Basement I in 40h.
- `behavior/boss_kills = 0`. `behavior/use_item = 0`. `behavior/keys_used = 0`. `behavior/bombs_used_max = 1` (once, ever).

The v2 → v3 patch fixed the specific `seek_door` pump (per-episode cap of 1.5). It does not fix the pattern: **hand-crafted dense reward + a policy stuck at max entropy will always find whichever shaping term correlates with easy accumulation.** Every past cycle has produced a new hack:
- v0: bot idled in corner → `r_idle_penalty` added
- v1: bot wiggled to escape idle → `r_stationary_penalty` added
- v1.5: bot oscillated between two cleared rooms → `r_backtrack` added
- v2: `clear_idle_extra` at -0.03 became -2.83/ep policy-independent bias → 6× reduction + grace
- v2: `seek_door` pumped 6.34/ep, only 37% new_room → v3 per-ep cap
- v3: ? (**history says another hack will emerge**)

### Precedent

- **NLE / NetHack**: The strongest published results use behavioral-cloning on `AutoAscend`-style bots + RL fine-tune (Hambro et al. 2022, "Dungeons and Data"). Pure reward-shaped RL plateaus in the 1st dungeon level.
- **StarCraft II / AlphaStar**: SL pretrained on ~1M human replays. Without that bootstrap, self-play from scratch never reaches human grandmaster.
- **Dota / OpenAI Five**: Scale (152k CPU-years) + dense reward. Not reproducible on a 3060 Ti.
- **Roguelike / procedural games with sparse reward + human demos beat RL-from-scratch consistently.**

### Why Isaac is ideal for BC

- **Abundant public demo data.** Northernlion has ~10,000 recorded Isaac runs. Hutts, ThePlushGiant, DanRykerNL, tazek dozens of hours each. YouTube + Twitch VODs.
- **Deterministic seed replay.** Once a seed is fixed, Isaac's floor layout is reproducible — a recorded run can be re-run in the mod to log **exact obs → action pairs** in your schema, rather than trying to OCR the screen.
- **Small action space (45 combos).** BC on MultiDiscrete(9,5) is trivial; a 2-layer MLP fits in <5 min on the same 3060 Ti.
- **The obs is already collected by the mod.** No new instrumentation needed — just play the game with recording on.

### Counter-arguments and rebuttal

- *"Human obs ≠ our obs (humans see pixels)."* True in general; irrelevant here — as noted in Proposition B, we should be expanding the obs anyway. Once we expose active-charge, transformations, and door types, the obs is a *superset* of what a human uses functionally. And humans can be asked to play with a "training" HUD that shows the exact obs vector — the resulting policy is BC-compatible.
- *"We need the RL agent to discover novel strategies."* Fine — do BC to bootstrap → then RL fine-tune. AlphaStar did exactly this. Novelty emerges from fine-tuning, not from cold-start. Cold-start on 15Hz Isaac with ~2 env-instances is compute-infeasible.
- *"BC quality is capped by demonstrator skill."* Not a problem for our current bar (Stage-1 room clearing). Any human ≥ our current agent.

### Deceptive angle: RND intrinsic is dead and nobody sees it

`rnd/predictor_loss`: first=1.27e-3, last=6.80e-6, min=3.88e-6, mean=1.53e-5. **RND intrinsic reward is effectively zero for 99% of training.** The v3 patch (train predictor on real feats, not imagined) is correct in principle but even in the file `intrinsic_mean_raw` and `intrinsic_ema_mean` are ~5e-6 — smaller than the actor-entropy noise floor. **Exploration is not happening.** The RND was silently a no-op for the whole run and the TB dashboard shows no red flag because everyone looks at `rollout/ep_reward`, not `rnd/*`.

---

## Proposition D — 15Hz control rate

**Position: MILD AGREE that 15Hz is a problem — but not for the "misses tears" reason. The real bottleneck is that 15Hz combined with `seq_len=32`/`imag_horizon=20` gives a temporal window of 1–2 seconds, which is 20–60× shorter than the game's macro-action timescales.**

### The "misses tears" argument is weaker than the doc suggests
- Isaac's tear speed at default `ShotSpeed=1.0` is ~300 world-units/s. Room dimension is 480×270; a tear crossing the room takes 1.0–1.6 s = 15–24 ticks at 15Hz. That's plenty of observation for dodging.
- Isaac's *engine* runs at 30 update-Hz but internal physics interpolates smoothly (see `PositionOffset` we do read). Frame-skip of 2 for the RL controller is comparable to OpenAI Five (7.5Hz control on 30Hz engine).
- Empirical: Dota, Atari, NetHack all use 4-frame skip or coarser without projectile-avoidance collapse.
- Where 15Hz DOES hurt: Mom's Foot stomp (~15-frame telegraph = 1s), Fatty spike volleys with 3-frame windups. Some attacks are close to unavoidable at 15Hz control. Not a killer.

### The temporal-horizon argument is much stronger
- `seq_len = 32 ticks × (1/15 Hz) = 2.13 s` — the training window over which the RSSM must do temporal credit assignment.
- `imag_horizon = 20 ticks × (1/15 Hz) = 1.33 s` — the actor's imagination window.
- Isaac macro-action timescales:
  - Enemy dodge: 0.3–1s ✓ fits
  - Room clear: **5–20 s** — 4–15× the imagination window
  - Cross a door: 3–5 s including animation → 2–4× window
  - Full floor traversal: 60–120 s — 50–90× window
  - Boss fight: 30–60 s — 20–45× window
  - Shop transaction: ~15s from door to purchase → 10× window
- With `imag_horizon = 20` and `gamma=0.999`, the effective discount horizon is 1000 ticks — but the model **rolls out only 20 imagined steps**, then bootstraps by critic. So the value target for anything beyond 1.3s is *whatever the critic hallucinates*. Combined with `cont ≈ 1` almost everywhere, the target inflates: `loss/actor_target_mean = 47`. **This is not a genuine value estimate; it's a runaway.**

### Fix — cheap and expensive versions
- **Cheap**: Keep 15Hz control. Bump `seq_len 32 → 64` and `imag_horizon 20 → 60`. Memory cost ~2× on WM training; effective planning covers 4s. Still short for a boss but adequate for room clearing.
- **Expensive**: Move to a hierarchical policy (options; see Prop A). Option-level imagination collapses the horizon problem: 5 option-steps cover a room-clear.
- **Wrong fix**: Going to 30Hz control doubles compute cost and *shortens* the physical-time window of a fixed `seq_len`. Don't do it.

---

## Proposition E — Single-room curriculum

**Position: STRONG AGREE. `stage1_single_room_xs` is a curriculum-of-one that cannot possibly teach the game.**

### What the config actually does

`configs/stage1_single_room_xs.yaml`:
- `reset_stage: 1` — always Basement I.
- `max_episode_steps: 1800` (2 min at 15Hz).
- **No curriculum controller. No schedule. No graduation criterion.**

There is a `curriculum.py` file in `python/isaac_rl/`, but the Dreamer trainer never invokes it (grep it: it's used by the PPO path). Dreamer runs a single environment factory that always resets to stage 1.

### What the curriculum silently omits

| Skill | Where in Isaac | Present in curriculum? |
|-------|----------------|------------------------|
| Enemy targeting | Any room | ✓ |
| Basic dodging | Any room | ✓ |
| Room clearing | Any room | ✓ |
| Door choice | Multi-room floor | ✗ (never seen) |
| Boss fight | End of floor | ✗ (never entered — `boss_kills_max = 0` in 40h) |
| Item pedestal decision | Treasure room | ✗ (`treasure_first_entry` fires 21% of eps but nothing follows) |
| Shop economics | Shop room | ✗ (`shop_first_entry` mean = 0.027) |
| Active-item usage | All floors | ✗ (`use_item` fired 0× in 40h) |
| Card / pill usage | All floors | ✗ (no action head — has been removed from action space!) |
| Bomb-for-secret-room | All floors | ✗ (`secret_first_entry` = 0) |
| HP resource management | Cross-floor | ✗ (episode ends before matters) |
| Floor progression | Multi-floor run | ✗ (`floors_reached_max = 0`) |

**Confirmed: the curriculum trains 4 of 12 skills.** The other 8 are cited as objectives in the reward config but the agent never has a chance to practice them.

### Ancillary problem: episode termination pathology
`rollout/ep_end_mod_restart_frac = 0.97` — **97% of episodes end because Isaac's mod restarts the run**, not because the shaper terminated (`shaper_terminated_frac = 0.003`) or the episode truncated at 1800 (`truncated_frac = 0.027`). This means the terminal signal the agent learns from is not "died from HP=0" or "cleared a stage" but "mod handshake reset."

The v3 patch to `finalize_episode` fires the aggregate outcome bonuses on `mod_restart` too — good — but that still delivers `depth_end_bonus = 0` every single time (because floor never advances) and `survival_end_bonus` gets awarded to random restart events regardless of whether the agent was doing well. **The episode boundary carries almost no semantic content.**

### What a real curriculum would look like

1. **Phase 0 (BC only, 5–10h wall clock)**: Behavior cloning on 20–50h of human demo across Basement I–Depths II. No RL. Verify BC actor beats current agent on ep_reward.
2. **Phase 1 (RL fine-tune on room-clear rooms, ~24h)**: Isolated rooms with varied enemy sets from Basement I–Caves II. Learn to clear.
3. **Phase 2 (full-floor navigation)**: Full stage-1 floor. Reward: reach the boss room; no boss fight yet.
4. **Phase 3 (boss fights)**: Boss-only rooms. Warmed-up actor learns the specific dodge patterns.
5. **Phase 4 (unrestricted stage 1)**: Full run through Basement I. Graduation criterion: >30% floor-clear rate.
6. **Phase 5 (multi-floor)**: Enable stages 2–6 progressively.

The current design starts at "Phase 5 lite" (single room from a stage-1 pool) and never advances.

---

## Non-obvious ways the model deceives us

**(Required by mandate.) Listed in decreasing sneakiness.**

1. **`ep_reward_best` is not a learning curve — it is a coefficient-tuning curve.** `rollout/ep_reward_best` climbs from 3.96 to 76.7 over 40h. This looks like progress. But `actor_entropy=3.804` proves the policy is uniform. The climb is because (a) `r_new_room 0.2 → 5.0` was pumped mid-run, (b) `seek_door` at 6.34/ep was a design bug — a random policy accumulates it. The reward-best curve reflects the reward *function* changing, not the *policy* changing. **Anyone glancing at TB thinks the agent is learning; it isn't.** Confidence: 95%.

2. **The critic hallucinates a `V ≈ 47` everywhere and the actor gets a whitened advantage of `|adv| ≈ 0.037`, which the reward-EMA further compresses.** `loss/actor_target_mean=47, std=25` while `loss/actor_adv_abs_mean=0.037`. The advantage is 1000× smaller than the target because the target is uniform-inflated by `seek_door`-pump baseline. The actor gets no useful signal at all — every action looks equally-valued after normalization. **Symptom you'd never suspect from ep_reward alone.** Confidence: 90%.

3. **RND is a silent no-op.** `rnd/predictor_loss = 6.8e-6` and `intrinsic_mean_raw = 6.8e-6`. Multiplied by `rnd_intrinsic_scale=0.1`, intrinsic reward is ≈ 6.8e-7 per step — 5000× smaller than `r_step=-0.003`. Exploration curiosity is completely swamped. The v3 patch (train on real feats) is directionally right but the effective scale is still ~zero because `rnd_intrinsic_scale=0.1` was tuned to a predictor loss of `~1e-3`, not `~1e-5`. **Nobody has re-tuned the RND scale after v3.** Confidence: 90%.

4. **Symlog-disc value head clips r_beat_mom at +20.** `value_v_min=-20, value_v_max=20`. `r_beat_mom=+50`. The critic *literally cannot represent* the biggest reward in the game. The agent has no incentive-gradient to pursue Mom because the value function saturates at +20 well before Mom's contribution matters. Confidence: 99% (mechanical).

5. **`loss/kl_free_bits_frac = 0.85` — 85% of KL elements are above the free-bits floor.** This is *neither* fully binding nor fully slack — but it means the KL is close to saturating the free-bits budget across most of the latent. Given passives is a 256-bit binary vector and the discrete stoch latent is 32×32 (log₂32 = 5 bits per position × 32 = 160 bits of encodable capacity), **the WM is at capacity trying to compress passives + enemies + projectiles into 160 bits.** This will silently degrade rare-obs prediction (item pickups, boss appearance) in favor of common obs (empty rooms). Confidence: 70%.

6. **`n_envs=2`.** Two environment instances is a diversity floor. Even with `train_ratio=4`, replay-buffer diversity is stunted. The v2 postmortem notes SPS went from 21 to 2.36 mid-run — that's per-env compute, but even at peak the effective sample throughput is limited by n_envs=2. Nobody has raised n_envs because Isaac is heavy to launch, but this is silently starving the training. Confidence: 80%.

7. **`beh_room_types_seen` mean = 2.17, max = 3, and `_max=3` all-run**. Only 3 room types ever entered across 40h × ~1000 episodes: `1=Default`, `4=Treasure` (21% of eps first-entry fires), `10=Curse` and `13=Sacrifice` sporadically. Zero shops, boss rooms, secret rooms, arcades, devil, angel — despite reward being wired for all of them. The 12 room-type first-entry rewards are shaping terms **for events that never occur.** Confidence: 100%.

8. **`clear_idle_grace_ticks=30` at 15Hz = 2 seconds.** Then `clear_idle_extra` starts penalizing at -0.005/tick. If the agent enters a cleared room, 15Hz × 2s = 30 ticks grace, then -0.005 × ~200 ticks = -1.0. `reward/clear_idle_extra mean = -0.207/ep`. This is small but **it means the agent is still penalized policy-independently for existing in a cleared room** — even at v3's 6× reduction. Confidence: 100%.

---

## Top 3 fundamental changes I would make (ranked)

### 1. Kill the reward-shaping cycle. Pivot to BC-bootstrapped RL. _[highest impact / lowest risk / fastest wall-clock]_

**Rationale.** The postmortem cycle (v0 → v1 → v1.5 → v2 → v3, each finding a new hack) is a Sisyphean loop. Meanwhile the 40h run's policy is provably uniform-random. The compute budget of a consumer GPU + 2 envs cannot support cold-start RL against Isaac's complexity — precedent (NetHack, StarCraft, Dota) says so.

Concretely:
- **Week 1**: Instrument the mod to record obs+action tuples during human play, saved to disk with seed. No new obs schema — reuse `spaces.py`. Have 2–3 humans (skilled Isaac players) run 20–50h of Basement I → Womb runs.
- **Week 2**: Train a BC actor (same MultiDiscreteActionHead architecture as `MultiDiscreteActionHead` in `dreamer/action.py`) on the human data. Loss = cross-entropy on action distribution given obs. Should fit in 4h on the 3060 Ti.
- **Week 3**: Initialize the Dreamer/PPO actor from BC weights. Fine-tune with RL. Reward function can be MUCH simpler — remove all the door-seeking/idle/PBRS/stationary shaping. Keep only: `r_step`, `r_kill`, `r_room_clear`, `r_new_room`, `r_pickup_collectible`, `r_beat_mom`, `r_death`. That's it.

**Expected outcome.** BC alone will beat the current 346k-step agent on `rooms_visited`, `boss_kills`, `use_item`, `keys_used` — probably >10× on each. Then RL fine-tune finds novel strategies over that baseline.

**Risk.** Requires human demo data collection. Time cost ~40–100h of human play. Trivially recoverable if quality is bad — human replay is cheap.

---

### 2. Fix the obs. Add active-charge, transformations, item counts, minimap, door-target-types, cards/pills/trinket. _[highest impact / low risk / moderate wall-clock]_

**Rationale.** As Prop B shows, the agent literally cannot perceive half of what a human uses. No amount of algorithm improvement fixes this; it's an information bottleneck.

Priority order (add these fields to `obs.lua` and `spaces.py`):
1. `player.active_charge`, `player.active_max_charge`, `player.active_item_id` (embedding) — unblocks `r_use_item`.
2. `player.transformations[12]` (float, 0..3, counter per form) — unblocks item-planning.
3. `passives` count vector (int8[256]) instead of MultiBinary — unblocks stack awareness.
4. `player.trinket_id`, `player.card_slots[N]`, `player.pill_slots[N]` — unblocks non-passive-item strategy.
5. `doors[i].target_room_type` full enum (not just BOSS/TREASURE/SECRET) — unblocks room-choice.
6. `minimap.room_grid` (13×13 grid, per-cell type+visited-flag) — unblocks navigation planning.
7. `familiars` entity group — orbital positions.
8. Extend `PASSIVES_K` from 256 to ~730 (all Repentance collectibles) or use a learned embedding.

**Expected outcome.** The BC step (change #1) becomes strictly stronger because demo obs now captures what the demonstrator was reacting to. RL fine-tuning gains `use_item` and `bombs_used` as genuinely learnable behaviors.

**Risk.** Requires re-recording the human demos if #1 is done first. So do #2 *before* #1 records.

---

### 3. Move to a hierarchical actor. Options-over-primitives, imag_horizon in option-steps. _[medium-high impact / medium risk / higher wall-clock]_

**Rationale.** The temporal-horizon problem (Prop D) and the "never entered a boss room" problem (Prop E) share a root cause: the agent operates on a flat 15Hz primitive action space where meaningful outcomes are 200–1000 ticks away. Options collapse this to ~5–20 option-steps per meaningful event.

Concretely:
- Define an option set: `{move_to_door_slot_k (k∈4), clear_current_room, purchase_pedestal_j, use_active, drop_bomb, use_card_i, wait}`. ~10 options.
- Each option has a **termination predicate** (in Python, checked each tick): e.g., `move_to_door_slot_k` terminates on entering the target room or after 300 tick timeout.
- **High-level actor** picks options; runs at ~1Hz effective (once per option termination). Trained with imagination at option resolution — `imag_horizon = 20 options ≈ 20 × 5s = 100s of game time` covering full room clears.
- **Low-level controllers** per option: mostly hand-implemented (move_to_door = A*-toward-door heuristic) for speed, or shared learned intra-option policy trained by BC + RL.
- Compatible with Dreamer WM: WM stays flat, high-level policy queries the WM at option boundaries.

**Expected outcome.** Boss rooms are now reachable in the first 10 minutes of training (option `move_to_door_slot_k → clear_current_room` chains). Floor progression becomes a training signal.

**Risk.** Requires defining the option set and termination predicates correctly. If options are wrong (e.g., too coarse), the high-level actor is left with a bad action space. Iterate — start with 6 options, expand.

---

## Confidence & gaps

- **Very high confidence** (>90%): The `actor_entropy=3.804` + `ep_reward_best↑` deception, the `seek_door` reward-fraction, the missing obs enumeration in Prop B, the `curriculum-of-one` critique, the RND-silent-no-op, the value-head clipping of `r_beat_mom`.
- **High confidence** (70–90%): That Options-HRL is strictly better than flat Dreamer for Isaac; that BC bootstrap will beat cold-start RL by ≥10× on rooms_visited; that `n_envs=2` is under-provisioned.
- **Medium confidence** (50–70%): That MuZero > Dreamer for Isaac specifically (depends on MCTS wall-clock feasibility on 3060 Ti — I haven't measured); that expanding `PASSIVES_K` to 730 helps more than a learned embedding of 256 (either could work).
- **Low confidence** (30–50%): The exact impact of the `_decode_passives` unknown-item silent drop bug (Prop B, minor concern) — need to instrument the mod to check how often `Tables.COLLECTIBLES[cid]` returns nil for real Repentance items.

**Gaps** (things I couldn't verify without running code):
- Actual per-episode variance of `seek_door` at v3's `max_seek_door_reward_per_episode=1.5` cap. Prediction: agent finds another shaping term to pump; without a new run there's no data.
- Whether the mod-restart events are hiding actual death signals or genuinely restarting on crash. The `rollout/ep_end_isaac_crash_frac = 0` suggests it's not crashes — so what *is* the mod_restart cause? Worth a fork/instrumentation pass on env.py / bridge.
- Whether the WM's `loss/passives` head is actually learning items (specific per-key loss wasn't in the sanity print). If passives loss is high, the WM never actually encodes items, which invalidates using it for item-planning downstream.

_End of audit._
