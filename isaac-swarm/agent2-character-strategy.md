# Agent 2 — Character-Specific Playstyle Differences

## Current state: character context

**Reset command (main.lua ~L260-268):** The mod does NOT call `restart 0`. It calls bare **`restart`** (equivalent to pressing R), which restarts the *current* character in place. The task brief's premise — "line 261 calls `restart 0`" — is stale. The relevant comment in the file:

```
-- Use bare `restart` (equivalent to pressing R in-game), NOT `restart 0`.
-- Difference: `restart` restarts the CURRENT run in place ... `restart 0`
-- says 'new run as character 0 (Isaac)' and in some Repentance builds
-- triggers a deeper teardown path that ends the process entirely ...
-- Since we already boot as Isaac via --set-stage=1, we don't need to
-- respecify the character on every reset — pressing R is enough.
```
So the *first* run's character comes from whatever Isaac's save file / launcher selected (the comment claims Isaac, but `--set-stage=1` controls stage not character — real character selection depends on save state). Every subsequent reset preserves whichever character the game booted as. In practice the empirical assumption is Isaac (character id 0). `restart <N>` where N ∈ 0..40 is the documented way to select a specific character ([IsaacDocs debug console](https://wofsauge.github.io/IsaacDocs/rep/tutorials/DebugConsole.html)).

**Reward function character-awareness:**
- `reward.py::finalize_episode` hardcodes `max_hp = 6.0  # Isaac baseline`. Comment says "Could be dynamic later." Wrong for every non-standard-HP character (Blue Baby, Lost, Forgotten, Keeper, ???, J&E, Bethany, Tainted variants).
- HP-based death detector correctly gates on obs-reported `max_hp > 0`, so it works for characters with soul-only or bone HP — but the shaper's `hp_delta_red / hp_delta_other` split treats *soul + black* as "other" and *red* as "red". No handling for **coin hearts (Keeper), bone hearts (Forgotten), broken hearts (Tainted Magdalene), rotten hearts**. Currently coin/bone hearts do not appear in the obs at all (`hp_red / hp_soul / hp_black / hp_max` only in `_PLAYER_FIELDS`).
- Action space (`spaces.py`): `MultiDiscrete([9, 5])` — move + shoot only. The commit history in `reward.py` shows use_active / drop_bomb / pill_card action heads were REMOVED on 2026-07-02 to shrink the action space. This kills any character whose kit is built on the active-item slot: **Judas (Book of Belial), Lilith (Box of Friends, mandatory — she has 0 damage), Isaac (D6), Bethany (Book of Virtues), Eve (Razor Blade), Magdalene (Yum Heart), Cain (Lucky Foot is passive OK), Samson (Bloody Lust is passive OK)**. Lilith is essentially unplayable without space bar.

## Character taxonomy summary

Values sourced from [bindingofisaacrebirth.wiki.gg](https://bindingofisaacrebirth.wiki.gg/wiki/Characters), [platinumgod.co.uk](https://www.tboi.com/), [isaacguru.com](https://isaacguru.com/).

| ID | Char | Starting HP (half-hearts) | Base dmg × mult | Starting kit | Key mechanic | Strategy category |
|---|---|---|---|---|---|---|
| 0 | **Isaac** | 3 red containers (6/6) | 3.5 × 1.0 | The D6 (active) | Baseline. D6 rerolls pedestals. | Balanced / reroll snowball |
| 1 | Magdalene | 4 red containers (8/8) | 3.5 × 1.0 | Yum Heart (active, heals 1 red) | Tank, self-heal | HP-buffer aggression |
| 2 | Cain | 2 red containers (4/4), +1 key | 3.5 × 1.0, luck +1 | Lucky Foot (passive: luck up) | Fires from one eye; luck up | Coin/luck exploiter |
| 3 | **Judas** | 1 red container (2/2), +3 coins | 3.5 × **1.35** | Book of Belial (active: temp dmg up) | Highest damage mult; devil-deal focused. Dies with Birthright → **Black Judas** (2 black hearts, ×2.0 dmg) | Glass-cannon devil |
| 4 | ??? (**Blue Baby**) | 3 soul hearts (6/6 soul), 0 red cap | 3.5 × 1.05 | The Poop (active) | **Cannot gain red hearts** — HP-up → soul hearts. Devil deals cost soul at red-equivalent price. | Soul-only, permanent devil |
| 5 | Eve | 4 red containers (2 empty + 2 full = 4/8) | 2.63 × 0.75 | Whore of Babylon (passive: at ≤1 heart, +dmg/+speed), Dead Bird (passive), Razor Blade (active) | Rewards *staying low HP*; opposite of HP-preservation reward | Low-HP berserker |
| 6 | Samson | 2 red containers (4/4) | 3.08 × 0.88 | Bloody Lust (passive: dmg up on damage taken, resets on room exit) | Same low-HP incentive as Eve | Damage-scaling berserker |
| 7 | **Azazel** | 3 black hearts (6/6 black) | 3.5 × 1.0, **short-range Brimstone** | Brimstone (passive built-in), flight | Fires a **short beam** at 8 hits/s instead of tears; must be close; flies over pits | Melee-brimstone rusher |
| 8 | Lazarus | 3 red containers (6/6), Anemic pill | 3.08 × 0.88 | Lazarus' Rags (passive: on death, revive as **Lazarus Risen** with +stats) | Two-lives; first death is planned | 2-life mid-game gambler |
| 9 | **Eden** | Randomized 1–5 red (varies) | Randomized every stat | 1 random active + 1 random passive + optional trinket/pill/card; RNG per Eden token | **Every run is different** — reward must generalize | Wildcard / policy-diversity |
| 10 | **The Lost** | **0 HP total** (no red/soul/black/bone cap) | 3.5 × 1.0, flight, spectral tears | Holy Mantle (Repentance buff), D4 (active) | **Dies from any hit**. Cannot gain HP at all. Devil deals free. | Perfect-dodge / no-hit only |
| 11 | Lilith | 4 black hearts (8/8 black) | **0.0 base damage** | Incubus (passive familiar that shoots for her), Box of Friends (active, mandatory), Cambion Conception (passive) | She literally can't shoot; Incubus does damage. Space bar doubles familiars. | Familiar-summoner |
| 12 | **Keeper** | **2 coin hearts** (max 3 coin hearts) | 3.5 × 1.0, ×4 shot speed, ×0.75 speed | Wooden Nickel (active) | HP = coins. Damage taken drops 1 coin heart. **Heals only from picked-up coins.** Killing enemies drops coins. | Coin-farming, kill-for-heal |
| 13 | **Apollyon** | 3 red containers (6/6) | 3.5 × 1.0 | Void (active, absorbs items → permanent stats or charge) | Rewards picking up bad items to void | Item-eating snowball |
| 14 | **The Forgotten** | 4 bone hearts (max 8) + Soul (2 soul hearts, floats behind) | 4.72 (Forgotten melee) / 2.73 (Soul) | Bone club (melee swing/throw) + spectral-tear Soul; toggle with space (`SwitchPlayer`) | **Cannot fire regular tears as Forgotten.** Melee only; Soul is ranged. Red hearts convert to bone hearts. | Dual-form melee/ranged |
| 15 | **Bethany** | 3 red containers + **4 soul charges** (soul hearts here are ACTIVE-ITEM AMMO, not HP) | 3.5 × 1.0 | Book of Virtues (active: spawns wisp familiars per use) | Soul hearts fuel active-item uses; picking up soul heart adds charge, not HP | Wisp-army / active-spam |
| 16 | **Jacob & Esau** | Jacob 2 red + 1 soul, Esau 2 red + 1 soul (**two players simultaneously**) | Jacob 2.75, Esau 3.5 | None | Controls **both characters** at once; death of either ends the run | Dual-body control (breaks single-player obs) |

Repentance also has 17 Tainted variants (IDs 21+) with even more radical kits (Tainted Lost = 0 HP no Holy Mantle, Tainted Lazarus = swap on death and lose active items, Tainted Keeper = 3 keepers, etc.) — omitted here.

## Why "restart 0 = Isaac" may be right or wrong (both sides)

**Note:** The mod code as of the current commit does NOT use `restart 0`. It uses bare `restart`. The question below applies if we *were* to switch back to `restart 0` (or equivalently, if the game boots into character 0 initially).

**Right — Isaac (id 0) is a defensible training target:**
1. **Balanced kit.** 3 red containers, no exotic HP type, no mandatory active-item usage. Matches the obs vector's `hp_red / hp_soul / hp_black / hp_max` schema without gaps. Reward's `max_hp=6` matches exactly.
2. **Universal mechanics.** Isaac uses regular tears (matches the shoot action head), no melee, no forced flight/spectral. Damage/dodge fundamentals learned on Isaac transfer to Magdalene, Cain, Judas, Lazarus, Apollyon, Bethany with minimal reward changes.
3. **No unlock gate.** Isaac is available in a fresh save; other IDs need unlocks (Judas after Satan kill, Blue Baby after 10× Mom's Heart, Lost after specific death sequence). A vanilla mod install can guarantee id 0 works.
4. **Standard damage/HP calibration.** Death reward, HP-preservation reward, damage-taken splits (red vs soul) all designed against 6-HP baseline. Isaac hits this exactly.
5. **Community baselines exist.** Speedrun/mod communities use Isaac as reference for "average difficulty" runs. Any RL comparison numbers should assume Isaac.

**Wrong — Isaac-only training has real costs:**
1. **Overfits to red-heart HP semantics.** The `r_full_hp_tick` reward (0.005/tick when `cur_red >= max_hp`) NEVER fires for Blue Baby (0 red cap), Azazel (black only), Lost (0 HP), Forgotten (bone), Keeper (coin), Bethany (soul is ammo not HP). A policy trained only on Isaac will misvalue soul hearts (correctly non-preserved for Isaac; critical for Blue Baby).
2. **Never learns active-item use.** With the action space stripped to `[move, shoot]`, Isaac never presses space. D6 rerolls (Isaac's core mechanic) never happen. Even Isaac's optimal strategy is being under-trained.
3. **Never sees devil/angel decisions calibrated correctly.** Judas/Azazel/Lost want devil deals; Blue Baby's devil deals are half-price in soul hearts; Bethany's angel deals are boosted. Reward has no signal for these tradeoffs.
4. **Randomized-Eden generalization is untestable.** If we ever want a robust policy, Eden's random-stat runs are the natural stress test — Isaac-only training never exposes the policy to `damage=0.85` or `range=8.0` edge cases.
5. **Line 261 semantic ambiguity.** If the mod is ever "fixed" to actually issue `restart 0` (the task brief thought it did), then boot vs reset would use different characters — the first episode is save-file-dependent, subsequent ones are Isaac. Training data becomes non-stationary in a subtle way. Right now the bare `restart` avoids this, but it's fragile: any dev who thinks "let me force Isaac to be safe" will introduce this bug.

## Reward function assumptions that break for non-Isaac characters

Every one of these is a bug for the listed character:

1. **`max_hp = 6.0` hardcoded** (`finalize_episode`, survival_end_bonus). Correct for Isaac, Cain (4), Magdalene (8), Judas (2), Samson (4), Eve (8), Lazarus (6), Apollyon (6). **Wrong for:** Blue Baby (6 soul, 0 red → `total_hp / 6` accidentally works if we sum soul, but obs `hp_max` is 0 → guard skips), Lost (0), Keeper (coin — not tracked), Forgotten (bone — not tracked), Bethany (soul is charge not HP — inflates survival bonus).
2. **`hp_delta_red` = -1.0, `hp_delta_other` = -0.5.** Encodes "red is worth 2× soul." **Wrong for Blue Baby / Lost / Azazel** where non-red IS the full HP pool; **backwards for Eve / Samson** where losing red is *good* (Whore of Babylon / Bloody Lust trigger at low HP).
3. **`r_full_hp_tick` only fires when `cur_red >= max_hp`.** Never fires for Blue Baby (max_hp=0 in obs → guard blocks), Lost, Azazel, Forgotten, Keeper. Removes ~0.005 × 4500 ticks/ep = ~22.5 reward budget from every non-red-heart character.
4. **HP-based death detector gated on `max_hp > 0`.** Correctly disables for Lost (max_hp=0), but this means **Lost never gets a Python-side death signal** — depends entirely on mod-side death event, which the 2026-07-08 postmortem confirms is unreliable. Lost episodes could hang.
5. **Devil / angel deal room rewards.** `r_devil_first_entry = 3.0`, `r_angel_first_entry = 3.0`. Neutral for Isaac. **Way too low for Judas/Azazel/Lost** (devils are core progression). **Positively wrong for Blue Baby** (deals cost half in soul, so entering is nearly free). **Wrong for Bethany** (angel deals are boosted). Reward should scale by character kit.
6. **`r_pickup_heart = 0.5`.** Fires on red-heart increase only (`prev_hp_red_ticks` delta). Blue Baby/Lost/Azazel/Forgotten/Keeper NEVER pick up red hearts (they can't). Massive under-reward for their equivalent pickups (soul hearts for Blue Baby, coins for Keeper, bones for Forgotten).
7. **`r_use_bomb = 0.02`.** Bomb count only visible as `hp_bombs` delta. Fine mechanically, but **The Lost cannot afford to drop bombs (self-damage kills him)** → agent should be penalized for bomb near self, not rewarded.
8. **No coin-heart / bone-heart / rotten-heart obs.** `_PLAYER_FIELDS` in spaces.py has `hp_red / hp_soul / hp_black / hp_max` and nothing else. Keeper's coin hearts likely appear as `hp_red` (they use the red-heart slot) but `max_hp` semantics differ (max 3 coin hearts, not 6). Forgotten's bone hearts likely map to `hp_black` or a separate hidden field. Untested.
9. **Action space has no active-item / space-bar action.** `MultiDiscrete([9, 5])`. **Lilith is unplayable** — she has 0 base damage; her Incubus does damage but Box of Friends is mandatory for boss rooms. **Judas, Bethany, Isaac, Eve, Magdalene lose their core kit.** The Forgotten cannot switch to Soul form (SwitchPlayer is bound to space bar in default controls).
10. **`beat_mom` at stage 6→7 transition.** Correct for Isaac / Magdalene / Cain path. **Wrong path for The Lost** (Repentance encourages ??? or Delirium ending). Reward doesn't scale by "how much of a run this floor represents" per character.

## Recommendation: choose one path, defend it

**Path: "Isaac-only training now, add character-conditioned reward + action space when the base policy plateaus."**

Justification:
1. **Current state is pre-competence.** Per the reward.py comments (2026-07-06/08/09 postmortems), the agent has been stuck for 40+ hours on issues like "never opens doors post-clear," "0/322 episodes used bombs," "reward-hacked seek_door." Adding character variety now adds a *between-episode* non-stationarity on top of an agent that hasn't solved the *within-episode* fundamentals. Every reward-shaping fix so far has assumed Isaac. Changing the character now invalidates all of that tuning.
2. **Isaac is the largest-transferable base.** Skills that generalize (dodging, aim alignment, door navigation, room-clear pacing) are all present in Isaac. Kit-specific skills (Brimstone positioning for Azazel, coin-farming for Keeper, no-hit for Lost) are additive on top. Train the base, then curriculum.
3. **The mod already uses bare `restart`, which preserves character.** Zero cost to keep it that way. Explicitly force `restart 0` in the mod (with the process-death workaround understood) once we're ready to guarantee character determinism.
4. **Character variety as final-phase curriculum, not initial exploration.** Post-plateau, use `restart <id>` with a schedule: Isaac → Magdalene (more HP, same kit) → Cain (fewer HP, same kit) → Judas (dmg mult, active use forced) → Blue Baby (soul-only, tests HP generalization) → Azazel (short-range Brimstone, forces close-combat) → Bethany (active-item spam) → The Lost (no-hit end-game challenge). Skip Lilith/Forgotten/J&E until action space is fixed.
5. **Reward should DETECT the character (via `EntityPlayer:GetPlayerType()` on mod side, sent as `player.player_type` in obs)** and switch reward-config keys. Cheap; doesn't force us to train one policy per character. A single conditional adds a `player_type` int to the obs and the reward reads it. Character-conditioned reward > per-character models for compute reasons.

**Rejected alternatives:**
- **Train on all characters uniformly** — non-stationarity kills PPO/Dreamer sample efficiency. The 40h stalls are already caused by within-run non-stationarity; between-character adds a second dimension.
- **Train per-character separate policies** — 17× compute. Not viable at current budget.
- **Ignore character difference forever** — permanent bug when someone eventually unlocks a non-Isaac save.

## Priority-1 changes for next run (top 3)

1. **Explicitly force Isaac (id 0) in mod boot AND lock reward assumptions to id 0.**
   - Mod: add `Isaac.ExecuteCommand("restart 0")` on the first `MC_POST_GAME_STARTED` if `Game():GetFrameCount() == 0` and player type ≠ 0, then wait for the second GAME_STARTED. Or use `--set-character=0` if the launcher supports it. Document the design decision so future contributors don't get surprised.
   - Emit `player.player_type` (from `EntityPlayer:GetPlayerType()`) in every obs. Add to `_PLAYER_FIELDS` in spaces.py.
   - Python-side assertion: `assert obs.player.player_type == 0` in env reset until multi-character phase.
   - Effort: <1 hr. Blocks: nothing. Prevents silent breakage.

2. **Replace `max_hp = 6.0` with dynamic obs.player.hp_max, and drop the survival bonus if `hp_max <= 0`.**
   - `reward.py::finalize_episode`: `max_hp = float(self.state.prev_hp_max) if self.state.prev_hp_max > 0 else 6.0`. Track `prev_hp_max` in `RewardState` alongside `prev_hp_red`.
   - Same fix in the HP-based death detector: keep the `max_hp > 0` gate but also read the dynamic value.
   - Add `hp_coin` and `hp_bone` fields to obs schema (mod-side: `player:GetCoinHearts()` and `player:GetBoneHearts()`) so future Keeper/Forgotten training doesn't require another schema bump.
   - Effort: 2-3 hr including mod obs.lua edit. Blocks: no. Even under Isaac-only training this is a pure improvement (Eden's random HP would already benefit).

3. **Restore active-item action head (`use_active`) — but gate it behind `has_active_item` in the obs so it's masked to no-op when the slot is empty.**
   - Rationale: the 2026-07-02 removal comment blamed "random exploration harmful" — true, but the fix is action-masking (a solved technique), not action-space amputation. Without space bar the agent cannot express D6 rerolls, which are Isaac's whole identity.
   - Add `player.active_item_id` and `player.active_charge` to obs (mod already tracks these).
   - Action space: `MultiDiscrete([9, 5, 2])` — 3rd factor is press-space-or-not. Trainer applies a mask that zeros out logits for factor 3 when `has_active_item == 0` or `active_charge < required_charge`.
   - Reward: keep the existing `r_use_item = 2.0` event reward but pair with a small negative for wasting a charge (space when `active_charge == 0` should be a no-op, not a penalty — Isaac ignores it).
   - Effort: 4-6 hr (action-mask plumbing through PPO/Dreamer). Unblocks: Isaac's D6 → future Judas / Bethany / Lilith / Isaac variants.

---

## Sources

- **Kept:**
  - [Characters — bindingofisaacrebirth.wiki.gg](https://bindingofisaacrebirth.wiki.gg/wiki/Characters) — canonical HP / starting-item table.
  - [The Lost (Strategy)](https://bindingofisaacrebirth.wiki.gg/wiki/The_Lost_(Strategy)) — confirms 0 HP, can't gain HP, absorbs soul/black/eternal without gain.
  - [Keeper wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Keeper) — coin hearts, max 3, heal via coin pickup.
  - [??? (Character) wiki](https://bindingofisaacrebirth.wiki.gg/wiki/%3F%3F%3F_(Character)) — soul-only HP, devil deals cost soul at red price.
  - [Azazel — isaacguru.com](https://isaacguru.com/wiki/isaac/chr7) — short-range Brimstone, 8 hits/s, flight.
  - [The Forgotten wiki](https://bindingofisaacrebirth.wiki.gg/wiki/The_Forgotten) — bone club, cannot fire regular tears, bone-heart conversion.
  - [Debug Console — wofsauge/IsaacDocs](https://wofsauge.github.io/IsaacDocs/rep/tutorials/DebugConsole.html) — confirms `restart [character id]` semantics with ids 0–40.
  - [Judas wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Judas) — 1.35× dmg mult, Book of Belial, Black Judas transformation.
  - [Bethany fandom](https://bindingofisaacrebirth.fandom.com/wiki/Bethany) — soul hearts as active-item charges (Book of Virtues wisps).
  - [Jacob & Esau wiki](https://bindingofisaacrebirth.fandom.com/wiki/Jacob_%26_Esau) — two simultaneous players.
  - [Platinum God — tboi.com](https://www.tboi.com/) — item/pool cheat sheet.
- **Dropped:**
  - eathealthy365.com character guide — SEO farm, some facts imprecise (Isaac starting HP shown as "3 red hearts" without unit clarity).
  - gamerant.com console commands article — general-audience overview, redundant to IsaacDocs.
  - steamsolo.com unlock guide — useful but overlapping with wiki; kept only for cross-check on stat numbers.

## Gaps

- **Exact obs mapping for coin hearts / bone hearts / rotten hearts.** Mod-side `obs.lua` was not read in this pass. Confirmed only that `_PLAYER_FIELDS` in spaces.py exposes `hp_red / hp_soul / hp_black / hp_max`. Next step: `grep -n "hp_" mods/isaac-rl-bridge/obs.lua` to see how Keeper's coin hearts and Forgotten's bone hearts serialize today.
- **Whether `--set-stage=1` in the launcher script also sets a character.** The comment in main.lua asserts "we boot as Isaac" but the flag is stage-only. Depends on Steam save state. Next step: read the launcher shell/python that invokes Isaac.exe to see if `--player=0` or similar is passed.
- **Tainted characters (17 more IDs)** — omitted entirely. Tainted Isaac (id 21) has a lunch-box item system; Tainted Lost has no Holy Mantle. Out of scope for a one-pass brief; add to curriculum-design followup.
- **Empirical confirmation the current run IS Isaac.** No log/screenshot inspection. Recommend adding a `logger.info("run started as player_type=%d", obs.player.player_type)` on every reset.
