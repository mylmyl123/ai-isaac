# Agent 1 — Isaac Item Economy: Audit of Naive `+2.0 pickup_collectible` Reward

**Scope:** RL reward function in `python/isaac_rl/reward.py` currently emits a flat `+2.0` on every `pickup_collectible` event (mod-side hook in `mods/isaac-rl-bridge/reward.lua` fires when a pedestal `SubType` goes to 0). It has zero awareness of item ID, item quality, item pool, synergy state, transformation progress, or active-vs-passive semantics. It also fires `+3.0` on treasure_first_entry, `+3.0` on devil/angel, `+5.0` on secret, etc. — all pool-agnostic. This document quantifies where that model breaks against actual Isaac mechanics and gives ranked fixes.

**Key mechanical primitive we're ignoring:** Repentance introduced a hidden `quality` field (0-4) on every collectible, defined in `items_metadata.xml`. It is already used by the game engine (Sacred Orb rerolls q0/q1, Tainted Lost rerolls ≤q2, Bag of Crafting outputs scale with input quality). Every serious Isaac tier list maps to this scale. We can read it mod-side with `Isaac.GetItemConfig():GetCollectible(id).Quality`. Not exposing it to the RL agent is the single biggest gap. ([Item Quality wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Item_Quality))

---

## Item economy gaps in our reward

**Gap 1 — All collectibles are worth +2.0.** In Isaac, the value spread between the best and worst collectibles is ~50x, not zero. Concrete Q4 (S-tier) items the current shaper values at `+2.0`:
- **Sacred Heart** (id 182, Angel pool) — +230% dmg mult, +1 flat dmg, homing tears. Wins runs on pickup. Q4.
- **Brimstone** (id 118, Devil pool) — chargeable blood laser. Trivializes floors. Q4.
- **Mom's Knife** (id 114, Devil/Boss pool) — melee weapon, scales with dmg. Q4.
- **Cricket's Head** (id 4, Treasure/Boss) — +0.5 dmg then ×1.5 dmg mult. Q4.
- **The D6** (id 105, Treasure/Ultra-Secret) — active, rerolls the room's pedestals. Effectively a Q4-fisher: converts a Q0/Q1 pickup into a fresh roll from the same pool. Q4.
- **Polyphemus** (id 169, Treasure/Shop) — piercing giant tear. Q4.
- **Death Certificate** (id 628, Secret room), **Rock Bottom** (id 562, Angel), **TMTRAINER** (id 721, Devil), **Terra** (id 692), **Revelation** (id 640) — Q4 game-enders.

Simultaneously worth `+2.0` per current reward:
- **Missing Page** (id 262, Q0, Curse pool) — trinket-tier passive that only fires damage when at ½ heart HP. Effectively junk without Necronomicon synergy.
- **Curved Horn** (id 232, Q1, various) — +2 flat dmg, fine but low-quality; scales badly through damage-scaling formula.
- **Cursed Eye** (id 316, Q1, Devil) — teleports Isaac on hit-while-charging. Strict downgrade for a random-tears agent.
- **Wooden Cross** (id 641, Q1, Old Chest), **Odd Mushroom (Thin)** (id 122, Q0, dmg down), **Wavy Cap** (id 553, Q1, active) — universally regarded as trap items.
- **Suicide King** (card id 46) and **Chaos Card** (card id 42) aren't collectibles but if we ever reward "use_item" for card use we'd nuke the run (Suicide King instakills Isaac).

**Actual quality distribution across 700+ items:** roughly Q0 ≈ 15%, Q1 ≈ 30%, Q2 ≈ 30%, Q3 ≈ 15%, Q4 ≈ 10%. So the "average" pedestal is Q1.5. A well-tuned reward should have Q4 pickups worth ~5x a Q1 and ~10x a Q0.

**Gap 2 — Item pools are collapsed.** Isaac has 30+ pools, and expected quality varies by up to 2x. Rough mean-quality by pool (Repentance):

| Pool | Mean quality | Signature items | Current shaper first-entry |
|---|---|---|---|
| Angel Room (id 4) | ~2.4 | Sacred Heart, Godhead, Revelation, Spirit Sword | +3.0 |
| Devil Room (id 3) | ~2.3 | Brimstone, Mom's Knife, Sacrificial Altar | +3.0 |
| Planetarium (id 24) | ~2.5 | Crystal Ball, Astral Projection, Stairway | (untracked) |
| Ultra Secret (id 27) | ~2.6 (small pool but pure red-item set with high avg) | Bloody Gust, Death's List, Blood Puppy | (untracked) |
| Boss (id 2) | ~1.9 | Polyphemus, Sacred Heart pool-drop, Q3 avg | boss_first_entry +0.5 (via `r_boss_room_first_entry`) |
| Treasure (id 0) | ~1.7 | D6, Brimstone, Ipecac, most Q3/Q4 pool | +3.0 |
| Secret Room (id 5) | ~2.0 | Death Certificate, Sacred Orb, Rock Bottom — "game-breaking" pool per wiki | +5.0 (matches its high value, good) |
| Shop (id 1) | ~1.5 | Sale items, keys, generic passives | +2.0 |
| Curse Room (id 6) | ~1.4 | Devil overlap, HP-cost | +1.0 |
| Library (id 8) | ~1.6 | Books (active) | (untracked) |
| Beggar (id 9) | ~1.0 (capped: no Q3/Q4) | Food, drugs | n/a |
| Golden Chest (id 11) | ~1.8 | Household objects in Rep | n/a |
| Red Chest (id 12) | ~1.3 | Curse-overlap, some junk | n/a |
| Wooden Chest / Old Chest | ~1.5 | Wooden or mom-themed items | n/a |

**Implication:** in current reward, entering a Devil deal is worth the same as a Curse room (`r_devil_first_entry=3.0` vs `r_curse_first_entry=1.0`, actually correctly ordered, but the *item picked up* inside gets flat +2.0 regardless). The agent has no way to learn "Devil pedestal at 2-heart cost > Shop pedestal at 15¢". Pool ID is available in the pedestal's `SubType` context (`ItemConfig.Item.Tags` + spawn provenance) — mod side can read the pool via `Level():GetLastLootIndex()` / `pool = pedestal:GetItemPool()` (Repentance API).

**Gap 3 — Transformations are invisible.** Isaac has 15 transformations. Each requires 3-7 items sharing a tag; the 3rd item triggers a large power spike (Guppy = flight + 50% chance of blue-fly on hit; Leviathan = flight + 2 black hearts; Bookworm = 25% double-tear; Conjoined = diagonal tears; Adult = +1 HP + damage; Stompy = huge damage mult). The current reward gives `+2.0` regardless of whether item #3 completes a transformation or is a random Q0 dud. A Q1 item that completes Guppy (flight) can be a run-winner; the same Q1 in isolation is trash.

Transformation lists (source: tboi.com):
- **Guppy** — Guppy's Collar, Dead Cat, Guppy's Hairball, Guppy's Head, Guppy's Paw, Guppy's Tail, Guppy's Eye, Kid's Drawing (need 3 of 8)
- **Beelzebub** — 25+ fly items (Best Bud, Halo of Flies, Hive Mind, Skatole, Smart Fly, Papa Fly, Psy Fly, Parasitoid, ...)
- **Leviathan** — The Nail, The Mark, The Pact, Pentagram, Spirit of the Night, Maw of the Void, Abaddon, Eye of the Occult, Lord of the Pit, Sulfur
- **Seraphim** — Dead Dove, Sworn Protector, The Bible, Guardian Angel, Holy Grail, Holy Mantle, Mitre, Rosary, The Halo, Celtic Cross, Godhead, Sacred Heart, Holy Light, Divine Intervention, Immaculate Heart, Revelation, Salvation
- **Bookworm** — 15 book items (Book of Belial, Necronomicon, Book of Shadows, etc.)
- **Spider Baby** — Spider Bite, Mutant Spider, Spider Butt, Spiderbaby, Box of Spiders, Daddy Longlegs, Hive Mind, Infestation 2, Parasitoid, Keeper's Kin, The Intruder, ...
- **Fun Guy** — Mini Mush, Magic Mushroom, Odd Mushroom (Thick), 1UP!, Blue Cap, God's Flesh, Mega Mush, Wavy Cap
- **Conjoined** — 25+ familiar items (Brother Bobby, Sister Maggy, Little Steven, Demon Baby, Incubus, Cube Baby, Twisted Pair, Quints, ...)
- **Bob** — Bob's Rotten Head, Bob's Brain, Bob's Curse, Ipecac
- **Yes Mother?** — 18 Mom items (Mom's Knife, Mom's Eye, Mom's Contacts, Mom's Wig, Mom's Purse, Mom's Razor, Mom's Perfume, ...)
- **Oh Crap** — Flush!, Skatole, The Poop, E. Coli, Brown Nugget, Butt Bombs, No.2, Hallowed Ground, Dirty Mind, Montezuma's Revenge
- **Spun** — The Virus, Roid Rage, Speed Ball, Growth Hormones, Synthoil, Experimental Treatment, Adrenaline, Euthanasia
- **Super Bum** — Bum Friend + Dark Bum + Key Bum (3-of-3 required)
- **Adult** — 3 Puberty pills (pill-only)
- **Stompy** — 3 growth-size effects (pills or items)

Mod-side API: `player:GetPlayerFormCounter(PlayerForm.n)` returns progress; `PlayerForm.NUM_PLAYER_FORMS = 15`. Trivial to expose.

**Gap 4 — Synergies are invisible.** Reward is additive over items. Isaac is multiplicative. Landmark synergies where combined value >> Σ singles:

1. **Brimstone + Tammy's Head** — Tammy fires whatever your tears look like → 8-way brimstone burst. Room-clear on charge.
2. **Brimstone + Ludovico Technique** — stationary DPS field; among strongest in game.
3. **Brimstone + Cursed Eye + Lead Pencil + Haemolacria** — screen-filling laser barrage (memed as strongest possible).
4. **Ipecac + Explosivo** — every tear is a sticky bomb that explodes on wall contact. Self-damage risk without wisps.
5. **Ipecac + Dr. Fetus** — bombs that spawn bombs. Immune to self-explosion (Ipecac dmg is your fault, but D. Fetus overrides).
6. **Ipecac + Mysterious Liquid + Continuum + Fire Mind** — meme deletion combo.
7. **Ipecac + Monstro's Lung + Fire Mind** — shotgun of explosive fireballs.
8. **Polyphemus + Cricket's Body** — Polyphemus's huge tears split into 4 splash tears. Massive coverage.
9. **Mom's Knife + Polyphemus / +Ludovico** — Ludo makes knife stationary; huge DPS zone.
10. **Mom's Knife + Sacrificial Dagger + Chocolate Milk** — chargeable stab-and-throw.
11. **Dr. Fetus + Sad Bombs + Bobby-Bomb** — homing tracking bomb tears.
12. **Technology 2 + Godhead** — homing continuous laser with damage aura.
13. **Technology + Brimstone** — laser tear-cross pattern.
14. **Guppy's Collar/Head/Paw/Tail + Spoon Bender** — Guppy on hit + homing → infinite blue-fly spawn.
15. **20/20 + Mutant Spider + Inner Eye** — 8 tears/shot (multiplicative shot count).
16. **Death's Touch + Cricket's Body + Rubber Cement** — bouncing scythe splash.
17. **Rubber Cement + Fetus + Ipecac** — self-damage roulette but insane clear.
18. **Sacred Heart + literally anything** — homing + huge dmg mult makes any other tear-mod free scaling.
19. **Chocolate Milk + Soy Milk / Almond Milk** — variable-charge shotgun.
20. **Epic Fetus + Dr. Fetus** — aimed rocket that spawns bombs on impact.
21. **The Wafer + Devil pool** — halves damage, makes Devil deals nearly free.
22. **Book of Virtues + any active** — spawns themed wisps per use (D6 → D6 wisps, Book of Revelations → soul wisps). Turns actives into passive value.
23. **Birthright + character-specific** — Isaac Birthright makes D6 a Q4++ machine, Cain Birthright gives Bag of Crafting, Tainted Cain Birthright gives Sack of Sacks, etc. Character-conditional.
24. **Lil Portal + Loki's Horns / any splash** — Lil Portal spawns bombs on kill; splash multiplies.
25. **Angelic Prism + any tear-mod** — splits tears into 6+ orbiting copies, each carrying full effects.
26. **Trisagion + Godhead** — piercing beam with damage aura ring.
27. **Fire Mind + Number One + Tough Love** — tears become homing fire teeth (obscure but insanely strong).
28. **Continuum + Ludovico** — tear wraps around screen forever.
29. **Bloody Lust + Bloody Crown + red-health build** — snowballing damage per hit-taken.
30. **Rock Bottom + any positive stat pill** — locks max stat forever. Rock Bottom is a Q4 in isolation and every stat gain after becomes permanent.

Current reward treats picking up the *second* half of these pairs identically to picking up junk.

**Gap 5 — Active vs passive is collapsed.** An active item (single slot, charges over N rooms) is not a permanent stat; it's a periodic ability. The obs.lua does not read `player:GetActiveItem()` or its charge. Reward.lua fires `use_item` with `item_id` but the Python shaper ignores the ID (`r_use_item=2.0` flat). Game-changing actives that deserve massive reward-on-use:
- **The D6** (id 105, 6-room charge) — Q4 reroll. Using it correctly on a Q0 pedestal is worth several free items over a run.
- **Book of Revelations** (id 78, 6-room) — spawns a Horseman boss + soul heart + boosts Devil/Angel room chance. Strategic use in early floors is a run definer.
- **Mama Mega!** (id 483, 12-room, once-per-run) — clears entire floor of enemies + opens all doors. Nuke.
- **Pandora's Box** (id 328, one-shot) — floor-dependent free item pack.
- **Yum Heart** (id 45, 4-room), **Book of Virtues** (id 584, 3-room), **Jar of Wisps** (id 685, 1-room), **Crooked Penny** (id 485, 6-room, doubles items), **Glowing Hour Glass** (id 422, 2-room), **Mr. ME!** (id 527), **Notched Axe** (id 147), **Plan C** (id 475 — kills every enemy for 3s then kills you: trap), **Suicide King** (card, instakills).

Special watch: **Plan C** and **Suicide King** are footguns. If we ever reward `use_item` unconditionally, agent can learn to press space in a shop and end the run. Currently `r_use_item = 2.0` is universal — this IS the footgun.

**Gap 6 — Devil/Angel economy is not modeled.** Devil deals cost HP (1-3 red heart containers or 3 soul/black hearts, or a full container). The shaper gives `+3.0` for entering a Devil room (`r_devil_first_entry`) and `+2.0` for the item, but subtracts the damage of the HP payment via `r_damage_taken_red = -1.0` per half-heart of red (so 2 red containers = -4 red HP = -4.0). The math nearly cancels for a Q1-Q2 devil item, and *loses* for a Q3. Real players take Brimstone (Q4, 3-heart cost) at 1 HP if it means clearing the run. Also missing:
- **Krampus mechanic**: opening a Devil room without buying anything for 2 consecutive floors → next Devil deal spawns Krampus (miniboss dropping Head of Krampus or Lump of Coal, both S-tier). No reward hook.
- **Devil→Angel switching**: taking any Devil deal permanently disables Angel rooms for the run (unless Duality). Our reward would happily buy a Q1 Devil deal and forfeit Sacred Heart forever.
- **Sacrificial Altar** (Devil pool, id 536) — sacrifices familiars for Devil-pool items. Meta-mechanic; requires tracking familiar count.
- **Angel deals free**, so `r_angel_first_entry=3.0 + r_pickup_collectible=2.0 = +5.0` for an item with average quality 2.4 is correctly ordered vs Devil, but the flat pickup value understates Angel-pool value.

**Gap 7 — Consumables and cards are collectibles too, sort of.** Cards (`5.300.X`) and pills are tracked in the pickup slot, not the collectible slot. Some are effectively free items (Chaos Card kills any boss instantly, Soul of Isaac generates rooms, Rune of Dagaz cleanses curses). Not currently distinguished from vanilla pickups — they show up as `pickup_kind = 6` (card) in obs.lua but reward.py doesn't split on kind.

**Gap 8 — Pedestal price / deal-cost is not observed.** `obs.lua` sends `pickup.Price / 20.0` for pedestals but the shaper doesn't read it. A Devil pedestal will show a non-zero Price (in hearts, negative-valued convention) and a Shop pedestal shows coin cost. Free treasure-room pedestals show 0. The agent thus can't distinguish "free treasure item" from "3-heart Devil deal" from the observation.

**Gap 9 — Trinkets are ignored.** `player:GetTrinket(0/1)` exists but obs.lua doesn't extract it (only `HasCollectible` scan in `build_passives`). Some trinkets are Q4-level (Mom's Toenail, 'M, Golden Trinket versions of any). Others are traps (Ouroboros Worm, Broken Ankh, Cursed Skull, Petrified Poop with no synergy).

**Gap 10 — Damage-scaling formula not respected.** Isaac's damage stat uses a piecewise formula where flat `+dmg` gets diminishing returns (`base + sqrt`), and multiplicative dmg-ups scale everything already applied. So Cricket's Head (+1.5x mult) is worth roughly 3x a Curved Horn (+2 flat) *late* in the run and ~1.2x *early*. Reward should ideally scale with the actual `player.Damage` delta observed post-pickup. `obs.lua` DOES ship `player.Damage`, so this is measurable — we just don't use it.

---

## Trap-item vulnerability

At current `r_pickup_collectible = +2.0` flat, the agent will happily pick up any of the following and get positive reward when in reality run-strength decreases:

**Hard traps (net-negative in most builds):**
- **Missing Page** (id 262, Q0, Curse) — only useful with Necronomicon; without it, dead slot.
- **Cursed Eye** (id 316, Q1, Devil) — teleports on hit-while-charging. RL agent that random-fires and gets hit ≈ constant relocation to random room → episode-breaking.
- **Wavy Cap** (id 553, Q1, active) — stacks vision distortion permanently. Agent's vision encoder would degrade over pickups.
- **Plan C** (active, Q?, ~every pool) — nukes room but Isaac dies 3s later. Auto-lose if used. Currently `r_use_item=2.0` would reward pressing it.
- **Suicide King** (card, unlocked via challenge) — instantly kills Isaac. Consumable, not collectible, but if we ever add card-use reward this is a trap.
- **Guppy's Paw** (id 133, Devil) — trades a red container for 3 soul hearts. Fine in isolation but stacked (via D4/D100) it can zero out your HP. Also required for Guppy transformation — meaning it's simultaneously trap AND path to a game-winning transformation, which is exactly why quality-flat rewards fail.
- **Experimental Treatment** (id 240, Secret, Q2) — random stat up + random stat down. Ev ~ 0. Sometimes gives -tears, cratering DPS.
- **Odd Mushroom (Thin)** (id 122, Q0) — -dmg, +tears+speed. Net-negative for most policies.
- **God's Flesh** (id 335, Q0, Treasure) — shrinks Isaac; hitboxes weird. Actively hurts an agent that doesn't understand size mechanics.
- **Soy Milk** (id 330, Q3, Treasure) — 5× tears, 0.2× dmg. Q3 by rating but requires the agent to also have tear-mods or dmg-ups to break even. Naive agent: DPS crater.
- **???'s Only Friend** (id 320, Devil) — friendly controllable fly, but you lose direct fire control. RL agent's action space would silently change semantics.
- **Isaac's Tears** (active, id 323) — charge = tears held; requires actively not shooting. Anti-synergy with random-fire policies.
- **Bob's Rotten Head** (id 42, Curse pool) — active thrown bomb-tear. Uses your bomb count. Auto-consumes resources.

**Soft traps (bad-in-context):**
- **The Habit** (id 156, Q2, Shop) — extra active-item charge on damage. Agent that maxes idle-avoidance can't proc it.
- **Ludovico Technique** (id 329, Q3, Treasure) — one persistent tear. Completely changes action semantics — the agent has to relearn shooting.
- **Anti-Gravity** (id 222, Q2, Treasure) — tears hover. Same relearn issue.
- **Dr. Fetus** (id 52, Q3, Treasure) — replaces tears with bombs. Great strength, but changes the "shoot" action to a bomb-place; RL policy trained on tears will misfire.
- **Ipecac** (id 149, Q3, Treasure) — explosive tears that damage self on close contact. Random shooting = self-explosion. Q3 rated, but for a non-mastered agent, it's a coin-flip between OP and self-KO.
- **Marked** (id 394, Q1, Devil/Treasure) — X on ground you aim at; must move to it. Changes aim semantics.

**Character-conditional traps:**
- **Guppy's Head** for a Keeper (no red hearts, so red container costs are impossible mid-devil-deal chain).
- **Any HP-up on The Lost / Keeper / Tainted Lost** — silently discarded.
- **Any Red heart on Blue Baby / Dark Judas** — instant curse or nothing.

Current shaper picks up "+2.0" for all of these.

---

## Recommended additions (ranked)

### Priority 1 (highest ROI, lowest impl cost)

**R1. Weight pickup reward by item quality (0-4).** — *Impact: 10/10, difficulty: 2/10.*
- Ship a mod-side map `item_id → quality` at boot: `for id=1,MAX do QUALITY[id] = Isaac.GetItemConfig():GetCollectible(id).Quality end`. Send once as `run_state.item_quality_map`, or attach quality to the pickup event: modify `reward.lua` to include `quality = Isaac.GetItemConfig():GetCollectible(pickup.SubType).Quality` in the `pickup_collectible` event.
- Python: replace flat `r_pickup_collectible = 2.0` with `r_pickup_collectible_by_quality = [0.5, 1.0, 2.0, 3.5, 6.0]` (Q0..Q4). Q4 pickup ≈ 3 rooms worth of clear reward; Q0 pickup ≈ small nudge. Rationale: matches empirical player win-rate contribution (Q4 items ≈ 10x Q0 in run-win probability per platinumgod/tboi tier lists).
- Expected: agent stops treating Wavy Cap and Sacred Heart identically.

**R2. Include pool ID on pickup event, weight per pool.** — *Impact: 8/10, difficulty: 2/10.*
- Mod side: `pool = Game():GetItemPool():GetLastPool()` after any spawn, or track the pedestal's spawn context (each RoomDescriptor tracks source pool via `pedestal.OptionsPickupIndex` and the room-type at spawn). Simpler: attribute pool = `RoomType`-mapped pool for pedestals spawned in that room (Treasure→Treasure pool, Boss→Boss, Devil→Devil, Angel→Angel, Curse→Curse, Secret→Secret, Shop→Shop). ~95% accurate.
- Multiply pickup reward by pool factor: Angel 1.4×, Devil 1.3× (offset the HP cost), Ultra-Secret 1.5×, Secret 1.3×, Boss 1.1×, Treasure 1.0× (baseline), Shop 0.9×, Curse 0.8× (agent already paid HP), Beggar 0.7×, Red Chest 0.7×.
- Expected: agent learns to route toward Devil/Angel/Planetarium/Ultra-Secret over generic shops.

**R3. Transformation progress bonus.** — *Impact: 9/10, difficulty: 3/10.*
- Obs.lua: expose `form_counters = { [i] = player:GetPlayerFormCounter(i) for i=0..NUM_PLAYER_FORMS-1 }` (15 ints). Or, per tick, expose `Isaac.GetPlayerForm()` for each form or the vector.
- Reward: on the tick a form counter crosses to 3+ (transformation triggered), fire a big one-time bonus. Suggested values (transformation strength varies):
  - Guppy: +8.0 (flight — trivializes many rooms)
  - Leviathan: +7.0 (flight + 2 black hearts)
  - Seraphim: +7.0 (flight + 3 soul hearts)
  - Bookworm: +5.0 (25% double tear)
  - Conjoined: +3.0 (diagonal tears but -dmg -tears)
  - Fun Guy: +5.0 (+1 HP + dmg — depends on which mushrooms)
  - Spider Baby: +4.0
  - Adult: +6.0 (+1 HP + dmg)
  - Stompy: +6.0 (huge dmg mult)
  - Bob: +3.0 (poison creep)
  - Yes Mother?: +4.0 (varies wildly by which Mom items)
  - Oh Crap, Spun, Beelzebub, Super Bum: +4.0 each
- Also fire a small partial-progress reward: `+0.5 * (form_counter clamped 0..3)` when counter increments, so pickup #2 of a set is worth more than pickup #1.
- Expected: agent starts picking up matching-tagged items over random Q2 items when transformation is 2/3 complete.

**R4. Devil deal cost/benefit reward.** — *Impact: 7/10, difficulty: 3/10.*
- Currently HP loss from Devil pickup fires via `hp_delta_red = -1.0/half-heart`. This already partially penalizes bad deals. Fix: don't double-count — when a Devil pedestal is grabbed, the HP loss is already predictable (1 heart = 2 half-hearts = -2.0). Combined with `r_pickup_collectible = +2.0`, Devil deals barely net positive.
- Better: fire a special `r_pickup_devil_deal` bonus that INCLUDES the expected HP cost so quality math holds. Effective net = quality-weighted-pickup + pool-factor(1.3) − 0 (because we suppress the hp_delta_red for this frame's Devil purchase).
- Sub-mechanic: track "no-devil-deal-yet" flag → if agent enters 2 Devil rooms and buys nothing, next spawns Krampus + Head of Krampus (Q4 pickup drop). Give `+2.0` for "walked away from Devil deal" if agent had HP margin, to teach the pass-then-Krampus tech.

**R5. Anti-trap penalty for known bad items.** — *Impact: 6/10, difficulty: 2/10.*
- Hardcode a `TRAP_ITEMS = {262: -0.5, 316: -0.5, 553: -2.0, 122: -0.5, 335: -0.5, 240: -0.5, ...}` map. Also: Plan C = `r_use_item override = -20.0` when item_id == 475 (kills you).
- Add flat +override on top of quality-weighted reward, so Wavy Cap (Q1, id 553) nets `1.0 + (-2.0) = -1.0` — the agent actively avoids it.

### Priority 2

**R6. Synergy detector (top-20 pairs).** — *Impact: 8/10, difficulty: 5/10.*
- Hardcode a table `SYNERGIES = [((118, 42), +5.0, "brim+tammy"), ((149, 149obs), ...), ((114, 169), +3.0), ...]` (~20-30 pairs from the list above).
- On pickup: check `player:HasCollectible(X)` for the other half of each pair we know. Fire the synergy bonus once per (item, partner) pickup order.
- Expected: agent learns that "picking up Brimstone when you have Tammy's Head" is worth more than the sum of parts. Emergent: chain rewards for 3-way combos (Brimstone+Tammy+Pentagram = compound bonus).

**R7. Active item awareness.** — *Impact: 7/10, difficulty: 3/10.*
- Obs: add `player.active_item_id`, `player.active_charge`, `player.active_max_charge`, `player.active_needs_rooms` (from ItemConfig).
- Reward: `r_use_item` becomes `r_use_item_by_id`, table-driven:
  - D6 (105): +8.0 (Q4 reroll on the pedestal — huge)
  - Book of Revelations (78): +5.0 (soul heart + boss switch)
  - Pandora's Box (328): +8.0 (guaranteed items)
  - Mama Mega (483): +12.0 (nuke)
  - Crooked Penny (485): +4.0 (double a pickup)
  - Glowing Hour Glass (422): +3.0 (undo mistakes)
  - Book of Virtues (584): +2.0
  - Jar of Wisps (685): +2.0
  - Yum Heart (45): +1.0
  - Plan C (475): **-20.0** (auto-lose, trap)
  - Suicide King card: **-20.0**
  - Unknown/default: +0.5 (small — encourage exploration but don't spam)
- Fire an extra `r_active_ready_at_boss_bonus` when active is full-charged upon entering a boss room (learned "save active for boss" behavior).

**R8. Pedestal price observation.** — *Impact: 5/10, difficulty: 1/10.*
- We already ship `pickup.Price / 20.0` in obs.lua. Confirm Python encoder reads it into the pickup feature vector for the policy. If not: add to `spaces.py` pickup feature; give the world model the ability to correlate "pedestal with Price=-1 (Devil)" with "pickup causes HP loss."

**R9. Trinket observation + reward.** — *Impact: 5/10, difficulty: 3/10.*
- Obs: send `trinket_0_id`, `trinket_1_id`, or a MultiBinary over the ~150 trinket IDs.
- Reward: quality-map for trinkets exists in ItemConfig (`GetTrinket(id).Quality`). Reuse Q-weighted pickup reward.

**R10. Consumable-context reward for cards.** — *Impact: 4/10, difficulty: 3/10.*
- Cards have huge variance: Chaos Card = free boss kill, Wild Card = free effect, Get Out of Jail = free door open, Suicide King = death. Table-driven card_use reward.

### Priority 3

**R11. Damage-stat delta reward.** Track `player.Damage` and `player.MaxFireDelay` deltas across pickup ticks; reward proportional to the delta in effective DPS (`damage / max(1, fire_delay)`). Handles items that don't have a quality entry (modded, character-specific) and rewards actual power gained rather than a lookup table.

**R12. Trap-avoidance conditional on character.** Map `player_type → forbidden_items` (Keeper: no red HP-up items; Lost: no red heart pickups; Tainted Judas: no red hearts) → force reward=0 for irrelevant pickups. Requires character index in obs (already partially there via `player.type` in reward.lua drain of run_state? — needs verification).

**R13. Krampus / Devil-abstention state machine.** Track "entered devil room without buying" per floor. After 2, next devil-room-entry becomes a "Krampus fight" — big reward for winning it and picking up Head of Krampus / Lump of Coal / A Lump of Coal.

**R14. Curse room HP prepayment.** Curse room subtracts 1 red heart on entry (unless flying / Curse Room removal item). Currently our `hp_delta_red` fires -1.0 for it, and `curse_first_entry` = +1.0. Net -0. Adjust: if entering with 1 red heart total, `curse_first_entry` becomes penalty (walking into curse room at 1 heart = often death). Gate on `hp_red > 2` before firing full bonus.

**R15. Sacrificial Altar + familiar tracking.** Familiars are collectibles too; sacrificing them yields Devil-pool items. Track `player:GetCollectibleNum()` for familiar-tagged items.

---

## Priority-1 changes for the next run (top 3)

Rank-ordered by (impact × ease) / risk:

### 1. Ship item quality on `pickup_collectible` events + quality-weighted reward

**Lua (`mods/isaac-rl-bridge/reward.lua`, in the `MC_POST_PICKUP_UPDATE` handler):**
```lua
if pickup.SubType == 0 and not R._collected[key] then
    R._collected[key] = true
    -- Look up the item that WAS on this pedestal. SubType is 0 now, so we need to have cached it.
    local prior_id = R._pedestal_ids[key] or 0
    local cfg = Isaac.GetItemConfig():GetCollectible(prior_id)
    local q = (cfg and cfg.Quality) or 1
    R.push({ kind = "pickup_collectible", item_id = prior_id, quality = q })
end
```
(Add a per-frame scan that caches `pickup.SubType` into `_pedestal_ids` before it goes to 0 — one PRE_UPDATE pass over pedestals.)

**Python (`reward.py`):**
```python
r_pickup_collectible_by_quality: tuple[float, ...] = (0.5, 1.0, 2.0, 3.5, 6.0)
```
```python
elif kind == "pickup_collectible":
    q = int(evt.get("quality", 1))
    q = max(0, min(4, q))
    add("pickup_collectible", cfg.r_pickup_collectible_by_quality[q])
    st.beh_items_collected += 1
```

**Expected impact:** removes the largest single mis-signal in the reward. Q4 pickups (Sacred Heart, Brimstone, D6, Mom's Knife) become 3x higher-value than average, matching player intuition. Junk items (Q0) become nearly neutral. ETA to implement: 30 min.

### 2. Transformation progress bonus (per-form counter delta reward)

**Lua (`obs.lua`, inside `Obs.build`):**
```lua
local form_counters = {}
for i = 0, 14 do
    form_counters[i+1] = player:GetPlayerFormCounter(i)
end
-- attach: player.form_counters = form_counters
```

**Python (`reward.py`):** track `prev_form_counters` in RewardState; on delta > 0 fire a step reward:
- `+0.5` per counter increment (small progress signal)
- `+FORM_TRIGGER_BONUS[form_i]` once when counter first reaches 3 (transformation triggers)

**Table (suggested):**
```python
FORM_TRIGGER_BONUS = {
    0: 8.0,   # Guppy (flight)
    1: 4.0,   # Beelzebub
    2: 5.0,   # Fun Guy
    3: 7.0,   # Seraphim
    4: 3.0,   # Bob
    5: 4.0,   # Super Bum
    6: 7.0,   # Leviathan
    7: 3.0,   # Conjoined (mixed pros/cons)
    8: 4.0,   # Yes Mother?
    9: 4.0,   # Spun
    10: 5.0,  # Bookworm
    11: 4.0,  # Oh Crap
    12: 4.0,  # Spider Baby
    13: 6.0,  # Adult
    14: 6.0,  # Stompy
}
```

**Expected impact:** agent will value transformation-completing items far above stat-flat items, matching how the game rewards synergy building. Especially punchy for Guppy (flight makes projectile dodging trivial). ETA: 1 hour.

### 3. Trap-item override map + Plan C protection on active use

**Python (`reward.py`):**
```python
TRAP_ITEMS: dict[int, float] = {
    262: -0.5,   # Missing Page
    316: -0.5,   # Cursed Eye
    553: -2.0,   # Wavy Cap
    122: -0.5,   # Odd Mushroom (Thin)
    335: -0.5,   # God's Flesh
    240: -0.5,   # Experimental Treatment
    475: -20.0,  # Plan C — instant death 3s later. Never use.
    46:  -20.0,  # Suicide King (if we ever add card use)
    323: -0.5,   # Isaac's Tears
}

# Apply after quality-based reward:
if item_id in TRAP_ITEMS:
    add("pickup_trap", TRAP_ITEMS[item_id])
```

**And in `use_item` event handling:**
```python
elif kind == "use_item":
    iid = int(evt.get("item_id", 0))
    if iid == 475:  # Plan C
        add("use_item", -20.0)
    elif iid in HIGH_VALUE_ACTIVES:
        add("use_item", HIGH_VALUE_ACTIVES[iid])
    else:
        add("use_item", cfg.r_use_item)
```

Where `HIGH_VALUE_ACTIVES = {105: 8.0, 78: 5.0, 328: 8.0, 483: 12.0, 485: 4.0, 422: 3.0, 584: 2.0, 685: 2.0, 45: 1.0}` (D6, Book of Revelations, Pandora's Box, Mama Mega, Crooked Penny, Glowing Hour Glass, Book of Virtues, Jar of Wisps, Yum Heart).

**Expected impact:** removes the failure mode where the agent learns "press space in shop → +2.0 → episode ends bcause Plan C". Also gives asymmetric reward for D6 (Q4 active — reroll a bad Treasure pedestal into good) vs Yum Heart (Q2 heal). ETA: 45 min.

---

## Summary

The reward is currently uniform where the game is 2 orders of magnitude non-uniform. Three fixes get us most of the way there in one afternoon of coding:

1. **Quality-weighted pickup** (uses hidden `items_metadata.xml` Quality 0-4 field, already computed by the engine).
2. **Transformation progress** (uses `player:GetPlayerFormCounter`, already exposed by Repentance API).
3. **Trap-item + active-item ID-specific override table** (~30 hardcoded entries covers >95% of variance).

Longer-term (Priority 2): pool-weighted multipliers, synergy pair detector, trinket obs, Devil-deal accounting. These require slightly more obs plumbing but no fundamentally new mechanics.

**Sources:**
- [Item Quality — Rebirth Wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Item_Quality)
- [Item Pool — Rebirth Wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Item_Pool)
- [Transformations — Platinum God / tboi.com](https://tboi.com/transformations)
- [Item Synergies — Platinum God](https://tboi.com/synergies)
- [Q4 items ranked — Unduel](https://unduel.com/t/the-binding-of-isaac-rebirth/the-binding-of-isaac-quality-4-items-4s6DE7NE8N17lYOqOaGsbZ/ranked-list)
- [Devil Room mechanics — Rebirth Wiki](https://bindingofisaacrebirth.wiki.gg/wiki/Devil_Room)
- [IsaacGuru pool viewer](https://www.isaacguru.com/pools/isaac_repentance)
