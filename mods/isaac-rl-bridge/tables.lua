-- tables.lua — collectible + NPC ID dense-index maps.
--
-- 2026-07-12 Track A: bumped from a curated top-256 list to an identity
-- mapping over ALL vanilla Repentance CollectibleType IDs (1..732). Prior
-- limitation: items like Sacred Orb (id=691), Angelic Prism (id=528),
-- Cricket's Body variants > 256, and many DLC items silently collapsed
-- to "unknown" (index 0). The MultiBinary(733) obs in spaces.py now covers
-- everything the game can hand out.

local M = {}

M.COLLECTIBLES = {}
for cid = 1, 732 do
    M.COLLECTIBLES[cid] = cid
end
M.PASSIVES_K = 733

-- NPC (entity type) -> dense index. Type is EntityType.ENTITY_* in the Isaac API.
-- We index by numeric EntityType value. The dense index is just this list's
-- position, so the mapping is stable/distinct as long as the numeric list is.
--
-- 2026-07-14: comments corrected against the canonical Repentance
-- resources/scripts/enums.lua (the prior comments were a wrong copy-paste guess
-- that mislabeled every entry — e.g. it called 12 "Clotty" and 26 "Boil",
-- which is what let Stage 0 ship spawning Maw(26) under the belief it was
-- Horf. Real values: Horf=12, Fly=13, Pooter=14, AttackFly=18, Maw=26).
-- The numeric list and its ordering are UNCHANGED — only the labels — so
-- dense indices (and therefore the policy's enemy type_idx feature) are
-- byte-for-byte identical to before. Types with no canonical enemy name in
-- enums.lua (mostly 106-110, 201-255) are kept as list padding and marked
-- "(unmapped)"; they never fire on the current curriculum.
M.NPC_TYPES = {}
local common_npcs = {
    10,  -- Gaper
    11,  -- Gusher
    12,  -- Horf            (Stage 0 control enemy: stationary blood-shot shooter)
    13,  -- Fly             (basic, harmless)
    14,  -- Pooter
    15,  -- Clotty
    16,  -- Mulligan
    17,  -- Shopkeeper
    18,  -- Attack Fly      (Stage A/B enemy: homing, contact damage)
    19,  -- Larry Jr. (boss)
    20,  -- Monstro (boss)
    21,  -- Maggot
    22,  -- Hive
    23,  -- Charger
    24,  -- Globin
    25,  -- Boom Fly
    26,  -- Maw
    27,  -- Host
    28,  -- Chub (boss)
    29,  -- Hopper
    30,  -- Boil
    32,  -- Brain
    33,  -- Fireplace
    34,  -- Leaper
    35,  -- Mr. Maw
    36,  -- Gurdy (boss)
    37,  -- (unmapped)
    38,  -- Baby
    39,  -- Vis
    41,  -- Knight
    42,  -- Stone Head
    43,  -- Monstro II (boss)
    44,  -- Poky
    45,  -- Mom (boss)
    46,  -- Sloth (miniboss)
    47,  -- Lust (miniboss)
    50,  -- Greed (miniboss)
    51,  -- Envy (miniboss)
    52,  -- Pride (miniboss)
    53,  -- Dople
    54,  -- Flaming Hopper
    55,  -- Leech
    56,  -- Lump
    57,  -- Membrain
    58,  -- Para-Bite
    59,  -- Fred
    60,  -- Eye
    61,  -- Sucker
    62,  -- Pin (boss)
    63,  -- Famine (boss)
    64,  -- Pestilence (boss)
    65,  -- War (boss)
    66,  -- Death (boss)
    67,  -- Duke of Flies (boss)
    68,  -- Peep (boss)
    69,  -- Loki (boss)
    70,  -- (unmapped)
    71,  -- Fistula (big)
    72,  -- Fistula (medium)
    73,  -- Fistula (small)
    74,  -- Blastocyst (boss)
    75,  -- Blastocyst (medium)
    76,  -- Blastocyst (small)
    77,  -- Embryo
    78,  -- Mom's Heart (boss)
    79,  -- Gemini (boss)
    80,  -- Moter
    81,  -- The Fallen (boss)
    82,  -- Headless Horseman (boss)
    83,  -- (unmapped)
    84,  -- (unmapped)
    85,  -- (unmapped)
    86,  -- (unmapped)
    87,  -- (unmapped)
    88,  -- (unmapped)
    89,  -- (unmapped)
    90,  -- (unmapped)
    91,  -- (unmapped)
    92,  -- (unmapped)
    93,  -- (unmapped)
    94,  -- (unmapped)
    95,  -- (unmapped)
    96,  -- (unmapped)
    97,  -- (unmapped)
    98,  -- (unmapped)
    99,  -- (unmapped)
    100, -- (unmapped)
    101, -- (unmapped)
    102, -- (unmapped)
    103, -- (unmapped)
    104, -- (unmapped)
    105, -- (unmapped)
    106, -- (unmapped)
    107, -- (unmapped)
    108, -- (unmapped)
    109, -- (unmapped)
    110, -- (unmapped)
    200, -- (unmapped)
    201, -- (unmapped)
    202, -- (unmapped)
    203, -- (unmapped)
    204, -- (unmapped)
    205, -- (unmapped)
    206, -- (unmapped)
    207, -- (unmapped)
    208, -- (unmapped)
    209, -- (unmapped)
    210, -- (unmapped)
    211, -- (unmapped)
    212, -- (unmapped)
    213, -- (unmapped)
    214, -- (unmapped)
    215, -- (unmapped)
    216, -- (unmapped)
    217, -- (unmapped)
    218, -- (unmapped)
    219, -- (unmapped)
    220, -- (unmapped)
    221, -- (unmapped)
    222, -- (unmapped)
    223, -- (unmapped)
    224, -- (unmapped)
    225, -- (unmapped)
    226, -- (unmapped)
    227, -- (unmapped)
    228, -- (unmapped)
    229, -- (unmapped)
    230, -- (unmapped)
    231, -- (unmapped)
    232, -- (unmapped)
    233, -- (unmapped)
    234, -- (unmapped)
    235, -- (unmapped)
    236, -- (unmapped)
    237, -- (unmapped)
    238, -- (unmapped)
    239, -- (unmapped)
    240, -- (unmapped)
    241, -- (unmapped)
    242, -- (unmapped)
    243, -- (unmapped)
    244, -- (unmapped)
    245, -- (unmapped)
    246, -- (unmapped)
    247, -- (unmapped)
    248, -- (unmapped)
    249, -- (unmapped)
    250, -- (unmapped)
    251, -- (unmapped)
    252, -- (unmapped)
    253, -- (unmapped)
    254, -- (unmapped)
    255, -- (unmapped)
}
for i, tid in ipairs(common_npcs) do
    M.NPC_TYPES[tid] = i
end
M.NPC_TYPES_K = 256

-- Pickup variant -> dense index. EntityPickup.Variant values.
-- 10=Heart, 20=Coin, 30=Key, 40=Bomb, 50=Chest, 70=Pill, 300=Card, 350=Trinket, 100=Collectible
M.PICKUP_KIND = {
    [10]  = 1,   -- heart
    [20]  = 2,   -- coin
    [30]  = 3,   -- key
    [40]  = 4,   -- bomb
    [50]  = 5,   -- chest
    [70]  = 6,   -- pill
    [300] = 7,   -- card
    [350] = 8,   -- trinket
    [100] = 9,   -- collectible pedestal
}
M.PICKUP_KIND_K = 9

return M
