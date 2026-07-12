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
-- We index by numeric EntityType value.
M.NPC_TYPES = {}
local common_npcs = {
    10,  -- ENTITY_FLY / Attack Fly
    11,  -- Pooter
    12,  -- Clotty
    13,  -- Mulligan
    14,  -- Shopkeeper
    15,  -- Larry Jr. (boss)
    16,  -- Monstro (boss)
    17,  -- Magnet Fly (?)
    18,  -- Maw of the Void (?)
    19,  -- Host
    20,  -- Chub (boss)
    21,  -- Hopper
    22,  -- Boil / Sack
    23,  -- Spitty
    24,  -- Nulls
    25,  -- Trite
    26,  -- Boil (dup)
    27,  -- Guts
    28,  -- Sucker
    29,  -- Sisters Vis
    30,  -- Fred (?)
    32,  -- Leech
    33,  -- Lump
    34,  -- Mrs. Mole (?)
    35,  -- Widow
    36,  -- Daddy Longlegs (boss)
    37,  -- Blastocyst
    38,  -- Embryo
    39,  -- Momma Gurdy
    41,  -- The Bloat
    42,  -- Peep
    43,  -- Baby Long Legs
    44,  -- Wizoob
    45,  -- Mom (BOSS)
    46,  -- Slide
    47,  -- Heart
    50,  -- Charger
    51,  -- Gaper (Rebirth/AB)
    52,  -- Horf
    53,  -- Fatty
    54,  -- Delirium (endgame)
    55,  -- Fly (variant)
    56,  -- Swarm
    57,  -- Dank
    58,  -- Boil (dup)
    59,  -- Deep Gaper
    60,  -- Round Worm
    61,  -- Level 2 Fly
    62,  -- Level 2 Spider
    63,  -- Nerve Ending
    64,  -- Camillo Jr.
    65,  -- Mama Gurdy
    66,  -- Trite (dup)
    67,  -- Grub
    68,  -- Larry Jr. seg
    69,  -- Pin
    70,  -- Fistula
    71,  -- Teratoma
    72,  -- Lokii
    73,  -- Nulls (dup)
    74,  -- Membrane
    75,  -- Envy / Sloth
    76,  -- Sloth
    77,  -- Lust
    78,  -- Wrath
    79,  -- Gluttony
    80,  -- Greed
    81,  -- Envy
    82,  -- Pride
    83,  -- Super Envy
    84,  -- Cage
    85,  -- Duke of Flies
    86,  -- The Husk
    87,  -- Larry Jr. (dup)
    88,  -- Krampus
    89,  -- Steven / Blighted Ovum
    90,  -- Mask + Heart
    91,  -- Widow (dup)
    92,  -- Daddy Long Legs (dup)
    93,  -- Isaac (boss version)
    94,  -- Deep Blob
    95,  -- Bee
    96,  -- Fistuloid
    97,  -- Nulls
    98,  -- Baby
    99,  -- Wall Creep
    100, -- Rag Man
    101, -- Uriel
    102, -- Gabriel
    103, -- The Fallen
    104, -- Satan
    105, -- Cyclopia
    106, -- Nulls
    107, -- Nulls
    108, -- Nulls
    109, -- Nulls
    110, -- Nulls
    200, -- Fireplace / grid-adjacent
    201, -- Nulls
    202, -- Nulls
    203, -- Nulls
    204, -- Nulls
    205, -- Nulls
    206, -- Nulls
    207, -- Nulls
    208, -- Nulls
    209, -- Nulls
    210, -- Nulls
    211, -- Nulls
    212, -- Nulls
    213, -- Nulls
    214, -- Nulls
    215, -- Nulls
    216, -- Nulls
    217, -- Nulls
    218, -- Nulls
    219, -- Nulls
    220, -- Nulls
    221, -- Nulls
    222, -- Nulls
    223, -- Nulls
    224, -- Nulls
    225, -- Nulls
    226, -- Nulls
    227, -- Nulls
    228, -- Nulls
    229, -- Nulls
    230, -- Nulls
    231, -- Nulls
    232, -- Nulls
    233, -- Nulls
    234, -- Nulls
    235, -- Nulls
    236, -- Nulls
    237, -- Nulls
    238, -- Nulls
    239, -- Nulls
    240, -- Nulls
    241, -- Nulls
    242, -- Nulls
    243, -- Nulls
    244, -- Nulls
    245, -- Nulls
    246, -- Nulls
    247, -- Nulls
    248, -- Nulls
    249, -- Nulls
    250, -- Nulls
    251, -- Nulls
    252, -- Nulls
    253, -- Nulls
    254, -- Nulls
    255, -- Nulls
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
