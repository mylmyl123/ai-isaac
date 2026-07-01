-- tables.lua — hand-picked top-K collectible and NPC IDs.
--
-- These lists give us small dense integer indices for compact one-hot encoding.
-- Anything outside the top-K collapses to the "other" bucket (index 0).
--
-- Curated for early-game (basement/caves/depths) since that's where the agent
-- spends >90% of its time on the road to Mom. Expand later.

local M = {}

-- Top 256 collectible IDs commonly found on floors 1-6.
-- Values 1..N are the dense indices; the raw CollectibleType goes as key.
M.COLLECTIBLES = {}
local common_collectibles = {
    1,   -- Sad Onion
    2,   -- The Inner Eye
    3,   -- Spoon Bender
    4,   -- Cricket's Head
    5,   -- My Reflection
    6,   -- Number One
    7,   -- Blood of the Martyr
    8,   -- Brother Bobby
    9,   -- Skatole
    10,  -- Halo of Flies
    11,  -- 1up!
    12,  -- Magic Mushroom
    13,  -- The Virus
    14,  -- Roid Rage
    15,  -- <3
    16,  -- Raw Liver
    17,  -- Skeleton Key
    18,  -- A Dollar
    19,  -- Boom!
    20,  -- Transcendence
    21,  -- The Compass
    22,  -- Lunch
    23,  -- Dinner
    24,  -- Dessert
    25,  -- Breakfast
    26,  -- Rotten Meat
    27,  -- Wooden Spoon
    28,  -- The Belt
    29,  -- Mom's Underwear
    30,  -- Mom's Heels
    31,  -- Mom's Lipstick
    32,  -- Wire Coat Hanger
    33,  -- The Bible
    34,  -- The Book of Belial
    35,  -- The Necronomicon
    36,  -- The Poop
    37,  -- Mr. Boom
    38,  -- Tammy's Head
    39,  -- Mom's Bra
    40,  -- Kamikaze!
    41,  -- Mom's Pad
    42,  -- Bob's Rotten Head
    44,  -- Teleport
    45,  -- Yum Heart
    46,  -- Lucky Foot
    47,  -- Doctor's Remote
    48,  -- Cupid's Arrow
    49,  -- Shoop Da Whoop!
    50,  -- Steven
    51,  -- Pentagram
    52,  -- Dr. Fetus
    53,  -- Magneto
    54,  -- Treasure Map
    55,  -- Mom's Eye
    56,  -- Lemon Mishap
    57,  -- Distant Admiration
    58,  -- Book of Shadows
    59,  -- The Book of Sin (was 59 in AB+; in Rep 59 is different — keep by ID anyway)
    60,  -- The Ladder / Rerolled — safe to include; excess IDs just take slots
    61,  -- Charm of the Vampire
    62,  -- A Quarter
    63,  -- PHD
    64,  -- X-Ray Vision
    65,  -- My Little Unicorn
    66,  -- Book of Revelations
    67,  -- The Mark
    68,  -- The Pact
    69,  -- Dead Cat
    70,  -- Lord of the Pit
    71,  -- The Nail
    72,  -- We Need To Go Deeper!
    73,  -- Deck of Cards
    74,  -- Monstro's Tooth
    75,  -- Loki's Horns
    76,  -- Little Chubby
    77,  -- Spider Bite
    78,  -- The Small Rock
    79,  -- Spelunker Hat
    80,  -- Super Bandage
    81,  -- The Gamekid
    82,  -- Sack of Pennies
    83,  -- Robo-Baby
    84,  -- Little C.H.A.D.
    85,  -- The Book of Sin
    86,  -- Relic
    87,  -- Little Gish
    88,  -- Little Steven
    89,  -- The Halo
    90,  -- Mom's Bottle of Pills
    91,  -- The Common Cold
    92,  -- Parasite
    93,  -- D6
    94,  -- Mr. Mega
    95,  -- Pinking Shears
    96,  -- The Wafer
    97,  -- Money = Power
    98,  -- Mom's Contacts
    99,  -- The Bean
    100, -- Guardian Angel
    101, -- Demon Baby
    102, -- Mom's Knife
    103, -- Ouija Board
    104, -- 9 Volt
    105, -- Dead Bird
    106, -- The Brimstone
    107, -- Blood Bag
    108, -- Odd Mushroom (Thin)
    109, -- Odd Mushroom (Large)
    110, -- Whore of Babylon
    111, -- Monster Manual
    112, -- Dead Sea Scrolls
    113, -- Bobby-Bomb
    114, -- Razor Blade
    115, -- Forget Me Now
    116, -- Forever Alone
    117, -- Bucket of Lard
    118, -- A Pony
    119, -- Book of Revelations (dup safe)
    120, -- Rubber Cement
    121, -- Anti-Gravity
    122, -- Pyromaniac
    123, -- Cricket's Body
    124, -- Gimpy
    125, -- Black Lotus
    126, -- Piggy Bank
    127, -- Mom's Purse
    128, -- Bogo Bombs
    129, -- Starter Deck
    130, -- Little Baggy
    131, -- Magic Scab
    132, -- Blood Clot
    133, -- Screw
    134, -- Hot Bombs
    135, -- IV Bag
    136, -- Best Friend
    137, -- Remote Detonator
    138, -- Stigmata
    139, -- Mom's Purse (dup)
    140, -- Bob's Curse
    141, -- Pageant Boy
    142, -- Scapular
    143, -- Speed Ball
    144, -- Bum Friend
    145, -- Guppy's Head
    146, -- Prayer Card
    147, -- Notched Axe
    148, -- Infestation
    149, -- Ipecac
    150, -- Tough Love
    151, -- The Mulligan
    152, -- Technology 2
    153, -- Mutant Spider
    154, -- Chemical Peel
    155, -- The Peeper
    156, -- Habit
    157, -- Bloody Lust
    158, -- Crystal Ball
    159, -- Spirit of the Night
    160, -- Crack the Sky
    161, -- Ankh
    162, -- Celtic Cross
    163, -- Ghost Baby
    164, -- The Candle
    165, -- Cat-o-nine-tails
    166, -- D20
    167, -- Harlequin Baby
    168, -- Epic Fetus
    169, -- Polyphemus
    170, -- Daddy Longlegs
    171, -- Spider Butt
    172, -- Sacrificial Dagger
    173, -- Mitre
    174, -- Rainbow Baby
    175, -- Dad's Key
    176, -- Stem Cells
    177, -- Portable Slot
    178, -- Holy Water
    179, -- Fate
    180, -- The Black Bean
    181, -- White Pony
    182, -- Sacred Heart
    183, -- Tooth Picks
    184, -- Holy Grail
    185, -- Dead Dove
    186, -- Blood Rights
    187, -- Guppy's Hairball
    188, -- Abel
    189, -- SMB Super Fan
    190, -- Pyro
    191, -- 3-Dollar Bill
    192, -- Telepathy for Dummies
    193, -- MEAT!
    194, -- Magic 8 Ball
    195, -- Mom's Coin Purse
    196, -- Squeezy
    197, -- Jesus Juice
    198, -- Box
    199, -- Mom's Key
    200, -- Mom's Eyeshadow
    201, -- Iron Bar
    202, -- Midas' Touch
    203, -- Humbleing Bundle
    204, -- Fanny Pack
    205, -- Sharp Plug
    206, -- Guillotine
    207, -- Ball of Bandages
    208, -- Champion Belt
    209, -- Butt Bombs
    210, -- Gnawed Leaf
    211, -- Spiderbaby
    212, -- Guppy's Collar
    213, -- Lost Contact
    214, -- Anemic
    215, -- Goat Head
    216, -- Ceremonial Robes
    217, -- Mom's Wig
    218, -- Placenta
    219, -- Old Bandage
    220, -- SAD Bombs
    221, -- Rubber Cement (dup)
    222, -- Anti-Gravity (dup)
    223, -- Pyromaniac (dup)
    224, -- Cricket's Body (dup)
    225, -- Gimpy (dup)
    226, -- Black Lotus (dup)
    227, -- Piggy Bank (dup)
    228, -- Mom's Purse (dup)
    229, -- Mom's Perfume
    230, -- Monstro's Lung
    231, -- Abaddon
    232, -- Ball of Tar
    233, -- Stop Watch
    234, -- Tiny Planet
    235, -- Infestation 2
    237, -- E. Coli
    238, -- Death's Touch
    239, -- Key Piece 1
    240, -- Experimental Treatment
    241, -- Contract from Below
    242, -- Infamy
    243, -- Trinity Shield
    244, -- Tech.5
    245, -- 20/20
    246, -- Blue Map
    247, -- BFFS!
    248, -- Hive Mind
    249, -- There's Options
    250, -- Bogo Bombs (dup)
    251, -- Starter Deck (dup)
    252, -- Little Baggy (dup)
    253, -- Magic Scab (dup)
    254, -- Blood Clot (dup)
    255, -- Screw (dup)
}
for i, cid in ipairs(common_collectibles) do
    M.COLLECTIBLES[cid] = i
end
M.PASSIVES_K = 256

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
