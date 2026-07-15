-- obs.lua — per-tick observation builder.
--
-- Keep this in sync with python/isaac_rl/spaces.py. Fields must appear under the
-- same keys the Python side expects, but missing fields are OK (Python zero-fills).

local Tables = require("tables")

local Obs = {}

-- Vanilla Repentance vs REPENTOGON: some Isaac API methods (Player:GetActiveMaxCharge,
-- Player:GetCard(1..3), Player:GetPill(1..3), Player:GetPlayerFormCounter, etc.)
-- are REPENTOGON-only. Calling them on vanilla with `player:Method()` throws
-- 'attempt to call a nil value'. safe_get wraps a member function so a missing
-- method degrades to a caller-specified default instead of crashing Obs.build.
--
-- Usage: safe_get(default, fn, self, arg1, arg2, ...)
--   e.g. safe_get(0, player.GetActiveMaxCharge, player, 0) is roughly
--        pcall(function() return player:GetActiveMaxCharge(0) end)
--
-- Note we pass `player.Method` (dot, not colon) + `player` as first arg so
-- pcall calls it correctly. If `player.Method` is nil, pcall catches the
-- 'attempt to call nil' immediately with no error propagating up.
local function safe_get(default, fn, ...)
    if type(fn) ~= "function" then return default end
    local ok, res = pcall(fn, ...)
    if ok and res ~= nil then return res end
    return default
end

-- Room interior is 480x270 world-units per Repentance conventions.
-- We normalize positions to be roughly [0, 1] against the room bounding box for policy stability.
local function room_bounds(room)
    local tl = room:GetTopLeftPos()
    local br = room:GetBottomRightPos()
    return tl, br
end

local function normalize_pos(pos, tl, br)
    local w = math.max(1.0, br.X - tl.X)
    local h = math.max(1.0, br.Y - tl.Y)
    return (pos.X - tl.X) / w, (pos.Y - tl.Y) / h
end

-- Build the enemies feature array. Fixed-size (24) with mask, so Python can pad.
local MAX_ENEMIES = 24
local MAX_PROJ    = 48
local MAX_PICKUPS = 16

local function build_enemies(room, player, tl, br)
    local out = {}
    local mask = {}
    local player_x, player_y = player.Position.X, player.Position.Y
    local n = 0
    local entities = Isaac.GetRoomEntities()
    for _, e in ipairs(entities) do
        if n >= MAX_ENEMIES then break end
        local npc = e:ToNPC()
        if npc and npc:IsVulnerableEnemy() and not npc:IsDead() then
            local nx, ny = normalize_pos(e.Position, tl, br)
            local rvx, rvy = e.Velocity.X, e.Velocity.Y
            local hp = e.HitPoints
            local max_hp = math.max(1.0, e.MaxHitPoints)
            local type_idx = Tables.NPC_TYPES[e.Type] or 0
            out[n + 1] = {
                nx, ny,
                (e.Position.X - player_x) / 480.0,
                (e.Position.Y - player_y) / 270.0,
                rvx / 10.0, rvy / 10.0,
                hp / max_hp,
                (npc:IsBoss() and 1) or 0,
                -- Flying detection: Isaac has NO `EntityFlag.FLAG_FLYING` — that
                -- reference throws 'bad argument #1 (number expected, got nil)'
                -- and blows up the whole Obs.build. Flying entities are
                -- identified by their grid-collision class: flyers use
                -- GRIDCOLL_NONE (value 0) so they don't collide with rocks/pits.
                ((e.GridCollisionClass or 5) == 0 and 1) or 0,
                (npc:IsChampion() and 1) or 0,
                type_idx,
                npc.State or 0,
                e.Size or 0,
                e.SpriteScale.X or 1.0,
                e.FrameCount or 0,
                0,  -- reserved
            }
            mask[n + 1] = 1
            n = n + 1
        end
    end
    return out, mask, n
end

local function build_projectiles(room, player, tl, br)
    local out = {}
    local mask = {}
    local player_x, player_y = player.Position.X, player.Position.Y
    local n = 0

    -- Projectiles (enemy tears)
    for _, e in ipairs(Isaac.FindByType(EntityType.ENTITY_PROJECTILE, -1, -1, false, false)) do
        if n >= MAX_PROJ then break end
        local nx, ny = normalize_pos(e.Position, tl, br)
        out[n + 1] = {
            nx, ny,
            (e.Position.X - player_x) / 480.0,
            (e.Position.Y - player_y) / 270.0,
            e.Velocity.X / 10.0, e.Velocity.Y / 10.0,
            (e.PositionOffset and e.PositionOffset.Y or 0) / 100.0,  -- height-ish
            0,                       -- is_laser
            e.Variant or 0,
            (e.FrameCount or 0) / 30.0,
        }
        mask[n + 1] = 1
        n = n + 1
    end

    -- Player's OWN tears (ENTITY_TEAR = 2). Post 2026-07-14 (Phase 1): this
    -- was MISSING from obs, meaning the agent could not see whether its
    -- fired tears would hit the target. Critical for an aim-and-shoot task.
    -- We flag them via Variant field being reserved above 100 so downstream
    -- can distinguish own tears from enemy projectiles.
    for _, e in ipairs(Isaac.FindByType(EntityType.ENTITY_TEAR, -1, -1, false, false)) do
        if n >= MAX_PROJ then break end
        local nx, ny = normalize_pos(e.Position, tl, br)
        out[n + 1] = {
            nx, ny,
            (e.Position.X - player_x) / 480.0,
            (e.Position.Y - player_y) / 270.0,
            e.Velocity.X / 10.0, e.Velocity.Y / 10.0,
            (e.PositionOffset and e.PositionOffset.Y or 0) / 100.0,
            0,                       -- is_laser
            (e.Variant or 0) + 1000, -- +1000 marker: this is our own tear
            (e.FrameCount or 0) / 30.0,
        }
        mask[n + 1] = 1
        n = n + 1
    end

    -- Lasers count as projectiles too.
    for _, e in ipairs(Isaac.FindByType(EntityType.ENTITY_LASER, -1, -1, false, false)) do
        if n >= MAX_PROJ then break end
        local nx, ny = normalize_pos(e.Position, tl, br)
        out[n + 1] = {
            nx, ny,
            (e.Position.X - player_x) / 480.0,
            (e.Position.Y - player_y) / 270.0,
            0, 0,     -- lasers don't have meaningful velocity
            0,        -- height
            1,        -- is_laser
            e.Variant or 0,
            (e.FrameCount or 0) / 30.0,
        }
        mask[n + 1] = 1
        n = n + 1
    end

    return out, mask, n
end

local function build_pickups(room, player, tl, br)
    local out = {}
    local mask = {}
    local player_x, player_y = player.Position.X, player.Position.Y
    local n = 0
    for _, e in ipairs(Isaac.FindByType(EntityType.ENTITY_PICKUP, -1, -1, false, false)) do
        if n >= MAX_PICKUPS then break end
        local pickup = e:ToPickup()
        if pickup and pickup.SubType ~= 0 then
            local nx, ny = normalize_pos(e.Position, tl, br)
            local kind = Tables.PICKUP_KIND[e.Variant] or 0
            out[n + 1] = {
                nx, ny,
                (e.Position.X - player_x) / 480.0,
                (e.Position.Y - player_y) / 270.0,
                kind,
                e.SubType or 0,
                (pickup.Timeout or 0) / 100.0,
                (pickup.Price or 0) / 20.0,
            }
            mask[n + 1] = 1
            n = n + 1
        end
    end
    return out, mask, n
end

-- 4-channel room grid: {wall/pit, rock, spike/fire, poop/tnt}
-- We flatten each channel to a 9*15=135 float array; Python reshapes to (4,9,15).
--
-- The four channel arrays are module-level and reused across calls to cut Lua
-- garbage. Each Obs.build used to allocate 4 fresh 135-slot tables; at 15 Hz
-- that was ~8100 table slots/sec churned through Lua's GC. Reusing keeps the
-- allocation footprint per exchange near-zero for this section.
local _grid_W, _grid_H = 15, 9
local _grid_N = _grid_W * _grid_H
local _walls  = {}
local _rocks  = {}
local _spikes = {}
local _poop   = {}
for i = 1, _grid_N do _walls[i] = 0; _rocks[i] = 0; _spikes[i] = 0; _poop[i] = 0 end

local function build_room_grid(room)
    local W, H, n = _grid_W, _grid_H, _grid_N
    local walls, rocks, spikes, poop = _walls, _rocks, _spikes, _poop
    -- Zero the reused arrays in place.
    for i = 1, n do walls[i] = 0; rocks[i] = 0; spikes[i] = 0; poop[i] = 0 end

    -- The interior grid of a 1x1 room starts at grid index (W+2)+1 with a border.
    -- Iterate over every grid entity and place it into the 9x15 interior grid.
    local grid_size = room:GetGridSize()
    local room_w = room:GetGridWidth()
    for i = 0, grid_size - 1 do
        local ge = room:GetGridEntity(i)
        if ge then
            -- Convert grid index -> interior x,y (skip 1-tile border).
            local gx = i % room_w
            local gy = math.floor(i / room_w)
            local ix = gx - 1
            local iy = gy - 1
            if ix >= 0 and ix < W and iy >= 0 and iy < H then
                local pos = iy * W + ix + 1
                local t = ge:GetType()
                if t == GridEntityType.GRID_WALL or t == GridEntityType.GRID_PIT then
                    walls[pos] = 1
                elseif t == GridEntityType.GRID_ROCK or t == GridEntityType.GRID_ROCKT
                    or t == GridEntityType.GRID_ROCK_BOMB or t == GridEntityType.GRID_ROCK_ALT
                    or t == GridEntityType.GRID_ROCK_SS or t == GridEntityType.GRID_ROCK_SPIKED
                    or t == GridEntityType.GRID_ROCKB then
                    rocks[pos] = 1
                elseif t == GridEntityType.GRID_SPIKES or t == GridEntityType.GRID_SPIKES_ONOFF
                    or t == GridEntityType.GRID_FIREPLACE then
                    spikes[pos] = 1
                elseif t == GridEntityType.GRID_POOP or t == GridEntityType.GRID_TNT then
                    poop[pos] = 1
                end
            end
        end
    end

    -- Return the reused arrays wrapped in a fresh 4-key table. The wrapper
    -- table itself is small; json.encode reads the four keys without
    -- allocating additional storage. Do NOT mutate these arrays elsewhere.
    return { walls = walls, rocks = rocks, spikes = spikes, poop = poop }
end

-- =====================================================================
-- Egocentric multi-channel grid (2026-07-15 architecture rebuild).
--
-- Rasterizes EVERYTHING (player, enemies, enemy projectiles, own tears,
-- pickups, terrain) into a player-CENTERED 14-channel 21x21 image so a CNN
-- can read spatial relationships (aim direction, incoming fire, cover, walls)
-- that the old flat entity-list obs could not represent. The player is fixed
-- at the exact center cell (CENTER,CENTER); the crop translates with the
-- player, so the same relative geometry always lands on the same cells
-- (translation-equivariant -> aiming is a near-linear conv readout).
--
-- CELL_PX=16 (half-tile), CROP=21 -> a 336x336 world-px window. Room interior
-- is 480x270, so the window covers the full height + the aim-relevant width;
-- entities beyond the crop clamp to the border ring (presence only).
--
-- Channels (0-based, matches python/isaac_rl/spaces.py _decode_ego_grid):
--   0 player_self         1.0 at center only
--   1 enemy_presence      min(count/3, 1)
--   2 enemy_hp_frac       hp/max_hp of nearest-in-cell enemy
--   3 enemy_threat        max(boss*1 + champion*0.6 + flying*0.3)
--   4 enemy_vel_x         nearest-in-cell enemy vx/10 clip[-1,1]
--   5 enemy_vel_y         nearest-in-cell enemy vy/10 clip[-1,1]
--   6 enemy_proj_presence 1.0 where an enemy projectile/laser is
--   7 enemy_proj_vel_x    nearest-in-cell proj vx/12 clip[-1,1]
--   8 enemy_proj_vel_y    nearest-in-cell proj vy/12 clip[-1,1]
--   9 own_tear_presence   1.0 where player's own tears are
--  10 pickup_presence     1.0 where a pickup is
--  11 wall_pit            1.0 static wall/pit
--  12 rock                1.0 rocks (all GRID_ROCK* variants)
--  13 hazard_poop         1.0 spikes/fire OR poop/tnt (merged)
--
-- Reused module-level buffers (zero in place) to avoid GC churn, mirroring
-- build_room_grid. 14 channel arrays + one nearest-distance scratch array.
local EGO_C, EGO_G = 14, 21
local EGO_N = EGO_G * EGO_G          -- 441
local EGO_CELL_PX = 16.0
local EGO_CENTER = 10                -- 0-based center index (21//2)
local _ego = {}
for c = 1, EGO_C do
    _ego[c] = {}
    for i = 1, EGO_N do _ego[c][i] = 0 end
end
local _ego_bestd = {}               -- per-cell nearest-dist scratch (value channels)
for i = 1, EGO_N do _ego_bestd[i] = math.huge end

local function _clip1(v)
    if v > 1.0 then return 1.0 elseif v < -1.0 then return -1.0 else return v end
end

-- world (ex,ey) relative to player (px,py) -> 1-based flat index, or nil if OOR.
-- clamp=true snaps out-of-range to the border ring (for presence channels).
local function _ego_cell(ex, ey, px, py, clamp)
    local cx = math.floor((ex - px) / EGO_CELL_PX + 0.5) + EGO_CENTER
    local cy = math.floor((ey - py) / EGO_CELL_PX + 0.5) + EGO_CENTER
    if cx < 0 or cx >= EGO_G or cy < 0 or cy >= EGO_G then
        if not clamp then return nil end
        if cx < 0 then cx = 0 elseif cx >= EGO_G then cx = EGO_G - 1 end
        if cy < 0 then cy = 0 elseif cy >= EGO_G then cy = EGO_G - 1 end
    end
    return cy * EGO_G + cx + 1
end

local function build_ego_grid(room, player)
    local px, py = player.Position.X, player.Position.Y
    -- Zero channels + scratch in place.
    for c = 1, EGO_C do
        local ch = _ego[c]
        for i = 1, EGO_N do ch[i] = 0 end
    end
    for i = 1, EGO_N do _ego_bestd[i] = math.huge end

    -- ch0 player_self at center.
    _ego[1][EGO_CENTER * EGO_G + EGO_CENTER + 1] = 1.0

    -- Enemies + own tears + enemy projectiles + pickups from one entity scan.
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        local et = e.Type
        local ep = e.Position
        if et == EntityType.ENTITY_TEAR then
            -- player's own tear (ch9 presence)
            local idx = _ego_cell(ep.X, ep.Y, px, py, true)
            if idx then _ego[10][idx] = 1.0 end
        elseif et == EntityType.ENTITY_PROJECTILE or et == EntityType.ENTITY_LASER then
            -- enemy projectile (ch6 presence + ch7/8 nearest-wins velocity)
            local pidx = _ego_cell(ep.X, ep.Y, px, py, true)
            if pidx then
                _ego[7][pidx] = 1.0
                local vidx = _ego_cell(ep.X, ep.Y, px, py, false)  -- value only if in-range
                if vidx then
                    local dx = ep.X - px; local dy = ep.Y - py
                    local d = dx * dx + dy * dy
                    if d < _ego_bestd[vidx] then
                        _ego_bestd[vidx] = d
                        _ego[8][vidx] = _clip1((e.Velocity.X or 0) / 12.0)
                        _ego[9][vidx] = _clip1((e.Velocity.Y or 0) / 12.0)
                    end
                end
            end
        elseif et == EntityType.ENTITY_PICKUP then
            local idx = _ego_cell(ep.X, ep.Y, px, py, true)
            if idx then _ego[11][idx] = 1.0 end
        else
            local npc = e:ToNPC()
            if npc and npc:IsVulnerableEnemy() and not npc:IsDead() then
                local pidx = _ego_cell(ep.X, ep.Y, px, py, true)
                if pidx then
                    -- ch1 enemy_presence (additive, clamped to 1 at /3)
                    local pv = _ego[2][pidx] + (1.0 / 3.0)
                    _ego[2][pidx] = (pv > 1.0) and 1.0 or pv
                    -- value channels: nearest-to-player wins
                    local dx = ep.X - px; local dy = ep.Y - py
                    local d = dx * dx + dy * dy
                    if d < _ego_bestd[pidx] then
                        _ego_bestd[pidx] = d
                        local max_hp = math.max(1.0, e.MaxHitPoints)
                        _ego[3][pidx] = e.HitPoints / max_hp
                        local threat = 0.0
                        if npc:IsBoss() then threat = threat + 1.0 end
                        if npc:IsChampion() then threat = threat + 0.6 end
                        if (e.GridCollisionClass or 5) == 0 then threat = threat + 0.3 end
                        _ego[4][pidx] = (threat > 1.0) and 1.0 or threat
                        _ego[5][pidx] = _clip1((e.Velocity.X or 0) / 10.0)
                        _ego[6][pidx] = _clip1((e.Velocity.Y or 0) / 10.0)
                    end
                end
            end
        end
    end

    -- Terrain, rasterized EGOCENTRICALLY (the key change vs build_room_grid):
    -- map each grid entity's world center through the same transform.
    local grid_size = room:GetGridSize()
    for i = 0, grid_size - 1 do
        local ge = room:GetGridEntity(i)
        if ge then
            local ok, wp = pcall(function() return room:GetGridPosition(i) end)
            if ok and wp then
                local idx = _ego_cell(wp.X, wp.Y, px, py, false)
                if idx then
                    local t = ge:GetType()
                    if t == GridEntityType.GRID_WALL or t == GridEntityType.GRID_PIT then
                        _ego[12][idx] = 1.0
                    elseif t == GridEntityType.GRID_ROCK or t == GridEntityType.GRID_ROCKT
                        or t == GridEntityType.GRID_ROCK_BOMB or t == GridEntityType.GRID_ROCK_ALT
                        or t == GridEntityType.GRID_ROCK_SS or t == GridEntityType.GRID_ROCK_SPIKED
                        or t == GridEntityType.GRID_ROCKB then
                        _ego[13][idx] = 1.0
                    elseif t == GridEntityType.GRID_SPIKES or t == GridEntityType.GRID_SPIKES_ONOFF
                        or t == GridEntityType.GRID_FIREPLACE
                        or t == GridEntityType.GRID_POOP or t == GridEntityType.GRID_TNT then
                        _ego[14][idx] = 1.0
                    end
                end
            end
        end
    end

    -- Return the 14 reused channel arrays wrapped fresh (channel-major).
    -- Python reshapes to (14, 21, 21). Do NOT mutate elsewhere.
    return {
        _ego[1], _ego[2], _ego[3], _ego[4], _ego[5], _ego[6], _ego[7],
        _ego[8], _ego[9], _ego[10], _ego[11], _ego[12], _ego[13], _ego[14],
    }
end

local DOOR_SLOTS = { DoorSlot.LEFT0, DoorSlot.UP0, DoorSlot.RIGHT0, DoorSlot.DOWN0 }

-- 2026-07-12 Track A: expanded from 6 feats [exists, open, locked, boss,
-- treasure, secret] to 18 feats [exists, open, locked, then 15 one-hot
-- flags for room types (boss, treasure, secret, shop, arcade, curse,
-- sacrifice, devil, angel, library, miniboss, challenge, dungeon,
-- planetarium, chest)]. The prior 3-of-15 encoding meant the agent could
-- never distinguish e.g. shop from curse room via the door — both looked
-- like plain "other".
--
-- Repentance API exposes most RoomType.ROOM_* constants but a few (LIBRARY,
-- DUNGEON) are missing on some builds. Use numeric literals with TODO for
-- those; verify with a real playthrough that the one-hots fire correctly.
local function build_doors(room)
    local out = {}
    for i, slot in ipairs(DOOR_SLOTS) do
        local d = room:GetDoor(slot)
        if d then
            local t = d.TargetRoomType
            out[i] = {
                1,
                (d:IsOpen() and 1) or 0,
                (d:IsLocked() and 1) or 0,
                (t == RoomType.ROOM_BOSS and 1) or 0,
                (t == RoomType.ROOM_TREASURE and 1) or 0,
                (t == RoomType.ROOM_SECRET and 1) or 0,
                (t == RoomType.ROOM_SHOP and 1) or 0,
                (t == RoomType.ROOM_ARCADE and 1) or 0,
                (t == RoomType.ROOM_CURSE and 1) or 0,
                (t == RoomType.ROOM_SACRIFICE and 1) or 0,
                (t == RoomType.ROOM_DEVIL and 1) or 0,
                (t == RoomType.ROOM_ANGEL and 1) or 0,
                (t == 8 and 1) or 0,   -- LIBRARY (TODO: verify RoomType.ROOM_LIBRARY constant)
                (t == RoomType.ROOM_MINIBOSS and 1) or 0,
                (t == RoomType.ROOM_CHALLENGE and 1) or 0,
                (t == 16 and 1) or 0,  -- DUNGEON (TODO: verify RoomType.ROOM_DUNGEON constant)
                (t == RoomType.ROOM_PLANETARIUM and 1) or 0,
                (t == RoomType.ROOM_CHEST and 1) or 0,
            }
        else
            out[i] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        end
    end
    return out
end

local function build_passives(player)
    -- Return a sparse list of dense indices into the Passives one-hot.
    -- Python turns this into a MultiBinary(256) vector.
    local out = {}
    for cid, idx in pairs(Tables.COLLECTIBLES) do
        if player:HasCollectible(cid) then
            out[#out + 1] = idx
        end
    end
    return out
end

function Obs.build(tick, reward_events, run_state)
    local game = Game()
    local level = game:GetLevel()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)
    local pos = player.Position
    local vel = player.Velocity
    local tl, br = room_bounds(room)

    local enemies, e_mask, n_enemies = build_enemies(room, player, tl, br)
    local proj,    p_mask, n_proj    = build_projectiles(room, player, tl, br)
    local pickups, k_mask, n_pickups = build_pickups(room, player, tl, br)
    -- Egocentric CNN grid (2026-07-15 rebuild). Built from its own entity scan
    -- so the legacy flat lists above stay intact until the probe gate passes.
    local ego_ok, ego_grid = pcall(build_ego_grid, room, player)
    if not ego_ok then
        Isaac.DebugString("[isaac-rl-bridge] build_ego_grid failed: " .. tostring(ego_grid))
        ego_grid = nil
    end

    return {
        schema = 2,
        tick = tick,
        player = {
            x = pos.X, y = pos.Y,
            vx = vel.X, vy = vel.Y,
            hp_red = player:GetHearts(),
            hp_soul = player:GetSoulHearts(),
            hp_black = player:GetBlackHearts(),
            hp_max = player:GetMaxHearts(),
            keys = player:GetNumKeys(),
            bombs = player:GetNumBombs(),
            coins = player:GetNumCoins(),
            damage = player.Damage,
            fire_delay = player.MaxFireDelay,
            move_speed = player.MoveSpeed,
            tear_range = player.TearRange,
            shot_speed = player.ShotSpeed,
            luck = player.Luck,
            can_shoot = player:CanShoot(),
            -- Fire cooldown (Phase 2, 2026-07-14): FireCooldown is the number
            -- of frames until the player can fire the next tear (0 = ready).
            -- MaxFireDelay is the constant period. The RL agent previously saw
            -- only can_shoot (a bare bool) and MaxFireDelay (constant), so it
            -- could not learn "wait N frames then fire" timing on the
            -- aim-and-shoot task. We emit the raw countdown; the Python decoder
            -- normalizes it by MaxFireDelay. safe_get so a missing field on
            -- some Repentance build degrades to 0 instead of crashing Obs.build.
            fire_cooldown = safe_get(0, function() return player.FireCooldown end),
            frame_count = player.FrameCount,
            is_dead = player:IsDead(),
            -- ADDED 2026-07-12 for BC recording (Track A obs rehab).
            -- All new fields are optional — the RL trainer's encode_obs
            -- (schema v2) silently ignores them. The BC training loader
            -- and, later, an expanded encoder will consume them.
            --
            -- Vanilla Repentance vs REPENTOGON: many of the getters below
            -- (GetActiveMaxCharge, GetCard(1..3), GetPill(1..3),
            -- GetPlayerFormCounter) are REPENTOGON-only extensions and
            -- crash with 'attempt to call a nil value' on vanilla builds.
            -- Every new field is now wrapped via safe_get() so a missing
            -- method degrades to 0 instead of killing the mod. For
            -- active_max_charge specifically we fall back to Isaac's
            -- ItemConfig database, which IS vanilla-safe.
            --
            -- Character identity. Repentance has 34 characters (Isaac=0,
            -- Magdalene=1, Cain=2, Judas=3, ???=4, Eve=5, Samson=6, Azazel=7,
            -- Lazarus=8, Eden=9, The Lost=10, Lazarus Risen=11, Black Judas=12,
            -- Lilith=13, Keeper=14, Apollyon=15, The Forgotten=16, The Soul=17,
            -- Bethany=18, Jacob=19, Esau=20, then 21-33 = Tainted variants).
            -- Every character has different base HP, damage, active item,
            -- passives. Without this the BC actor averages over characters
            -- and Lilith (0 base damage) collapses to Isaac (3.5 damage).
            player_type = safe_get(0, player.GetPlayerType, player),
            -- Active item: primary slot (space bar). GetActiveItem returns 0
            -- when no active is held. GetActiveCharge and GetActiveMaxCharge
            -- expose the charge bar so BC can learn 'save the D6 for the
            -- shop pedestal' vs 'use it now on the cursed pedestal'.
            active_item_id = safe_get(0, player.GetActiveItem, player, 0),
            active_charge = safe_get(0, player.GetActiveCharge, player, 0),
            -- GetActiveMaxCharge is REPENTOGON-only. Fall back to ItemConfig
            -- (vanilla-safe) using the active_item_id we just read.
            active_max_charge = (function()
                local v = safe_get(nil, player.GetActiveMaxCharge, player, 0)
                if v ~= nil then return v end
                -- Fallback: read MaxCharges from Isaac.GetItemConfig().
                local id = safe_get(0, player.GetActiveItem, player, 0)
                if id and id > 0 then
                    local cfg = safe_get(nil, function()
                        return Isaac.GetItemConfig():GetCollectible(id)
                    end)
                    if cfg and cfg.MaxCharges then return cfg.MaxCharges end
                end
                return 0
            end)(),
            -- Secondary active slot (Schoolbag = 2 active items).
            active_item_id_2 = safe_get(0, player.GetActiveItem, player, 1),
            active_charge_2 = safe_get(0, player.GetActiveCharge, player, 1),
            -- Trinket slot (2 in Repentance if you have Mom's Purse).
            trinket_id_1 = safe_get(0, player.GetTrinket, player, 0),
            trinket_id_2 = safe_get(0, player.GetTrinket, player, 1),
            -- Card / pill slot. Vanilla Repentance only exposes slot 0;
            -- REPENTOGON adds slots 1..3. safe_get returns 0 for missing.
            card_id_1 = safe_get(0, player.GetCard, player, 0),
            card_id_2 = safe_get(0, player.GetCard, player, 1),
            card_id_3 = safe_get(0, player.GetCard, player, 2),
            card_id_4 = safe_get(0, player.GetCard, player, 3),
            pill_id_1 = safe_get(0, player.GetPill, player, 0),
            pill_id_2 = safe_get(0, player.GetPill, player, 1),
            pill_id_3 = safe_get(0, player.GetPill, player, 2),
            pill_id_4 = safe_get(0, player.GetPill, player, 3),
            -- Transformation progress. Repentance has 15 forms indexed 0..14
            -- (Guppy=0, Beelzebub=1, Fun Guy=2, Seraphim=3, Bob=4, Spun=5,
            -- Yes Mother=6, Conjoined=7, Leviathan=8, Oh Crap=9, Bookworm=10,
            -- Adult=11, Spider Baby=12, Stompy=13, Super Bum=14). Each
            -- returns 0-N counter of transformation items collected.
            -- GetPlayerFormCounter is REPENTOGON-only; all-zero on vanilla.
            transformations = (function()
                local t = {}
                for i = 0, 14 do
                    t[#t + 1] = safe_get(0, player.GetPlayerFormCounter, player, i)
                end
                return t
            end)(),
        },
        passives = build_passives(player),
        room_grid = build_room_grid(room),
        ego_grid = ego_grid,   -- 14-channel egocentric grid (nil if build failed)
        doors = build_doors(room),
        enemies = { feats = enemies, mask = e_mask, count = n_enemies },
        projectiles = { feats = proj, mask = p_mask, count = n_proj },
        pickups = { feats = pickups, mask = k_mask, count = n_pickups },
        -- Room geometry (added 2026-07-02 for spatial-features obs). Uses
        -- the same tl/br as the entity normalizers so Python can compute
        -- player_normalized_position + wall distances + door directions
        -- deterministically. Fields:
        --   tl_x, tl_y: top-left world coords of the room's playable area
        --   br_x, br_y: bottom-right world coords
        -- Backward compatible: older Python clients that don't read this
        -- field simply ignore it.
        room_bounds = { tl_x = tl.X, tl_y = tl.Y, br_x = br.X, br_y = br.Y },
        global = {
            stage = level:GetStage(),
            stage_type = level:GetStageType(),
            room_index = level:GetCurrentRoomIndex(),
            safe_grid_index = level:GetCurrentRoomDesc().SafeGridIndex,
            room_type = room:GetType(),
            is_clear = room:IsClear(),
            curses = level:GetCurses(),
            frames_since_room = run_state.frames_since_room or 0,
            frames_since_hit = run_state.frames_since_hit or 0,
            visited_rooms = run_state.visited_rooms_count or 0,
        },
        events = reward_events or {},
    }
end

return Obs
