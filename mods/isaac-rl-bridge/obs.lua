-- obs.lua — per-tick observation builder.
--
-- Keep this in sync with python/isaac_rl/spaces.py. Fields must appear under the
-- same keys the Python side expects, but missing fields are OK (Python zero-fills).

local Tables = require("tables")

local Obs = {}

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

local DOOR_SLOTS = { DoorSlot.LEFT0, DoorSlot.UP0, DoorSlot.RIGHT0, DoorSlot.DOWN0 }

local function build_doors(room)
    local out = {}
    for i, slot in ipairs(DOOR_SLOTS) do
        local d = room:GetDoor(slot)
        if d then
            out[i] = {
                1,
                (d:IsOpen() and 1) or 0,
                (d:IsLocked() and 1) or 0,
                (d.TargetRoomType == RoomType.ROOM_BOSS and 1) or 0,
                (d.TargetRoomType == RoomType.ROOM_TREASURE and 1) or 0,
                (d.TargetRoomType == RoomType.ROOM_SECRET and 1) or 0,
            }
        else
            out[i] = { 0, 0, 0, 0, 0, 0 }
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

    return {
        schema = 1,
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
            frame_count = player.FrameCount,
            is_dead = player:IsDead(),
        },
        passives = build_passives(player),
        room_grid = build_room_grid(room),
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
