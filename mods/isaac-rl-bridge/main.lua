-- main.lua — Isaac RL bridge entry point.
--
-- Callback flow (30 Hz game clock):
--   MC_POST_GAME_STARTED  — connect socket, handshake, reset per-run state
--   MC_POST_NEW_LEVEL     — record floor descent event; reset per-floor state
--   MC_POST_NEW_ROOM      — record room entry; reset per-room state
--   MC_ENTITY_TAKE_DMG    — reward.lua captures damage events
--   MC_POST_PICKUP_UPDATE — reward.lua captures pedestal grabs
--   MC_POST_UPDATE (30Hz) — every FRAME_SKIP ticks: build obs, send, block for action
--   MC_INPUT_ACTION       — return cached action booleans
--
-- Control rate is 15 Hz by default (FRAME_SKIP=2). Bump to 1 for 30 Hz.

local Net    = require("net")
local Obs    = require("obs")
local Reward = require("reward")
local json   = require("json")

local FRAME_SKIP = 2
local HOST = "127.0.0.1"
local PORT = tonumber(os.getenv("ISAAC_RL_PORT") or "9500")

local mod = RegisterMod("isaac-rl-bridge", 1)
local conn = nil
local tick = 0

-- Per-run state that Python's reward shaper wants to know about.
local run_state = {
    frames_since_room = 0,
    frames_since_hit  = 0,
    visited_rooms = {},          -- set of SafeGridIndex -> true
    visited_rooms_count = 0,
    last_stage = 0,
    last_room_clear = false,
    pending_events = {},          -- non-damage events (new_room, new_level, death, room_clear)
}

local function reset_run_state()
    run_state.frames_since_room = 0
    run_state.frames_since_hit  = 0
    run_state.visited_rooms = {}
    run_state.visited_rooms_count = 0
    run_state.last_stage = 0
    run_state.last_room_clear = false
    run_state.pending_events = {}
    Reward.reset_run()
end

-- Cached action written by Python, read by MC_INPUT_ACTION.
local cached_action = {
    [ButtonAction.ACTION_LEFT] = false,
    [ButtonAction.ACTION_RIGHT] = false,
    [ButtonAction.ACTION_UP] = false,
    [ButtonAction.ACTION_DOWN] = false,
    [ButtonAction.ACTION_SHOOTLEFT] = false,
    [ButtonAction.ACTION_SHOOTRIGHT] = false,
    [ButtonAction.ACTION_SHOOTUP] = false,
    [ButtonAction.ACTION_SHOOTDOWN] = false,
    [ButtonAction.ACTION_BOMB] = false,
    [ButtonAction.ACTION_ITEM] = false,
    [ButtonAction.ACTION_PILLCARD] = false,
    [ButtonAction.ACTION_DROP] = false,
}

-- Decode MultiDiscrete([9, 5, 2, 2, 2]) into ButtonAction booleans.
local mv_table = {
    [1] = {up = true},                       [2] = {up = true, right = true},
    [3] = {right = true},                    [4] = {down = true, right = true},
    [5] = {down = true},                     [6] = {down = true, left = true},
    [7] = {left = true},                     [8] = {up = true, left = true},
}
local function apply_action(a)
    for k in pairs(cached_action) do cached_action[k] = false end
    local m = mv_table[a.move or 0]
    if m then
        cached_action[ButtonAction.ACTION_UP]    = m.up    == true
        cached_action[ButtonAction.ACTION_DOWN]  = m.down  == true
        cached_action[ButtonAction.ACTION_LEFT]  = m.left  == true
        cached_action[ButtonAction.ACTION_RIGHT] = m.right == true
    end
    local s = a.shoot or 0
    if     s == 1 then cached_action[ButtonAction.ACTION_SHOOTUP]    = true
    elseif s == 2 then cached_action[ButtonAction.ACTION_SHOOTRIGHT] = true
    elseif s == 3 then cached_action[ButtonAction.ACTION_SHOOTDOWN]  = true
    elseif s == 4 then cached_action[ButtonAction.ACTION_SHOOTLEFT]  = true
    end
    cached_action[ButtonAction.ACTION_ITEM]     = a.use_active == 1 or a.use_active == true
    cached_action[ButtonAction.ACTION_BOMB]     = a.drop_bomb  == 1 or a.drop_bomb  == true
    cached_action[ButtonAction.ACTION_PILLCARD] = a.pill_card  == 1 or a.pill_card  == true
end

-- Drain damage events, merge with pending non-damage events, return combined list.
local function collect_events()
    local dmg = Reward.drain()
    local pend = run_state.pending_events
    run_state.pending_events = {}
    -- Concatenate.
    for i = 1, #pend do dmg[#dmg + 1] = pend[i] end
    return dmg
end

local function exchange()
    local events = collect_events()
    local obs = Obs.build(tick, events, run_state)
    local ok, payload = pcall(json.encode, obs)
    if not ok then
        Isaac.DebugString("[isaac-rl-bridge] json.encode failed: " .. tostring(payload))
        return
    end
    conn:send(payload)

    local msg = conn:recv()
    if not msg then return end
    local ok2, action = pcall(json.decode, msg)
    if not ok2 or type(action) ~= "table" then
        Isaac.DebugString("[isaac-rl-bridge] json.decode failed: " .. tostring(action))
        return
    end
    if action.reset then
        if action.seed then Isaac.ExecuteCommand("seed " .. tostring(action.seed)) end
        if action.stage then Isaac.ExecuteCommand("stage " .. tostring(action.stage)) end
        Isaac.ExecuteCommand("restart 0")
        return
    end
    apply_action(action)
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, is_continued)
    tick = 0
    reset_run_state()
    if conn then conn:close() end
    conn = Net.connect(HOST, PORT, 0.25)
    local seed = Game():GetSeeds():GetStartSeed()
    conn:send(json.encode({hello = true, schema = 1, seed = seed, is_continued = is_continued}))
    Isaac.DebugString("[isaac-rl-bridge] connected to trainer on port " .. tostring(PORT))
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, function()
    local stage = Game():GetLevel():GetStage()
    run_state.pending_events[#run_state.pending_events + 1] = {
        kind = "new_level", stage = stage,
    }
    Reward.reset_room()
    run_state.last_stage = stage
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    local level = Game():GetLevel()
    local sgi = level:GetCurrentRoomDesc().SafeGridIndex
    local room = Game():GetRoom()
    local is_new = not run_state.visited_rooms[sgi]
    if is_new then
        run_state.visited_rooms[sgi] = true
        run_state.visited_rooms_count = run_state.visited_rooms_count + 1
    end
    run_state.pending_events[#run_state.pending_events + 1] = {
        kind = "new_room",
        safe_grid_index = sgi,
        is_new = is_new,
        room_type = room:GetType(),
    }
    run_state.frames_since_room = 0
    run_state.last_room_clear = room:IsClear()
    Reward.reset_room()
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    tick = tick + 1
    run_state.frames_since_room = run_state.frames_since_room + 1
    run_state.frames_since_hit = run_state.frames_since_hit + 1

    -- Detect room-clear transition (false -> true).
    local room = Game():GetRoom()
    local is_clear = room:IsClear()
    if is_clear and not run_state.last_room_clear then
        run_state.pending_events[#run_state.pending_events + 1] = {
            kind = "room_clear",
            safe_grid_index = Game():GetLevel():GetCurrentRoomDesc().SafeGridIndex,
        }
    end
    run_state.last_room_clear = is_clear

    -- Detect death this frame (Python turns it into terminated=True).
    local player = Isaac.GetPlayer(0)
    if player:IsDead() then
        run_state.pending_events[#run_state.pending_events + 1] = { kind = "death" }
    end

    if (tick % FRAME_SKIP) == 0 and conn then
        exchange()
    end
end)

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
    if entity and entity:ToPlayer() == nil then return nil end
    local pressed = cached_action[action]
    if pressed == nil then return nil end
    if hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then
        return pressed
    elseif hook == InputHook.GET_ACTION_VALUE then
        return pressed and 1.0 or 0.0
    end
    return nil
end)

-- Reset the "frames since hit" counter whenever the player takes damage.
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity)
    if entity and entity:ToPlayer() then
        run_state.frames_since_hit = 0
    end
    return nil
end)

Reward.attach(mod)

-- Auto-start a run from the main menu so training doesn't need a human click.
-- The bridge waits ~30 render frames after Isaac finishes booting so the intro
-- animation / mod-loading has time to settle, then executes `restart 0`.
--
-- We use MC_POST_RENDER (fires every render frame including menus) rather than
-- MC_POST_UPDATE (only fires during a run).
local menu_wait_frames = 0
local AUTO_START_AFTER_FRAMES = 90    -- ~1.5s of intro/menu at 60 render Hz
mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    -- Game():IsPaused() also returns true on the menu, but Room():GetFrameCount()==0
    -- and Game():GetFrameCount()==0 is a more reliable "we're not in a run" check.
    if Game():GetFrameCount() > 0 then
        menu_wait_frames = 0
        return
    end
    menu_wait_frames = menu_wait_frames + 1
    if menu_wait_frames == AUTO_START_AFTER_FRAMES then
        Isaac.DebugString("[isaac-rl-bridge] auto-starting run from menu")
        Isaac.ExecuteCommand("restart 0")   -- 0 = Isaac
    end
end)

Isaac.DebugString("[isaac-rl-bridge] mod loaded (port " .. tostring(PORT) .. ", frame skip " .. FRAME_SKIP .. ")")
