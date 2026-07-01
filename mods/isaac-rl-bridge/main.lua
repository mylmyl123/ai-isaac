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
-- Frames-to-skip counter set after any `restart 0` / big transition. While > 0,
-- MC_POST_UPDATE returns immediately without calling exchange(), preventing us
-- from iterating entity pointers that Isaac is currently tearing down.
local reset_cooldown = 0
-- Applied on the NEXT MC_POST_GAME_STARTED. Lets us defer `stage N` / `seed N`
-- console commands until the new run is fully initialized, avoiding races
-- with `restart 0`.
local pending_stage = nil
local pending_seed = nil

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
    -- Wrap Obs.build in pcall. It calls into Isaac's C bindings (GetRoomEntities,
    -- FindByType, player methods, etc.) which can throw Lua-level errors when
    -- called during game state transitions. Without pcall, a Lua error here
    -- would abort the callback and could crash the interpreter on the next
    -- tick when it retries against still-invalid state.
    local ok_obs, obs = pcall(Obs.build, tick, events, run_state)
    if not ok_obs then
        Isaac.DebugString("[isaac-rl-bridge] Obs.build failed: " .. tostring(obs))
        return
    end
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
        -- Order matters: `restart 0` must run FIRST. It works from any game state
        -- (in-run, death animation, 'You Died' screen, game over). If we run
        -- `stage N` first while the player is dead, Isaac can freeze or crash
        -- because the stage command teleports a corpse rather than starting a
        -- new run.
        Isaac.ExecuteCommand("restart 0")
        -- Only touch stage if we want something other than the default (1).
        -- `restart 0` already boots into Basement 1, so a `stage 1` afterward
        -- is at best redundant and at worst races against the new run's init.
        if action.stage and tonumber(action.stage) and tonumber(action.stage) ~= 1 then
            pending_stage = tonumber(action.stage)
        end
        if action.seed then
            pending_seed = tostring(action.seed)
        end
        reset_cooldown = 20
        return
    end
    apply_action(action)
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, is_continued)
    tick = 0
    reset_run_state()
    -- Fresh run: give Isaac a few ticks before we start iterating entities.
    reset_cooldown = 10
    -- Apply any deferred console commands from the reset that just fired.
    if pending_seed then
        Isaac.ExecuteCommand("seed " .. pending_seed)
        pending_seed = nil
    end
    if pending_stage then
        Isaac.ExecuteCommand("stage " .. tostring(pending_stage))
        pending_stage = nil
    end
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
    -- Level transitions also invalidate entity iteration for a few ticks.
    reset_cooldown = math.max(reset_cooldown, 10)
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
    -- Skip a few ticks so entity refs from the previous room are fully released.
    reset_cooldown = math.max(reset_cooldown, 5)
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    tick = tick + 1
    run_state.frames_since_room = run_state.frames_since_room + 1
    run_state.frames_since_hit = run_state.frames_since_hit + 1

    -- After a `restart 0` we spend a fixed number of ticks NOT calling exchange().
    -- Iterating entities via Isaac.GetRoomEntities()/FindByType() during the
    -- game's teardown/rebuild triggers Lua5.3.3r.dll access-violation crashes
    -- (0xc0000005). Room/level transitions also skip their exchange to be safe.
    if reset_cooldown > 0 then
        reset_cooldown = reset_cooldown - 1
        return
    end

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
    if player and player:IsDead() then
        run_state.pending_events[#run_state.pending_events + 1] = { kind = "death" }
    end

    if (tick % FRAME_SKIP) == 0 and conn then
        -- Extra safety: room must have ticked at least once so its entity
        -- lists are populated, and the player must exist.
        if not player then return end
        if room:GetFrameCount() < 2 then return end
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

-- Auto-start fallback: if the game somehow lands on the main menu (e.g. the
-- `--set-stage=N` launch flag wasn't passed or Isaac ignored it), issue a
-- `restart 0` from the menu to boot into a run.
--
-- IMPORTANT: this must NOT fire during an active run. We use MC_POST_UPDATE
-- to definitively decide "a run is active" — that callback only fires while a
-- run is running (not on the menu, not during the intro cinematic, and
-- critically NOT while the game is paused-on-focus-loss). Using MC_POST_RENDER
-- alone is unsafe: render frames keep advancing while the game is paused, so
-- a purely-render-based watchdog will eventually fire `restart 0` on top of a
-- live run whose owner just tabbed away, killing the socket.
local auto_start_fired = false
local render_frames_on_menu = 0
local AUTO_START_AFTER_FRAMES = 600   -- ~10s at 60 render Hz — well past logos + intro

-- Death auto-restart. When the player dies, Isaac shows a death animation and
-- then the 'You Died' screen. During that screen MC_POST_UPDATE stops firing,
-- which means the mod's exchange() loop stalls — the death event was sent to
-- Python and Python's reset command is sitting in the socket buffer, but the
-- mod can't process it. Without a fallback here training deadlocks: game
-- waits for input, Python waits for the next obs, neither will happen.
--
-- Fix: MC_POST_RENDER fires continuously (including during 'You Died'). If we
-- see the player dead for a fixed number of render frames, fire `restart 0`
-- ourselves. Python's reset command will still be processed on the new run's
-- MC_POST_GAME_STARTED handler (which reconnects a fresh socket), so the
-- trainer stays in lockstep.
local death_render_frames = 0
local DEATH_AUTO_RESTART_FRAMES = 120   -- ~2s at 60 render Hz

-- MC_POST_UPDATE fires only during an active, unpaused run. The instant we see
-- one, we know the run is live and disable the fallback permanently.
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    if not auto_start_fired then
        auto_start_fired = true
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    -- Menu auto-start branch (only runs before any MC_POST_UPDATE ever fired).
    if not auto_start_fired then
        if Game():GetFrameCount() > 0 then
            auto_start_fired = true
        else
            render_frames_on_menu = render_frames_on_menu + 1
            if render_frames_on_menu == AUTO_START_AFTER_FRAMES then
                auto_start_fired = true
                Isaac.DebugString("[isaac-rl-bridge] auto-start fallback firing (main menu detected after intro)")
                Isaac.ExecuteCommand("restart 0")
            end
        end
        return
    end

    -- Death auto-restart branch (only relevant after a run has been active).
    local ok, player = pcall(Isaac.GetPlayer, 0)
    if not ok or not player then
        death_render_frames = 0
        return
    end
    if player:IsDead() then
        death_render_frames = death_render_frames + 1
        if death_render_frames == DEATH_AUTO_RESTART_FRAMES then
            Isaac.DebugString("[isaac-rl-bridge] player dead too long — forcing restart 0")
            Isaac.ExecuteCommand("restart 0")
            death_render_frames = 0   -- reset; MC_POST_GAME_STARTED will fire
        end
    else
        death_render_frames = 0
    end
end)

Isaac.DebugString("[isaac-rl-bridge] mod loaded (port " .. tostring(PORT) .. ", frame skip " .. FRAME_SKIP .. ")")
