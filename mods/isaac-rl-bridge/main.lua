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

-- Minimal-mode: when ISAAC_RL_MINIMAL is set, the mod strips its behaviour to
-- the absolute bare minimum needed to reproduce/rule-out Isaac engine crashes:
--   * NO exchange() with Python. No sockets. No JSON.
--   * NO Obs.build entity iteration.
--   * NO reward events, no MC_ENTITY_TAKE_DMG hooks, no PICKUP hooks.
--   * ONLY the two runtime-critical callbacks left:
--       - MC_POST_UPDATE: watch for player death and restart cleanly.
--       - MC_INPUT_ACTION: still installed but always returns nil.
-- Purpose: if Isaac still crashes at isaac-ng.exe 0x003a93b5 with this mode
-- ON, the crash is fundamental to Isaacs restart cycle on your machine and
-- has nothing to do with our RL data path. If the crash STOPS with this mode
-- on, we've bisected the fault into either Obs.build, apply_action, or the
-- reward hooks, and can re-enable them one by one to find the culprit.
local MINIMAL_MODE = os.getenv("ISAAC_RL_MINIMAL") == "1"
if MINIMAL_MODE then
    Isaac.DebugString("[isaac-rl-bridge] MINIMAL_MODE=1 — running with training I/O disabled")
end

-- ISAAC_RL_NO_ONESHOT: when set to "1", hardwires the mod to IGNORE pill_card,
-- drop_bomb, and use_active in Python actions. Character will still move and
-- shoot normally. Purpose: bisect whether crashes are caused by one-shot input
-- events (pill uses, bomb drops, active-item uses) or by something else in
-- the data path. If Isaac stops crashing with this flag on, we know one-shot
-- inputs are the problem. If crashes continue, the culprit is elsewhere
-- (probably Obs.build, entity iteration, or the network path).
local NO_ONESHOT = os.getenv("ISAAC_RL_NO_ONESHOT") == "1"
if NO_ONESHOT then
    Isaac.DebugString("[isaac-rl-bridge] NO_ONESHOT=1 — pill_card, drop_bomb, use_active will be ignored")
end

-- ISAAC_RL_RECORD: when set to "1", switches the mod into HUMAN DEMO RECORDING
-- mode (for BC-bootstrap training data collection). In this mode:
--   * MC_INPUT_ACTION returns nil so the human's keyboard/gamepad input
--     passes through to Isaac unmodified (agent is NOT driving the game).
--   * On every FRAME_SKIP tick, exchange() reads the human's current input
--     state via Input.IsActionPressed, packages it as a MultiDiscrete([9,5])
--     action tuple (same schema as the RL agent's output), attaches it to
--     the obs payload, sends to Python, and does NOT block for a response.
--   * apply_action is never called — no injected keys, no crash surface
--     from one-shot events.
-- The Python side (isaac_rl.record) accepts the stream and writes one JSONL
-- line per tick to demos/session_<ts>.jsonl for later BC training.
local RECORD_MODE = os.getenv("ISAAC_RL_RECORD") == "1"
-- ISAAC_RL_STAGE0: when set to "1", enables the Stage-0 curriculum — the
-- simplest possible learning task, isolated from the full game so the trainer
-- can prove convergence on ONE thing.
--
-- Behavior: on every MC_POST_NEW_ROOM (fires on first entry + after every
-- room transition), wait a few ticks for the room to stabilize, then:
--   1. Remove ALL existing NPCs from the room.
--   2. Spawn ONE Attack Fly (EntityType 13, variant 1) at a fixed offset
--      inside the room.
-- No door manipulation — the agent CAN leave, but the reward function
-- for Stage 0 makes kill/room_clear the dominant positive signals so a
-- learning agent should discover "kill the fly, room clears" fast.
--
-- Purpose: distinguish 'Dreamer implementation can't learn' from 'Isaac
-- environment is too complex'. If the trainer can't converge on 'kill one
-- fly in <30 ticks', the RL/WM code is broken and Isaac reward tuning is
-- premature. If it CAN converge on this, ratchet complexity by disabling
-- the flag and moving to Stage 1.
local STAGE0_MODE = os.getenv("ISAAC_RL_STAGE0") == "1"
if STAGE0_MODE then
    Isaac.DebugString("[isaac-rl-bridge] STAGE0_MODE=1 — single-fly curriculum enabled")
end
-- Ticks to wait before running stage0_setup after MC_POST_NEW_ROOM fires.
-- 0 = idle (no setup pending); >0 = countdown, run setup when it hits 1.
local stage0_setup_pending = 0

-- Stage-0 room rewrite: wipe all NPCs, spawn one Attack Fly. Wrapped in a
-- pcall so a broken entity list can't crash the mod — if it fails, the
-- current room stays whatever Isaac spawned and the agent will just have
-- a normal-difficulty room this episode. Non-fatal.
local function stage0_setup_room()
    if not STAGE0_MODE then return end
    local ok, err = pcall(function()
        local room = Game():GetRoom()
        if not room then return end
        -- 1. Remove existing NPCs. We look for anything with EntityFlag ENEMY
        --    OR EntityType.ENTITY_MONSTRO-and-family, then Remove(). We do NOT
        --    touch the player, projectiles, effects, or pickups.
        local ents = Isaac.GetRoomEntities()
        local removed = 0
        for _, e in ipairs(ents) do
            if e:IsVulnerableEnemy() or e:IsActiveEnemy(false) then
                e:Remove()
                removed = removed + 1
            end
        end
        -- 2. Spawn one Attack Fly (EntityType 13, variant 0 = normal Fly).
        --    Attack Fly is the least dangerous enemy in the game: 1 HP, no
        --    projectiles, slow. Perfect Stage-0 target.
        local center = room:GetCenterPos()
        -- Offset from center so the fly isn't ON the player.
        local spawn_pos = Vector(center.X + 120, center.Y)
        Isaac.Spawn(EntityType.ENTITY_FLY, 0, 0, spawn_pos, Vector(0, 0), nil)
        Isaac.DebugString("[isaac-rl-bridge] STAGE0: cleared " .. tostring(removed)
                          .. " enemies, spawned 1 fly at ("
                          .. tostring(spawn_pos.X) .. ", " .. tostring(spawn_pos.Y) .. ")")
    end)
    if not ok then
        Isaac.DebugString("[isaac-rl-bridge] STAGE0 setup failed: " .. tostring(err))
    end
end
-- Diagnostic: always log the raw env-var value so we can distinguish
-- 'RECORD_MODE=1 set but disabled' from 'env-var never reached the process'.
-- Fires unconditionally at mod load time.
Isaac.DebugString("[isaac-rl-bridge] boot: ISAAC_RL_RECORD="
    .. tostring(os.getenv("ISAAC_RL_RECORD"))
    .. " ISAAC_RL_PORT=" .. tostring(os.getenv("ISAAC_RL_PORT"))
    .. " record_mode_active=" .. tostring(RECORD_MODE))
if RECORD_MODE then
    Isaac.DebugString("[isaac-rl-bridge] RECORD_MODE=1 — human plays; obs+action stream written to Python")
end

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
-- Set to true on the FIRST MC_POST_UPDATE tick where player:IsDead() is true.
-- Prevents us from calling Isaac.GetRoomEntities() / FindByType() again during
-- the death animation — that combination is exactly what causes the
-- Lua5.3.3r.dll 0xc0000005 crash that closes the Isaac window.
local death_announced = false
-- Track player HP across ticks so we can fire the death handler the
-- moment HP crosses to zero (before player:IsDead() flips). Reset in
-- MC_POST_GAME_STARTED. See handle_player_death for the rationale.
local player_hp_prev = nil
-- Forward-declare so callbacks defined below (line ~344, ~568) can reference
-- it. The actual function body is defined further down where the other
-- helpers live (line ~530). Without the forward-declare, Lua would resolve
-- `handle_player_death` at callback-compile time as a global lookup, hit
-- nil at runtime, and crash the mod on the first death.
local handle_player_death

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
-- Previous tick's action state — snapshot right before apply_action rebuilds.
local last_cached_action = {
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
-- One-shot 'triggered' consumption tracker. When a fresh press (false->true
-- edge) occurs, the corresponding entry flips to false (not yet consumed).
-- The FIRST time MC_INPUT_ACTION returns true for IS_ACTION_TRIGGERED on that
-- action, the entry flips to true (consumed) and all subsequent queries for
-- that action return false until the NEXT edge.
--
-- Without this, cached_action=true persists for 2 game frames (FRAME_SKIP=2),
-- and Isaac's engine polls IS_ACTION_TRIGGERED on EACH of those frames —
-- often multiple times per frame across its subsystems — resulting in
-- multiple one-shot events per Python action. That flood is what filled the
-- user's crash log with thousands of 'Action PillCard Triggered' lines and
-- eventually made Isaacs engine collapse.
local triggered_consumed = {
    [ButtonAction.ACTION_LEFT] = true,
    [ButtonAction.ACTION_RIGHT] = true,
    [ButtonAction.ACTION_UP] = true,
    [ButtonAction.ACTION_DOWN] = true,
    [ButtonAction.ACTION_SHOOTLEFT] = true,
    [ButtonAction.ACTION_SHOOTRIGHT] = true,
    [ButtonAction.ACTION_SHOOTUP] = true,
    [ButtonAction.ACTION_SHOOTDOWN] = true,
    [ButtonAction.ACTION_BOMB] = true,
    [ButtonAction.ACTION_ITEM] = true,
    [ButtonAction.ACTION_PILLCARD] = true,
    [ButtonAction.ACTION_DROP] = true,
}

local function clear_cached_action()
    for k in pairs(cached_action) do cached_action[k] = false end
    for k in pairs(last_cached_action) do last_cached_action[k] = false end
    for k in pairs(triggered_consumed) do triggered_consumed[k] = true end
end

-- Decode MultiDiscrete([9, 5, 2, 2, 2]) into ButtonAction booleans.
local mv_table = {
    [1] = {up = true},                       [2] = {up = true, right = true},
    [3] = {right = true},                    [4] = {down = true, right = true},
    [5] = {down = true},                     [6] = {down = true, left = true},
    [7] = {left = true},                     [8] = {up = true, left = true},
}
local function apply_action(a)
    -- Snapshot current -> last so MC_INPUT_ACTION can compute press edges.
    for k, v in pairs(cached_action) do last_cached_action[k] = v end
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
    -- Action space (2026-07-02): removed use_active / drop_bomb / pill_card as
    -- separate policy outputs. They were harmful when triggered by random
    -- exploration (dropping a bomb hurts the player, random pills are often
    -- negative). Old JSON payloads may still include those keys; we accept
    -- them for backward compatibility but the policy no longer emits them.
    cached_action[ButtonAction.ACTION_ITEM]     = (not NO_ONESHOT) and (a.use_active == 1 or a.use_active == true) or false
    cached_action[ButtonAction.ACTION_BOMB]     = (not NO_ONESHOT) and (a.drop_bomb  == 1 or a.drop_bomb  == true) or false
    cached_action[ButtonAction.ACTION_PILLCARD] = (not NO_ONESHOT) and (a.pill_card  == 1 or a.pill_card  == true) or false

    -- For every action that just transitioned from unpressed to pressed,
    -- reset its triggered_consumed flag to false. The next IS_ACTION_TRIGGERED
    -- query for that action will fire ONCE and set the flag to true; every
    -- subsequent query for the same action gets false until the next edge.
    for k, v in pairs(cached_action) do
        if v and not last_cached_action[k] then
            triggered_consumed[k] = false
        end
    end
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

-- Read the human's current input state and encode as (move, shoot, use_item,
-- drop_bomb, use_pillcard) matching an *extended* MultiDiscrete([9, 5, 2, 2, 2]).
--
-- The last three factors (item / bomb / pillcard) are the action heads that
-- were removed from the RL policy on 2026-07-02 (they were harmful when
-- triggered by random exploration — a random actor would drop bombs on
-- itself and use unidentified pills). Human demos re-introduce them here
-- because a human plays them purposefully; the BC training loader picks up
-- these fields, and the RL fine-tune restores masked heads on top of the
-- BC-warm actor. Existing RL trainer code ignores unknown factors, so this
-- is backward compatible.
--
-- Factor meanings:
--   move  (9): 0=idle, 1=up, 2=up+right, 3=right, 4=down+right,
--              5=down,  6=down+left,  7=left,  8=up+left
--   shoot (5): 0=none, 1=up, 2=right, 3=down, 4=left
--   use_item     (2): 0=no,  1=yes (space bar / gamepad A)
--   drop_bomb    (2): 0=no,  1=yes (E / gamepad LB)
--   use_pillcard (2): 0=no,  1=yes (Q / gamepad RB)
--
-- Diagonal shoot combinations collapse to priority order up > right >
-- down > left, matching the RL policy's constraint.
local function read_human_action(player_idx)
    player_idx = player_idx or 0
    local up    = Input.IsActionPressed(ButtonAction.ACTION_UP, player_idx)
    local down  = Input.IsActionPressed(ButtonAction.ACTION_DOWN, player_idx)
    local left  = Input.IsActionPressed(ButtonAction.ACTION_LEFT, player_idx)
    local right = Input.IsActionPressed(ButtonAction.ACTION_RIGHT, player_idx)
    local move = 0
    if up and right then move = 2
    elseif up and left then move = 8
    elseif down and right then move = 4
    elseif down and left then move = 6
    elseif up then move = 1
    elseif right then move = 3
    elseif down then move = 5
    elseif left then move = 7
    end

    local sup    = Input.IsActionPressed(ButtonAction.ACTION_SHOOTUP, player_idx)
    local sright = Input.IsActionPressed(ButtonAction.ACTION_SHOOTRIGHT, player_idx)
    local sdown  = Input.IsActionPressed(ButtonAction.ACTION_SHOOTDOWN, player_idx)
    local sleft  = Input.IsActionPressed(ButtonAction.ACTION_SHOOTLEFT, player_idx)
    local shoot = 0
    if sup then shoot = 1
    elseif sright then shoot = 2
    elseif sdown then shoot = 3
    elseif sleft then shoot = 4
    end

    -- One-shot buttons: sampled every FRAME_SKIP=2 game frames. Human
    -- presses are typically 30-100ms (2-6 game frames) so a 15 Hz sample
    -- reliably catches them. If we start missing single-frame taps we can
    -- upgrade this to a MC_POST_UPDATE latch (OR-accumulate each game frame,
    -- clear at exchange time).
    local use_item     = Input.IsActionPressed(ButtonAction.ACTION_ITEM,     player_idx) and 1 or 0
    local drop_bomb    = Input.IsActionPressed(ButtonAction.ACTION_BOMB,     player_idx) and 1 or 0
    local use_pillcard = Input.IsActionPressed(ButtonAction.ACTION_PILLCARD, player_idx) and 1 or 0

    return {
        move = move,
        shoot = shoot,
        use_item = use_item,
        drop_bomb = drop_bomb,
        use_pillcard = use_pillcard,
    }
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

    -- RECORD_MODE: read the human's current input, attach to obs, send, don't
    -- wait for a response. The human's keys go into Isaac through the normal
    -- input path (MC_INPUT_ACTION returns nil in this mode, so nothing is
    -- injected on top of the human's actual keypresses).
    if RECORD_MODE then
        obs.human_action = read_human_action(0)
        local ok, payload = pcall(json.encode, obs)
        if not ok then
            Isaac.DebugString("[isaac-rl-bridge] json.encode failed (record): " .. tostring(payload))
            return
        end
        conn:send(payload)  -- fire-and-forget; Python is a passive listener
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
        -- If we've already announced a death this run and fired 'restart', the
        -- Python reset command is redundant — Isaac is already in mid-restart.
        -- Firing a second 'restart' on top of the queued one exits Isaacs
        -- process cleanly on some Repentance builds (window closes, no crash
        -- dump). Skip the duplicate.
        if death_announced then
            Isaac.DebugString("[isaac-rl-bridge] reset command received but restart already queued (death_announced=true), skipping duplicate")
            reset_cooldown = 60
            return
        end
        clear_cached_action()   -- release any held inputs before restart
        -- Use bare `restart` (equivalent to pressing R in-game), NOT `restart 0`.
        -- Difference: `restart` restarts the CURRENT run in place — the process
        -- stays alive and Isaac tears down + rebuilds the run state internally.
        -- `restart 0` says 'new run as character 0 (Isaac)' and in some
        -- Repentance builds triggers a deeper teardown path that ends the
        -- process entirely (window closes) instead of restarting in-place.
        -- Since we already boot as Isaac via --set-stage=1, we don't need to
        -- respecify the character on every reset — pressing R is enough.
        Isaac.DebugString("[isaac-rl-bridge] issuing 'restart' from reset command")
        Isaac.ExecuteCommand("restart")
        -- Only touch stage if we want something other than the default (1).
        -- The initial --set-stage=1 boot already put us on Basement 1, so a
        -- `stage 1` afterward is at best redundant and at worst races against
        -- the new run's init.
        if action.stage and tonumber(action.stage) and tonumber(action.stage) ~= 1 then
            pending_stage = tonumber(action.stage)
        end
        if action.seed then
            pending_seed = tostring(action.seed)
        end
        -- Long cooldown: see the death-handler comment. 2s is safe.
        reset_cooldown = 60
        return
    end
    apply_action(action)

    -- Run a small incremental GC step every exchange (≈15 Hz). Each Obs.build
    -- allocates dozens of fresh tables (enemies, projectiles, pickups, grid
    -- rows) plus a JSON string; the resulting garbage accumulates fast. Without
    -- this, Isaac's default automatic GC eventually triggers a full sweep on
    -- a large heap and stalls the game for hundreds of ms — the user-visible
    -- symptom is a periodic ≈30s hitch that can escalate to a hard crash if
    -- the frame budget overruns badly enough. `step, 100` collects a small
    -- chunk each tick; the heap stays flat and the cost is invisible per-frame.
    collectgarbage("step", 100)
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, is_continued)
    tick = 0
    reset_run_state()
    reset_cooldown = 30
    death_announced = false
    player_hp_prev = nil  -- reset HP tracking so the first tick of a new run seeds it
    clear_cached_action()
    if MINIMAL_MODE then
        Isaac.DebugString("[isaac-rl-bridge] MINIMAL_MODE: run started, socket disabled")
        return   -- no socket, no handshake, no I/O with Python
    end
    -- Apply any deferred console commands from the reset that just fired.
    if pending_seed then
        Isaac.ExecuteCommand("seed " .. pending_seed)
        pending_seed = nil
    end
    if pending_stage then
        Isaac.ExecuteCommand("stage " .. tostring(pending_stage))
        pending_stage = nil
    end
    if RECORD_MODE then
        -- RECORD_MODE: keep the existing socket across run restarts. The
        -- recorder is a single-accept server — if we close and reconnect
        -- here, the mod's Net.connect(HOST, PORT) will hang forever waiting
        -- for a new accept() that never comes, freezing Isaac. In training
        -- mode the trainer's vec_env keeps re-accepting for every episode
        -- boundary; the recorder has no such loop by design (one JSONL per
        -- record session). If we don't have a conn yet, this is the first
        -- start — fall through to the normal connect path.
        if conn then
            Isaac.DebugString("[isaac-rl-bridge] RECORD_MODE=1 — run restart; keeping existing socket to recorder")
            -- Send a lightweight 'run started' marker so the recorder
            -- can segment episodes inside a single JSONL. Non-fatal on send
            -- failure since the connection is presumed still open.
            local seed = Game():GetSeeds():GetStartSeed()
            pcall(function()
                conn:send(json.encode({
                    hello = true, schema = 2, seed = seed,
                    is_continued = is_continued,
                    run_restart = true,
                }))
            end)
            return
        end
    end
    if conn then conn:close() end
    conn = Net.connect(HOST, PORT, 0.05)   -- see net.lua for why 50 ms
    local seed = Game():GetSeeds():GetStartSeed()
    conn:send(json.encode({hello = true, schema = 2, seed = seed, is_continued = is_continued}))
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
    -- 2026-07-13: STAGE0 curriculum. Rewrite the room contents so every
    -- new room is 'one fly, no other threats'. Schedule the rewrite for
    -- a few ticks after the room callback so Isaac's own room-init pass
    -- (spawning the default enemies + pickups) has finished. Doing it
    -- inline in this callback races Isaac's engine and can leave stray
    -- half-initialized entities.
    if STAGE0_MODE then
        stage0_setup_pending = 6   -- number of MC_POST_UPDATE ticks to wait
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

-- ---------------------------------------------------------------------------
-- 2026-07-12 Track A: emit pickup_collectible event with the item's Quality.
-- ---------------------------------------------------------------------------
-- Fires when a collectible pedestal is grabbed (SubType transitions to 0).
-- We hook MC_POST_PICKUP_UPDATE and detect the transition; MC_POST_PICKUP_INIT
-- fires on spawn, not on pickup. Quality comes from Isaac.GetItemConfig()
-- which is a stable engine call — pcall wrap for safety.
--
-- Consumed by python/isaac_rl/reward.py which now scales the pickup reward
-- by quality (Q0=0.5, Q1=1.0, Q2=2.0, Q3=3.5, Q4=6.0) instead of a flat +2.
local pedestal_last_subtype = {}  -- InitSeed -> last-seen SubType

mod:AddCallback(ModCallbacks.MC_POST_PICKUP_UPDATE, function(_, pickup)
    if MINIMAL_MODE then return end
    if pickup.Variant ~= PickupVariant.PICKUP_COLLECTIBLE then return end
    local seed = pickup.InitSeed
    local prev = pedestal_last_subtype[seed]
    local curr = pickup.SubType
    -- Grab detected: subtype was non-zero (item present) and is now 0 (empty).
    if prev and prev > 0 and curr == 0 then
        local quality = -1
        local ok, cfg = pcall(function()
            return Isaac.GetItemConfig():GetCollectible(prev)
        end)
        if ok and cfg and cfg.Quality ~= nil then
            quality = cfg.Quality
        end
        run_state.pending_events[#run_state.pending_events + 1] = {
            kind = "pickup_collectible",
            item_id = prev,
            quality = quality,
        }
    end
    pedestal_last_subtype[seed] = curr
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    tick = tick + 1
    run_state.frames_since_room = run_state.frames_since_room + 1
    run_state.frames_since_hit = run_state.frames_since_hit + 1

    -- Stage-0 room rewrite: countdown started in MC_POST_NEW_ROOM. Fires
    -- ONCE per new-room event when the counter hits 1. Runs before the
    -- reset_cooldown check because we DO want it to fire during the
    -- post-transition cooldown window — that's exactly when the room is
    -- stable enough to modify.
    if stage0_setup_pending > 0 then
        stage0_setup_pending = stage0_setup_pending - 1
        if stage0_setup_pending == 0 then
            stage0_setup_room()
        end
    end

    if reset_cooldown > 0 then
        reset_cooldown = reset_cooldown - 1
        return
    end

    if MINIMAL_MODE then
        -- Only need death → restart handling. No exchange, no obs, nothing.
        local pl = Isaac.GetPlayer(0)
        if pl and pl:IsDead() then
            if not death_announced then
                death_announced = true
                clear_cached_action()
                Isaac.DebugString("[isaac-rl-bridge] MINIMAL_MODE: player died, restart")
                Isaac.ExecuteCommand("restart")
                reset_cooldown = 60
            end
        end
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

    -- Player death handling. When the player's HP reaches 0 Isaac starts a
    -- death animation and then transitions to the 'You Died' screen. During
    -- BOTH those phases the game is tearing down / has torn down its entity
    -- lists, and any call to Isaac.GetRoomEntities() / FindByType() / player
    -- methods can dereference freed C++ pointers — which crashes the Isaac
    -- process (Lua5.3.3r.dll 0xc0000005, verified via Event Viewer). The
    -- window closing on death was this crash, not our `restart` command.
    --
    -- Fix: on the FIRST tick of death, send Python a minimal terminal-obs
    -- (no entity iteration), swallow whatever it sends back, then fire
    -- `restart` immediately. All subsequent ticks skip exchange entirely
    -- until MC_POST_GAME_STARTED clears the death_announced flag.
    local player = Isaac.GetPlayer(0)
    -- Broadened trigger: catch dying BEFORE Isaac's IsDead flag flips.
    -- Isaac defers the IsDead flag by 1-2 ticks after the fatal damage.
    -- On some deaths MC_POST_UPDATE stops firing during that window,
    -- causing this callback to NEVER see IsDead()=true and the whole
    -- death path to fall through to the render watchdog. Also check
    -- total HP <= 0 so we fire the moment damage is applied — catches
    -- the death 1 tick earlier while the update callbacks are still
    -- firing. Wrap in pcall because during the death animation any
    -- player method can dereference freed C++ pointers (0xc0000005).
    if player then
        local hp_ok, total_hp = pcall(function()
            return (player:GetHearts() or 0) + (player:GetSoulHearts() or 0) + (player:GetBlackHearts() or 0)
        end)
        -- Fire on either IsDead() (Isaac's official flag — may be delayed
        -- by 1-2 ticks), or on the HP transition >0 → ≤0 (catches the
        -- damage-application tick before IsDead flips). The transition
        -- check is important: we can't just use `total_hp <= 0` because
        -- some characters (e.g. The Lost) live at 0 HP. Only firing on
        -- the CROSS-TO-ZERO transition avoids false positives.
        local hp_transitioned_to_zero =
            hp_ok and player_hp_prev ~= nil
            and player_hp_prev > 0 and total_hp <= 0
        local is_dying = player:IsDead() or hp_transitioned_to_zero
        if hp_ok then
            player_hp_prev = total_hp
        end
        if is_dying then
            handle_player_death("MC_POST_UPDATE")
            return
        end
    end

    if (tick % FRAME_SKIP) == 0 and conn then
        -- Extra safety: room must have ticked at least once so its entity
        -- lists are populated, and the player must exist. Bumped from 2 to 10
        -- so post-teardown entity churn has fully settled by the time we
        -- iterate. Isaac's engine has been observed to crash at 0xc0000005
        -- inside isaac-ng.exe (not Lua) when reading half-initialized entity
        -- state right after a restart; this is a defense-in-depth guard on
        -- top of reset_cooldown.
        if not player then return end
        if room:GetFrameCount() < 10 then return end
        exchange()
    end
end)

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
    if MINIMAL_MODE then return nil end   -- no input injection at all
    if RECORD_MODE then return nil end    -- human plays; don't inject anything on top of real input
    -- Only inject cached inputs into a valid PLAYER entity. Isaac's input
    -- system also queries with entity=nil for menu / transition / You-Died-
    -- screen contexts. Feeding those stale gameplay inputs (SHOOTRIGHT, ITEM,
    -- etc.) from the last live action into menu code has been observed to
    -- push isaac-ng.exe into an unrecoverable state (0xc0000005 access
    -- violation deep in the engine, offset 0x3a93b5). Skip all of those.
    if not entity then return nil end
    if entity:ToPlayer() == nil then return nil end
    local pressed = cached_action[action]
    if pressed == nil then return nil end
    if hook == InputHook.IS_ACTION_TRIGGERED then
        -- Return TRUE exactly ONCE per fresh press. Uses a consumption latch
        -- so multiple queries within the same press window (either across
        -- the 2 game frames the same cached_action persists, or multiple
        -- calls per frame from different Isaac subsystems) all return false
        -- after the first successful read. Prevents Isaac from processing
        -- the same pill/bomb/item use dozens of times per Python action.
        if not pressed then return false end
        if triggered_consumed[action] then return false end
        triggered_consumed[action] = true
        return true
    elseif hook == InputHook.IS_ACTION_PRESSED then
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
-- Menu auto-start timeout in render frames. Reduced from 600 (10s) to 240 (4s)
-- so if --set-stage=N somehow doesn't take effect and Isaac lands on the main
-- menu, we auto-fire 'restart 0' quickly instead of leaving the user staring
-- at a menu. 240 frames ≈ 4s at 60 render Hz — well past intro cinematics
-- (which don't seem to fire MC_POST_RENDER anyway due to Theora playback).
local AUTO_START_AFTER_FRAMES = 240

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

-- Player death handling. When the player's HP reaches 0 Isaac starts a
-- death animation and then transitions to the 'You Died' screen. During
-- BOTH those phases the game is tearing down / has torn down its entity
-- lists, and any call to Isaac.GetRoomEntities() / FindByType() / player
-- methods can dereference freed C++ pointers — which crashes the Isaac
-- process (Lua5.3.3r.dll 0xc0000005, verified via Event Viewer). The
-- window closing on death was this crash, not our `restart` command.
--
-- Fix: on the FIRST tick of death, send Python a minimal terminal-obs
-- (no entity iteration), swallow whatever it sends back, then fire
-- `restart` immediately. All subsequent ticks skip exchange entirely
-- until MC_POST_GAME_STARTED clears the death_announced flag.
--
-- 2026-07-08 REV: extracted into a helper so both MC_POST_UPDATE (fast
-- path) and MC_POST_RENDER (fallback — fires when MC_POST_UPDATE stops
-- during the death animation) call the same code. The old MC_POST_RENDER
-- watchdog only fired `restart` without notifying Python, which caused
-- 100% of episodes to be classified as mod_socket_error and skipped the
-- shaper's death event (verified 2026-07-08 run: 85k steps, 222
-- episodes, zero `reward/death` firings). Also broadened the trigger
-- condition so we fire on HP<=0 even if player:IsDead() hasn't flipped
-- yet, and captured the actual send_blocking return value (previously
-- the pcall wrapper swallowed it, so we had no signal on failure).
-- Body for the forward-declared `handle_player_death`. Assigning to the
-- existing local (rather than `local function` here, which would create a
-- NEW local scope and shadow the forward-declared one) so the callbacks
-- above see the same function reference.
handle_player_death = function(source)
    if death_announced then return end
    death_announced = true
    clear_cached_action()
    run_state.pending_events[#run_state.pending_events + 1] = { kind = "death" }
    Isaac.DebugString("[isaac-rl-bridge] handle_player_death firing (source=" .. source .. ")")
    if conn then
        local minimal = {
            schema = 2,
            tick = tick,
            player = { is_dead = true, hp_red = 0 },
            events = { { kind = "death" } },
        }
        -- pcall returns (ok, retvals...). Capture BOTH so we know whether
        -- send_blocking actually delivered the frame or timed out. Previous
        -- code was `local ok = pcall(function() return send_blocking(...) end)`
        -- which silently threw away the delivered=true/false return.
        local pcall_ok, delivered = pcall(function()
            return conn:send_blocking(json.encode(minimal), 2.0)
        end)
        if pcall_ok and delivered then
            Isaac.DebugString("[isaac-rl-bridge] terminal-obs DELIVERED (source=" .. source .. ")")
            -- Consume Python's reset command so the socket buffer doesn't
            -- hold a stale message across MC_POST_GAME_STARTED.
            pcall(function() conn:recv() end)
        else
            Isaac.DebugString("[isaac-rl-bridge] terminal-obs FAILED source=" .. source .. " pcall_ok=" .. tostring(pcall_ok) .. " delivered=" .. tostring(delivered))
        end
    end
    Isaac.DebugString("[isaac-rl-bridge] issuing 'restart' from " .. source)
    -- RECORD_MODE: DO NOT auto-restart the run. The human is playing and
    -- Isaac's own game-over screen will show; they pick continue/restart/menu
    -- themselves. Firing Isaac.ExecuteCommand('restart') here on top of
    -- Isaac's built-in death flow has been observed to fully close the
    -- process on some Repentance builds (see death_announced comment near
    -- top of file), which kills the recording session mid-play. In record
    -- mode we just emit the death event to Python (already done above) and
    -- let the human handle the restart via Isaac's UI or the R key.
    if RECORD_MODE then
        Isaac.DebugString("[isaac-rl-bridge] RECORD_MODE=1 — skipping ExecuteCommand('restart'); human handles restart manually")
        reset_cooldown = 60
        return
    end
    Isaac.ExecuteCommand("restart")
    reset_cooldown = 60
end

-- MC_POST_UPDATE fires only during an active, unpaused run. The instant we see
-- one, we know the run is live and disable the fallback permanently.
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    if not auto_start_fired then
        auto_start_fired = true
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    -- RECORD_MODE: the whole render-driven auto-start / auto-restart machinery
    -- is training-mode-only. In record mode the human picks their character
    -- and mode from the main menu, and handles death themselves via Isaac's
    -- built-in game-over screen or R key. Any Isaac.ExecuteCommand('restart')
    -- fired here would either (a) skip the user's character choice, or (b)
    -- close the Isaac process (see comments above). Short-circuit immediately.
    if RECORD_MODE then return end

    -- Menu auto-start branch (only runs before any MC_POST_UPDATE ever fired).
    if not auto_start_fired then
        if Game():GetFrameCount() > 0 then
            auto_start_fired = true
        else
            render_frames_on_menu = render_frames_on_menu + 1
            if render_frames_on_menu == AUTO_START_AFTER_FRAMES then
                auto_start_fired = true
                Isaac.DebugString("[isaac-rl-bridge] auto-start fallback firing (main menu detected after intro)")
                -- On the menu we DO need `restart 0` — there's no active run
                -- to soft-restart, so we're explicitly saying 'new run as
                -- Isaac'. This is the one place bare `restart` won't work.
                Isaac.ExecuteCommand("restart 0")
            end
        end
        return
    end

    -- Death auto-restart branch (only relevant after a run has been active).
    -- CRITICAL: never fire here if death_announced is already true. That flag
    -- means MC_POST_UPDATE's death handler already ran Isaac.ExecuteCommand
    -- ('restart') for this death. Firing a SECOND restart on top of the queued
    -- one has been observed to trigger Isaac's full-shutdown path in some
    -- Repentance builds — the game exits cleanly (no Application Error, no
    -- crash dump) but the window closes and Python sees a socket RST. Only
    -- fire from here if the update-side handler somehow missed the death
    -- (MC_POST_UPDATE stopped firing before it could detect IsDead()=true).
    local ok, player = pcall(Isaac.GetPlayer, 0)
    if not ok or not player then
        death_render_frames = 0
        return
    end
    if player:IsDead() and not death_announced then
        -- 2026-07-08 REV: previously waited DEATH_AUTO_RESTART_FRAMES
        -- (~2s) of continuous player-dead render frames before firing
        -- `restart`, and NEVER notified Python of the death. Result:
        -- 100% of episodes appeared to Python as socket errors instead
        -- of proper deaths, and the shaper never saw a `death` event
        -- (verified 2026-07-08 run: zero `reward/death` firings across
        -- 222 episodes). Now: fire the death handler on the FIRST
        -- render frame where player is dead, going through the same
        -- code path as MC_POST_UPDATE (delivers terminal obs to Python
        -- so the shaper can apply r_death + hp deltas properly).
        handle_player_death("MC_POST_RENDER")
    else
        death_render_frames = 0
    end
end)

Isaac.DebugString("[isaac-rl-bridge] mod loaded (port " .. tostring(PORT) .. ", frame skip " .. FRAME_SKIP .. ")")
