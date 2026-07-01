-- main.lua — Isaac RL bridge entry point.
--
-- Registers callbacks that:
--   1. Connect to the Python trainer on game start.
--   2. Each MC_POST_UPDATE (30 Hz), build an obs, send it, wait for an action.
--   3. Inject the cached action via MC_INPUT_ACTION.
--
-- Control rate is 15 Hz: we send/receive every other game frame and reuse the
-- cached action across the skipped frame (plan §1.3). Bump FRAME_SKIP to 1 to
-- drop to 30 Hz control if diagnosing lag.

local Net = require("net")
local Obs = require("obs")
local json = require("json")

local FRAME_SKIP = 2
local HOST = "127.0.0.1"
local PORT = tonumber(os.getenv("ISAAC_RL_PORT") or "9500")

local mod = RegisterMod("isaac-rl-bridge", 1)
local conn = nil
local tick = 0

-- Cached action written by Python, read by MC_INPUT_ACTION.
-- Keys match ButtonAction enum members. Values are booleans.
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

-- Decode a policy MultiDiscrete([9, 5, 2, 2, 2]) action into ButtonAction booleans.
-- move  ∈ 0..8: 0=none, 1..8 clockwise from up (up, up-right, right, down-right, down, down-left, left, up-left)
-- shoot ∈ 0..4: 0=none, 1..4 cardinal (up, right, down, left)
local function apply_action(a)
    for k in pairs(cached_action) do cached_action[k] = false end
    local move = a.move or 0
    local shoot = a.shoot or 0

    -- Movement direction table indexed by move value (1..8).
    local mv = {
        [1] = {up = true},
        [2] = {up = true, right = true},
        [3] = {right = true},
        [4] = {down = true, right = true},
        [5] = {down = true},
        [6] = {down = true, left = true},
        [7] = {left = true},
        [8] = {up = true, left = true},
    }
    local m = mv[move]
    if m then
        cached_action[ButtonAction.ACTION_UP]    = m.up    == true
        cached_action[ButtonAction.ACTION_DOWN]  = m.down  == true
        cached_action[ButtonAction.ACTION_LEFT]  = m.left  == true
        cached_action[ButtonAction.ACTION_RIGHT] = m.right == true
    end

    if     shoot == 1 then cached_action[ButtonAction.ACTION_SHOOTUP]    = true
    elseif shoot == 2 then cached_action[ButtonAction.ACTION_SHOOTRIGHT] = true
    elseif shoot == 3 then cached_action[ButtonAction.ACTION_SHOOTDOWN]  = true
    elseif shoot == 4 then cached_action[ButtonAction.ACTION_SHOOTLEFT]  = true
    end

    cached_action[ButtonAction.ACTION_ITEM]     = a.use_active == 1 or a.use_active == true
    cached_action[ButtonAction.ACTION_BOMB]     = a.drop_bomb  == 1 or a.drop_bomb  == true
    cached_action[ButtonAction.ACTION_PILLCARD] = a.pill_card  == 1 or a.pill_card  == true
end

-- Poll-and-block: send obs, wait for action. On timeout the previous action is reused.
local function exchange()
    local obs = Obs.build(tick)
    local ok, payload = pcall(json.encode, obs)
    if not ok then
        Isaac.DebugString("[isaac-rl-bridge] json.encode failed: " .. tostring(payload))
        return
    end
    conn:send(payload)

    local msg = conn:recv()
    if not msg then return end  -- timeout — keep previous cached_action
    local ok2, action = pcall(json.decode, msg)
    if not ok2 or type(action) ~= "table" then
        Isaac.DebugString("[isaac-rl-bridge] json.decode failed: " .. tostring(action))
        return
    end
    if action.reset then
        -- Trainer requested episode reset. Restart the run with an optional seed.
        if action.seed then
            Isaac.ExecuteCommand("seed " .. tostring(action.seed))
        end
        Isaac.ExecuteCommand("restart 0")  -- 0 = Isaac
        return
    end
    apply_action(action)
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, is_continued)
    tick = 0
    if conn then conn:close() end
    conn = Net.connect(HOST, PORT, 0.25)
    -- Handshake: schema version + start seed, so the trainer can log it.
    local seed = Game():GetSeeds():GetStartSeed()
    conn:send(json.encode({hello = true, schema = 1, seed = seed, is_continued = is_continued}))
    Isaac.DebugString("[isaac-rl-bridge] connected to trainer on port " .. tostring(PORT))
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    tick = tick + 1
    if (tick % FRAME_SKIP) == 0 and conn then
        exchange()
    end
end)

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
    -- Only override player inputs. Menu/pause pass through (nil return).
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

Isaac.DebugString("[isaac-rl-bridge] mod loaded (port " .. tostring(PORT) .. ", frame skip " .. FRAME_SKIP .. ")")
