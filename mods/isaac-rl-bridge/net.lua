-- net.lua — LuaSocket wrapper with 4-byte big-endian length prefix framing.
-- Requires Isaac launched with --luadebug (unlocks require/io/os/socket).

local Net = {}
Net.__index = Net

local ok_socket, socket = pcall(require, "socket")
if not ok_socket then
    error("[isaac-rl-bridge] require('socket') failed. Launch Isaac with --luadebug and place LuaSocket alongside Isaac.exe.")
end

-- Pack a 32-bit unsigned integer into a 4-byte big-endian string.
local function pack_u32_be(n)
    local b1 = math.floor(n / 16777216) % 256
    local b2 = math.floor(n / 65536) % 256
    local b3 = math.floor(n / 256) % 256
    local b4 = n % 256
    return string.char(b1, b2, b3, b4)
end

local function unpack_u32_be(s)
    local b1, b2, b3, b4 = s:byte(1, 4)
    return ((b1 * 256 + b2) * 256 + b3) * 256 + b4
end

function Net.connect(host, port, timeout_s)
    local self = setmetatable({}, Net)
    self.host = host or "127.0.0.1"
    self.port = port or 9500
    -- Short recv timeout is CRITICAL for smooth gameplay under RL training.
    -- The mod's exchange() blocks inside MC_POST_UPDATE waiting for Python's
    -- next action. Isaac's game loop is frozen while we're in that recv.
    -- If Python is mid-PPO-update it won't respond for several seconds
    -- — with a 250ms timeout we'd stack up 10+ back-to-back 250ms blocks
    -- per PPO update, giving a visible ~30s stutter every rollout. A short
    -- 50ms timeout keeps Isaac responsive; on timeout we just re-use the
    -- previous action for one tick, which is invisible to training.
    -- 2026-07-15: raised 50ms -> 120ms. The room_tensor obs is a larger payload
    -- than the old flat obs; under a PPO-update stall the OS send buffer can
    -- need >50ms to drain, causing partial sends -> socket close -> Python
    -- 'socket closed while reading frame'. 120ms gives the write room to
    -- complete while still keeping Isaac responsive (on timeout we reuse the
    -- previous action for one tick, invisible to training).
    self.timeout_s = timeout_s or 0.12
    self.sock = nil
    self:_reconnect()
    return self
end

function Net:_reconnect()
    if self.sock then
        pcall(function() self.sock:close() end)
        self.sock = nil
    end
    local s = socket.tcp()
    s:settimeout(self.timeout_s)
    -- Reconnect budget must OUTLAST a Python PPO gradient update. Every episode
    -- reset, Python closes the client socket and re-listens (env.py reset() ->
    -- _accept), so the mod reconnects each episode. Normally that succeeds in
    -- <100ms. But when a reset lands WHILE Python is mid-PPO-update, Python
    -- hasn't reached accept() yet — the connect() refuses/times out until the
    -- update finishes (a few seconds at this batch size). The old 5*0.25s=1.25s
    -- budget expired inside that window and error()'d at this line, producing
    -- the 'could not connect to trainer — timeout' bursts in the Isaac log
    -- (10-13 in a row, then recovery once accept() was reached).
    --
    -- Fix: poll FAST (short connect timeout + 60ms sleep) over a LONG total
    -- budget (~8s). Key point: a successful connect returns the INSTANT Python
    -- reaches accept(), so the game only stalls for as long as Python is
    -- genuinely busy — NOT the full budget. Fast polling keeps the normal-case
    -- reconnect invisible (<100ms) while surviving a full update window without
    -- crashing the PostUpdate callback. This is why the old 40*250ms froze the
    -- game (250ms sleep = coarse polling, up to 250ms wasted past accept-ready);
    -- 60ms polling drains almost immediately once the listener is back.
    s:settimeout(0.06)
    local last_err
    local deadline = 8.0        -- total seconds to keep trying before giving up
    local elapsed = 0.0
    while elapsed < deadline do
        local ok, err = s:connect(self.host, self.port)
        if ok then
            s:settimeout(self.timeout_s)   -- restore normal send/recv timeout
            s:setoption("tcp-nodelay", true)
            self.sock = s
            return true
        end
        last_err = err
        socket.sleep(0.06)
        elapsed = elapsed + 0.06
        -- A fresh TCP socket may be unusable after a failed connect() on some
        -- stacks; recreate it each attempt to be safe.
        pcall(function() s:close() end)
        s = socket.tcp()
        s:settimeout(0.06)
    end
    error("[isaac-rl-bridge] could not connect to trainer at " ..
        self.host .. ":" .. tostring(self.port) .. " — " .. tostring(last_err) ..
        " (gave up after " .. tostring(deadline) .. "s — trainer down, not just mid-update?)")
end

-- Send a length-prefixed frame. Reconnects on peer-death; drops-on-timeout.
function Net:send(payload)
    if not self.sock then self:_reconnect() end
    local frame = pack_u32_be(#payload) .. payload
    local sent, err, last = self.sock:send(frame)
    if sent then return end
    -- Check for PARTIAL write. LuaSocket returns (nil, "timeout", bytes_sent).
    -- If ANY bytes made it onto the wire before the timeout, framing is now
    -- corrupted: Python will read those bytes as the start of a new frame,
    -- get a bogus length prefix, and either hang or drop the connection.
    -- Closing the socket immediately is the only safe recovery — the next
    -- send() call will reconnect fresh with clean framing.
    if last and last > 0 and last < #frame then
        Isaac.DebugString("[isaac-rl-bridge] partial send (" .. tostring(last) .. "/" .. tostring(#frame) .. ") — closing socket to preserve framing")
        if self.sock then
            pcall(function() self.sock:close() end)
            self.sock = nil
        end
        return
    end
    -- Pure timeout (zero bytes written). Peer temporarily not reading. Buffer
    -- was already full at the OS level. Drop this frame; next tick will build
    -- fresh obs and retry.
    if err == "timeout" then
        return
    end
    -- Real error (peer closed, network reset). Try one reconnect + resend.
    self:_reconnect()
    self.sock:send(frame)
end

-- Send with a caller-specified timeout. Used for the terminal-obs send on
-- player death: the default 50ms timeout is too short when Isaac is
-- backgrounded and Windows throttles the process to ~3Hz (game window
-- unfocused). Under throttling the socket write can take hundreds of ms to
-- complete; if it doesn't, the mod drops the terminal frame and Python's
-- recv_frame() eventually sees ConnectionError, applies a bare -1 crash
-- penalty, and never gets the RewardShaper's death event. Result:
-- 100% of episodes end with no learning signal (verified on the
-- 2026-07-07 9.5h run at 100% crash rate).
--
-- Restores the previous timeout before returning so the regular exchange()
-- loop keeps its short 50ms responsiveness on the next call.
function Net:send_blocking(payload, timeout_s)
    if not self.sock then self:_reconnect() end
    local prev = self.timeout_s
    self.sock:settimeout(timeout_s)
    local frame = pack_u32_be(#payload) .. payload
    local sent, err, last = self.sock:send(frame)
    -- Always restore the short timeout for subsequent operations.
    self.sock:settimeout(prev)
    if sent then return true end
    -- Same partial-write / timeout handling as Net:send. The main
    -- difference is that we return a boolean so the caller (death handler)
    -- can log a warning when the terminal obs was dropped.
    if last and last > 0 and last < #frame then
        Isaac.DebugString("[isaac-rl-bridge] send_blocking partial send (" .. tostring(last) .. "/" .. tostring(#frame) .. ") — closing socket")
        if self.sock then
            pcall(function() self.sock:close() end)
            self.sock = nil
        end
        return false
    end
    if err == "timeout" then
        Isaac.DebugString("[isaac-rl-bridge] send_blocking TIMED OUT after " .. tostring(timeout_s) .. "s — terminal frame dropped (window backgrounded? Isaac throttled?)")
        return false
    end
    return false
end

-- Receive a length-prefixed frame. Returns the payload string or nil on timeout.
function Net:recv()
    if not self.sock then self:_reconnect() end
    local header, err = self.sock:receive(4)
    if not header then
        if err == "timeout" then return nil end
        self:_reconnect()
        header = self.sock:receive(4)
        if not header then return nil end
    end
    local n = unpack_u32_be(header)
    if n == 0 then return "" end
    local body, err2 = self.sock:receive(n)
    if not body then
        if err2 == "timeout" then return nil end
        error("[isaac-rl-bridge] recv body failed: " .. tostring(err2))
    end
    return body
end

function Net:close()
    if self.sock then
        pcall(function() self.sock:close() end)
        self.sock = nil
    end
end

return Net
