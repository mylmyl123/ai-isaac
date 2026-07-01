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
    self.timeout_s = timeout_s or 0.05
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
    -- Retry a few times so the trainer can start after Isaac.
    local last_err
    for _ = 1, 40 do
        local ok, err = s:connect(self.host, self.port)
        if ok then
            s:setoption("tcp-nodelay", true)
            self.sock = s
            return true
        end
        last_err = err
        socket.sleep(0.25)
    end
    error("[isaac-rl-bridge] could not connect to trainer at " ..
        self.host .. ":" .. tostring(self.port) .. " — " .. tostring(last_err))
end

-- Send a length-prefixed frame. Reconnects on transient failure.
function Net:send(payload)
    if not self.sock then self:_reconnect() end
    local frame = pack_u32_be(#payload) .. payload
    local sent, err, last = self.sock:send(frame)
    if not sent then
        -- Retry once after reconnect.
        self:_reconnect()
        self.sock:send(frame)
    end
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
