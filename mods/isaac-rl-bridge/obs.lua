-- obs.lua — build the observation table sent to Python each control tick.
-- M1 scope: minimal player + global fields. Entity/projectile/grid encoding
-- lands after the socket loop is verified end-to-end.

local Obs = {}

-- Return a compact table. Keep numeric fields as numbers (JSON will encode them).
function Obs.build(tick)
    local game = Game()
    local level = game:GetLevel()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)
    local pos = player.Position
    local vel = player.Velocity

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
        },
        global = {
            stage = level:GetStage(),
            stage_type = level:GetStageType(),
            room_index = level:GetCurrentRoomIndex(),
            room_type = room:GetType(),
            is_clear = room:IsClear(),
            curses = level:GetCurses(),
        },
        -- Placeholders for downstream milestones — keep the schema forward-compatible.
        enemies = {},
        projectiles = {},
        pickups = {},
    }
end

return Obs
