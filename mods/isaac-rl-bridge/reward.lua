-- reward.lua — event stream module.
--
-- Reward shaping lives on the Python side (see python/isaac_rl/reward.py), but we
-- catch damage events *here* because they arrive through MC_ENTITY_TAKE_DMG which
-- is a callback, not a per-frame poll. We buffer the events until the next
-- MC_POST_UPDATE tick, then obs.lua drains them into the outgoing frame.
--
-- Event shape:
--   { kind = "damage_to_npc",     dmg = N, npc_type = T, npc_hp_after = HP, killed = bool }
--   { kind = "damage_to_player",  dmg = N, damage_flags = F }
--   ... (extend freely; Python-side reward.py knows how to decode)

local R = {}
R.buffer = {}
R.stats = {
    total_damage_dealt = 0,
    total_damage_taken = 0,
    kills = 0,
}
-- Per-tick set of {InitSeed = true} for entities we've already counted as
-- killed. Cleared every MC_POST_UPDATE. Prevents overkill double-counting
-- when multiple tears hit the same enemy on the same tick.
R._killed_this_tick = {}

function R.push(evt)
    R.buffer[#R.buffer + 1] = evt
end

function R.drain()
    local out = R.buffer
    R.buffer = {}
    -- Clear the per-tick killed-seed set now that the caller (obs.lua) has
    -- consumed events. Next tick starts with a fresh dedup set.
    R._killed_this_tick = {}
    return out
end

function R.reset_run()
    R.buffer = {}
    R._killed_this_tick = {}
    R.stats.total_damage_dealt = 0
    R.stats.total_damage_taken = 0
    R.stats.kills = 0
end

function R.attach(mod)
    mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, flags, source, countdown)
        if not entity then return end
        local player = entity:ToPlayer()
        if player then
            R.stats.total_damage_taken = R.stats.total_damage_taken + amount
            R.push({
                kind = "damage_to_player",
                dmg = amount,
                damage_flags = flags,
            })
        else
            local npc = entity:ToNPC()
            if npc and npc:IsVulnerableEnemy() then
                R.stats.total_damage_dealt = R.stats.total_damage_dealt + amount
                local hp_after = math.max(0, entity.HitPoints - amount)
                local killed = hp_after <= 0
                -- Overkill dedup: if this entity already counted as killed
                -- on this tick, don't count again. Fixes bug where two tears
                -- landing on the same tick would both register 'killed=true'
                -- and inflate kills_mean by 1-2×.
                local seed = tostring(entity.InitSeed or 0)
                if killed then
                    if R._killed_this_tick[seed] then
                        killed = false  -- already counted this tick
                    else
                        R._killed_this_tick[seed] = true
                        R.stats.kills = R.stats.kills + 1
                        -- Smoke-gate instrumentation (2026-07-14): log the
                        -- entity type of every counted kill so an automated
                        -- identity check can assert kills come from the
                        -- intended enemy (Horf=12 on Stage 0) and not some
                        -- other NPC. Only fires on real (deduped) kills, so
                        -- volume is bounded and it never spams per-tear.
                        Isaac.DebugString("[isaac-rl-bridge] kill npc_type="
                            .. tostring(entity.Type) .. " variant=" .. tostring(entity.Variant))
                    end
                end
                R.push({
                    kind = "damage_to_npc",
                    dmg = amount,
                    npc_type = entity.Type,
                    npc_variant = entity.Variant,
                    npc_hp_after = hp_after,
                    npc_max_hp = entity.MaxHitPoints,
                    killed = killed,
                    is_boss = npc:IsBoss(),
                })
            end
        end
        return nil  -- do not modify damage
    end)

    -- Track pedestal grabs / pickup collection. Simpler than parsing entity list.
    mod:AddCallback(ModCallbacks.MC_POST_PICKUP_UPDATE, function(_, pickup)
        -- Subtype 0 on a collectible pedestal means "empty" — got picked up this frame.
        -- We only fire once per pedestal instance.
        if pickup.Variant == PickupVariant.PICKUP_COLLECTIBLE then
            local key = "coll_" .. tostring(pickup.InitSeed)
            if pickup.SubType == 0 and not R._collected[key] then
                R._collected[key] = true
                R.push({ kind = "pickup_collectible" })
            end
        end
    end)

    -- 2026-07-09: Track active-item usage (space bar). MC_USE_ITEM fires when
    -- the player presses SPACE with an active item that has enough charge.
    -- The Python-side shaper rewards r_use_item on every fire, plus an extra
    -- chain reward if `was_charged` is true (item was at full charge, i.e.
    -- the player waited to use it optimally rather than mashing space).
    --
    -- The signature is (item_id, rng, player, use_flags, active_slot,
    -- custom_var_data). Some of those may vary by Repentance version; we
    -- use pcall to defensively fall through if the callback doesn't fire.
    mod:AddCallback(ModCallbacks.MC_USE_ITEM, function(_, item_id, _rng, player, use_flags, active_slot)
        -- Determine if the item was fully charged. Best proxy: on-use we
        -- can check the item's max charge vs the pre-use charge. But the
        -- pre-use charge is already consumed by the time this fires — so
        -- we approximate 'was_charged' by checking use_flags for the
        -- 'USE_OWNED' flag which indicates a proper item use (not a
        -- passive trigger). Coarse but useful.
        local ok, was_charged = pcall(function()
            -- UseFlag.USE_OWNED = 1 (bit flag). If set, this was a
            -- player-initiated space-press with a fully-charged item.
            if type(use_flags) == "number" and (use_flags & 1) ~= 0 then
                return true
            end
            return false
        end)
        R.push({
            kind = "use_item",
            item_id = tonumber(item_id) or 0,
            was_charged = (ok and was_charged) or false,
        })
        return nil  -- do not modify the item's effect
    end)
end

R._collected = {}

function R.reset_room()
    R._collected = {}
end

return R
