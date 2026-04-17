function sysCall_init()

    filteredPos = nil
    filteredQuat = nil
    alpha = 0.2

    hand = sim.getObject('..')
    -- Pointer Finger (AKA index)
    pointer_prox = sim.getObject('/Allegro_Hand/pointer_base/link_0_0_respondable/pointer_proximal')
    pointer_mid = sim.getObject('/Allegro_Hand/pointer_base/link_0_0_respondable/pointer_proximal/link_1_0_respondable/pointer_middle')
    pointer_dist = sim.getObject('/Allegro_Hand/pointer_base/link_0_0_respondable/pointer_proximal/link_1_0_respondable/pointer_middle/link_2_0_respondable/pointer_distal')
    
    -- Middle Finger
    middle_prox = sim.getObject('/Allegro_Hand/middle_base/link_4_0_respondable/middle_proximal')
    middle_mid = sim.getObject('/Allegro_Hand/middle_base/link_4_0_respondable/middle_proximal/link_5_0_respondable/middle_middle')
    middle_dist = sim.getObject('/Allegro_Hand/middle_base/link_4_0_respondable/middle_proximal/link_5_0_respondable/middle_middle/link_6_0_respondable/middle_distal')
    
    -- Ring Finger
    ring_prox = sim.getObject('/Allegro_Hand/ring_base/link_8_0_respondable/ring_proximal')
    ring_mid = sim.getObject('/Allegro_Hand/ring_base/link_8_0_respondable/ring_proximal/link_9_0_respondable/ring_middle')
    ring_dist = sim.getObject('/Allegro_Hand/ring_base/link_8_0_respondable/ring_proximal/link_9_0_respondable/ring_middle/link_10_0_respondable/ring_distal')
    
    -- Thumb
    thumb_meta = sim.getObject('/Allegro_Hand/thumb_metacarpal')
    thumb_prox = sim.getObject('/Allegro_Hand/thumb_metacarpal/link_12_0_respondable/thumb_base/link_13_0_respondable/thumb_proximal')
    thumb_dist = sim.getObject('/Allegro_Hand/thumb_metacarpal/link_12_0_respondable/thumb_base/link_13_0_respondable/thumb_proximal/link_14_0_respondable/thumb_distal')
    thumb_base = sim.getObject('/Allegro_Hand/thumb_metacarpal/link_12_0_respondable/thumb_base')
    
    
    frames = {}
    frameIndex = 1

    local file = io.open("C:/Users/evanl/Desktop/HandRecordings/hand_recording_20260402_220501.json","r")

    for line in file:lines() do
        table.insert(frames, line)
    end

    file:close()

    print("Frames loaded:", #frames)

end

function parseWrist(line)

    local px,py,pz =
        line:match('"pos":%[(.-),(.-),(.-)%]')

    local qx,qy,qz,qw =
        line:match('"rot":%[(.-),(.-),(.-),(.-)%]')

    return {tonumber(px),tonumber(py),tonumber(pz)},
           {tonumber(qx),tonumber(qy),tonumber(qz),tonumber(qw)}

end

function driveFinger(line, prefix, joint_prox, joint_mid, joint_dist)

    local p = {line:match('"'..prefix..'Proximal":%[(.-),(.-),(.-),(.-)%]')}
    local m = {line:match('"'..prefix..'Intermediate":%[(.-),(.-),(.-),(.-)%]')}
    local d = {line:match('"'..prefix..'Distal":%[(.-),(.-),(.-),(.-)%]')}

    if p[1] then
        local qx,qy,qz,qw = tonumber(p[1]),tonumber(p[2]),tonumber(p[3]),tonumber(p[4])
        sim.setJointPosition(joint_prox, 2*math.atan2(qx,qw))
    end

    if m[1] then
        local qx,qy,qz,qw = tonumber(m[1]),tonumber(m[2]),tonumber(m[3]),tonumber(m[4])
        sim.setJointPosition(joint_mid, 2*math.atan2(qx,qw))
    end

    if d[1] then
        local qx,qy,qz,qw = tonumber(d[1]),tonumber(d[2]),tonumber(d[3]),tonumber(d[4])
        sim.setJointPosition(joint_dist, 2*math.atan2(qx,qw))
    end

end

function sysCall_actuation()

    if frameIndex > #frames then return end

    local line = frames[frameIndex]

    local pos,rot = parseWrist(line)

    if pos and rot then

        -- Unity coordinates
        local ux,uy,uz = pos[1],pos[2],pos[3]

        -- Convert Unity (X,Y,Z) ? CoppeliaSim (X,Y,Z)
        local simPos = {uz, ux, uy}
        if filteredPos == nil then
            filteredPos = simPos
        else
            filteredPos = {
                filteredPos[1] + alpha*(simPos[1]-filteredPos[1]),
                filteredPos[2] + alpha*(simPos[2]-filteredPos[2]),
                filteredPos[3] + alpha*(simPos[3]-filteredPos[3])
            }
        end

        sim.setObjectPosition(hand,-1,filteredPos)
        local qx,qy,qz,qw = rot[1],rot[2],rot[3],rot[4]

        local simQuat = {qz, qx, qy, qw}

        if filteredQuat == nil then
            filteredQuat = simQuat
        else
            filteredQuat = {
                filteredQuat[1] + alpha*(simQuat[1]-filteredQuat[1]),
                filteredQuat[2] + alpha*(simQuat[2]-filteredQuat[2]),
                filteredQuat[3] + alpha*(simQuat[3]-filteredQuat[3]),
                filteredQuat[4] + alpha*(simQuat[4]-filteredQuat[4])
            }

            local norm = math.sqrt(
                filteredQuat[1]^2 +
                filteredQuat[2]^2 +
                filteredQuat[3]^2 +
                filteredQuat[4]^2
            )

            filteredQuat = {
                filteredQuat[1]/norm,
                filteredQuat[2]/norm,
                filteredQuat[3]/norm,
                filteredQuat[4]/norm
            }
        end

        sim.setObjectQuaternion(hand,-1,filteredQuat)
        local movingDummy = sim.getObject('/hand_root')
        local kukaTarget = sim.getObject('/LBRiiwa14R820/kuka_target')

        local p = sim.getObjectPosition(movingDummy,-1)
        local q = sim.getObjectQuaternion(movingDummy,-1)

        sim.setObjectPosition(kukaTarget,-1,p)
        sim.setObjectQuaternion(kukaTarget,-1,q)

    end

    frameIndex = frameIndex + 1
    local t = sim.getSimulationTime()
    -- INDEX / POINTER
    driveFinger(line, "XRHand_Index", pointer_prox, pointer_mid, pointer_dist)

    -- MIDDLE
    driveFinger(line, "XRHand_Middle", middle_prox, middle_mid, middle_dist)

    -- RING
    driveFinger(line, "XRHand_Ring", ring_prox, ring_mid, ring_dist)
    

    -- ===== THUMB METACARPAL + BASE (CORRECTED DUAL CONTROL) =====

    local tm = {line:match('"XRHand_ThumbMetacarpal":%[(.-),(.-),(.-),(.-)%]')}

    if tm[1] then
        local qx,qy,qz,qw =
            tonumber(tm[1]),
            tonumber(tm[2]),
            tonumber(tm[3]),
            tonumber(tm[4])

        -- 1. PROPER EULER CONVERSION (Z-Y-X Convention)
        -- This isolates the axes and prevents the distortion caused by multi-axis rotations
        local sqx = qx * qx
        local sqy = qy * qy
        local sqz = qz * qz

        -- Twist (Rotation around X-axis)
        local roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (sqx + sqy))
        
        -- Swing (Rotation around Y-axis) - clamped to prevent math.asin from returning NaN
        local pitch = math.asin(math.max(-1.0, math.min(1.0, 2.0 * (qw * qy - qz * qx))))

        -- Raw signals correctly decoupled
        local swingRaw = pitch  -- side-to-side (Y-axis)
        local twistRaw = roll   -- rotation/opposition (X-axis)

        -- ===== BASELINE =====
        if thumbMetaBaseline == nil then
            thumbMetaBaseline = swingRaw
            thumbBaseBaseline = twistRaw
        end

        -- Relative motion
        local swing = swingRaw - thumbMetaBaseline
        local twist = twistRaw - thumbBaseBaseline

        -- ===== SCALE + SIGN (tune these depending on how fast you want the Allegro hand to respond) =====
        swing = -swing * 5.0
        twist =  twist * 4.0

        -- ===== OFFSETS =====
        local metaOffset = math.rad(15.069)
        local baseOffset = 0  -- adjust later if needed

        local metaAngle = metaOffset + swing
        local baseAngle = baseOffset + twist

        -- ===== CLAMP (CORRECTED LIMITS) =====
        local metaMin = math.rad(15.069)
        local metaMax = math.rad(79.985)

        if metaAngle < metaMin then metaAngle = metaMin end
        if metaAngle > metaMax then metaAngle = metaMax end

        -- (approximate base limits ? adjust later if needed)
        local baseMin = math.rad(-45)
        local baseMax = math.rad(45)

        if baseAngle < baseMin then baseAngle = baseMin end
        if baseAngle > baseMax then baseAngle = baseMax end

        -- ===== APPLY BOTH JOINTS =====
        sim.setJointPosition(thumb_meta, metaAngle)
        sim.setJointPosition(thumb_base, baseAngle)
    end
    -- Thumb proximal
    local tp = {line:match('"XRHand_ThumbProximal":%[(.-),(.-),(.-),(.-)%]')}
    if tp[1] then
        local qx,qy,qz,qw = tonumber(tp[1]),tonumber(tp[2]),tonumber(tp[3]),tonumber(tp[4])
        sim.setJointPosition(thumb_prox, 2*math.atan2(qx,qw))
    end

    -- Thumb distal
    local td = {line:match('"XRHand_ThumbDistal":%[(.-),(.-),(.-),(.-)%]')}
    if td[1] then
        local qx,qy,qz,qw = tonumber(td[1]),tonumber(td[2]),tonumber(td[3]),tonumber(td[4])
        sim.setJointPosition(thumb_dist, 2*math.atan2(qx,qw))
    end

end