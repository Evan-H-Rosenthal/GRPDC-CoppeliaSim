# CoppeliaSim Remote API Migration

This folder now has the beginnings of an external playback pipeline so the `.ttt` scene can focus on simulation and scene state, while Python owns recording parsing, schema adaptation, and playback orchestration.

## What moved out of the scene

- Recording file loading
- Old-vs-new JSON schema handling
- Root pose selection (`handRootTable` first, then older fallbacks)
- Unity-to-CoppeliaSim coordinate conversion
- Wrist smoothing
- Finger and thumb joint target generation

## New entrypoints

- `coppelia/scripts/remote_playback.py`
  Plays a recording into CoppeliaSim over the ZeroMQ Remote API.
- `coppelia/scripts/inspect_recording.py`
  Summarizes a recording so schema changes are easier to spot.
- `coppelia/config/scene_paths.json`
  The only file that should need edits when scene object names change.

## Install

```powershell
python -m pip install -r .\coppelia\requirements.txt
```

## First-use checklist

1. Open `coppelia/scenes/HandDataPointFollower.ttt` in CoppeliaSim.
2. Make sure the ZeroMQ Remote API add-on is available in your CoppeliaSim install.
3. Confirm the object names in `coppelia/config/scene_paths.json` match the scene aliases exactly.
4. Remove or disable the old in-scene playback script so it does not fight the Remote API client.

## Example commands

Inspect a recording:

```powershell
python .\coppelia\scripts\inspect_recording.py .\coppelia\recordings\hand_recording_20260411_203916.json
```

Validate parsing without connecting to the simulator:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260411_203916.json --dry-run
```

Play into CoppeliaSim with recorded timing:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260411_203916.json --start-sim --realtime
```

Play as fast as possible in stepping mode:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260411_203916.json --start-sim
```

## Suggested scene contract going forward

Keep the scene responsible for:

- Robot model, IK, collisions, dynamics, and sensors
- Named dummies/joints that the Remote API can target
- Optional child scripts only when they expose reusable sim-side services

Keep Python responsible for:

- Recording format adapters
- Playback sequencing
- Scene setup from recording metadata
- Future object spawning and environment placement

That split should make future JSON changes mostly a Python concern rather than a `.ttt` surgery problem.
