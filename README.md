# GRPDC-CoppeliaSim

This repository contains the CoppeliaSim half of the Gamified Robotic Pose Data Collection Project. It replays Quest hand and tracked-object recordings inside CoppeliaSim using the ZeroMQ Remote API.

The other half of the project lives here:

- Quest 3 repository: [GRPDC-Quest3](https://github.com/Evan-H-Rosenthal/GRPDC-Quest3)

## Project layout

- `coppelia/scenes/Follower_with_IKs.ttt`: main scene to open in CoppeliaSim
- `coppelia/scripts/remote_playback.py`: replays a recording into a running CoppeliaSim scene
- `coppelia/scripts/inspect_recording.py`: prints a quick summary of a recording file
- `coppelia/config/free_allegro_hand.json`: main playback/configuration file
- `coppelia/recordings/`: example output location for recording files

## Software setup

1. Install Python 3 if you do not already have it.
2. Install the Python dependency:

```powershell
python -m pip install -r .\coppelia\requirements.txt
```

3. Download the latest CoppeliaSim build from the official Coppelia Robotics download page:

- https://www.coppeliarobotics.com/

Use the download section on that page to pick the correct platform/build.

## Which scene to open

Open this scene:

- `coppelia/scenes/Follower_with_IKs.ttt`

This is the IK-enabled scene intended for the playback workflow in this repository.

## Running playback from Python

1. Launch CoppeliaSim.
2. Open `coppelia/scenes/Follower_with_IKs.ttt`.
3. Make sure the ZeroMQ Remote API add-on is available in your CoppeliaSim installation.
4. Start playback from this repository with a recording file.

Example:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260425_170958.json --start-sim --realtime
```

Useful variants:

Inspect a recording before playback:

```powershell
python .\coppelia\scripts\inspect_recording.py .\coppelia\recordings\hand_recording_20260425_170958.json --config .\coppelia\config\free_allegro_hand.json
```

Validate parsing without connecting to CoppeliaSim:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260425_170958.json --dry-run
```

Use the most recent recording in `coppelia/recordings`:

```powershell
python .\coppelia\scripts\remote_playback.py --latest --start-sim --realtime
```

Run without recorded timing, stepping frames as fast as possible:

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260425_170958.json --start-sim
```

## Port and Remote API configuration

The playback script reads its host/port from:

- `coppelia/config/free_allegro_hand.json`

Current defaults:

- host: `localhost`
- port: `23001`

You can configure the Remote API connection in either of these ways:

1. Edit `coppelia/config/free_allegro_hand.json`

```json
"remoteApi": {
  "host": "localhost",
  "port": 23001
}
```

2. Override it on the command line

```powershell
python .\coppelia\scripts\remote_playback.py .\coppelia\recordings\hand_recording_20260425_170958.json --host localhost --port 23001
```

If the port in CoppeliaSim and the port in this script do not match, the Python client will not connect.

## Parameters you can tune

Most tunable playback behavior lives in `coppelia/config/free_allegro_hand.json`.

Important sections:

- `remoteApi`: host and port for the ZeroMQ connection
- `playback.smoothingAlpha`: smoothing for the hand root motion
- `playback.rootPosePreference`: order of preferred root pose fields from the recording
- `transform`: axis remapping, sign flips, and offsets between recording space and CoppeliaSim space
- `jointAngleGains`: per-joint scaling for finger and thumb motion
- `fingerPlayback`: enable or disable index, middle, and ring finger playback
- `thumbPlayback.enabled`: enable or disable thumb playback
- `thumbCoupling`: thumb joint coupling and angle limits
- `sceneScaling`: scale the hand model, optionally from recording metadata
- `sceneAlignment`: adjust hand alignment offsets in the scene
- `trackedObjectSmoothing`: smoothing for fingertip targets and cubes
- `trackedObjectRotationOutlier`: reject sudden rotation jumps
- `trackedObjectRotationRetargeting`: stabilize cube rotations
- `trackedObjectPositionOutlier`: reject sudden position jumps
- `trackedObjectPlayback`: enable/disable tracked-object rotation playback and choose rotation modes

The playback script also exposes useful command-line parameters:

- `--config`: choose a different config file
- `--speed`: scale playback timing
- `--realtime`: use recorded timestamps
- `--start-sim`: start the simulation automatically if it is stopped
- `--no-stepping`: disable Remote API stepping mode
- `--dry-run`: parse the recording without connecting to CoppeliaSim
- `--root-pose`: force a specific root pose source
- `--host` and `--port`: override Remote API connection settings
- `--latest`: use the newest recording in `coppelia/recordings`
- `--no-dialog`: disable the file picker when no recording path is supplied

## Notes for future researchers

- If scene object names change, update the aliases in the config file so the Python client can still resolve them.
- If playback looks mirrored, rotated, or offset, start by checking the `transform` and `sceneAlignment` sections in `coppelia/config/free_allegro_hand.json`.
- If motion is too noisy or too sluggish, tune `playback.smoothingAlpha` and the `trackedObjectSmoothing` values.
- If cubes snap to bad poses, tune the tracked-object outlier and retargeting settings.
