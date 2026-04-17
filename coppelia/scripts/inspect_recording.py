from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COPPELIA_DIR = SCRIPT_DIR.parent
PYTHON_DIR = COPPELIA_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from hand_playback.recording import load_recording
from hand_playback.scene import load_scene_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a hand recording for schema support.")
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=COPPELIA_DIR / "config" / "scene_paths.json",
    )
    args = parser.parse_args()

    config = load_scene_config(args.config)
    frames = load_recording(
        args.recording,
        root_pose_preference=config["playback"]["rootPosePreference"],
    )

    source_spaces = Counter(frame.source_space for frame in frames)
    root_fields = Counter()
    frozen_frames = sum(1 for frame in frames if frame.table_origin_frozen)
    recording_start_frames = sum(
        1 for frame in frames if frame.table_origin_recording_start_world is not None
    )
    for frame in frames:
        for candidate in config["playback"]["rootPosePreference"]:
            if candidate == "handRootTableFrozen":
                if frame.table_origin_frozen and "handRootTable" in frame.raw:
                    root_fields[candidate] += 1
                    break
            elif candidate in frame.raw:
                root_fields[candidate] += 1
                break

    print(f"frames={len(frames)}")
    print(f"duration_seconds={frames[-1].timestamp - frames[0].timestamp:.3f}")
    print(f"source_spaces={dict(source_spaces)}")
    print(f"root_fields_used={dict(root_fields)}")
    print(f"table_origin_frozen_frames={frozen_frames}")
    print(f"table_origin_recording_start_frames={recording_start_frames}")
    print(f"first_frame_keys={sorted(frames[0].raw.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
