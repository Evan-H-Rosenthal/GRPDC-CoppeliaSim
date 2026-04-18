from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COPPELIA_DIR = SCRIPT_DIR.parent
PYTHON_DIR = COPPELIA_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from hand_playback.recording import load_recording
from hand_playback.scene import SceneController, load_scene_config


DEFAULT_CONFIG_PATH = COPPELIA_DIR / "config" / "free_allegro_hand.json"
DEFAULT_RECORDINGS_DIR = COPPELIA_DIR / "recordings"
DEFAULT_REMOTE_API_HOST_OVERRIDE: str | None = None
DEFAULT_REMOTE_API_PORT_OVERRIDE: int | None = 23000
DEFAULT_START_SIMULATION = True
DEFAULT_USE_REALTIME = True
DEFAULT_STEPPING_DISABLED = False
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_ROOT_POSE: str | None = None
DEFAULT_OPEN_FILE_DIALOG = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a Quest hand recording into CoppeliaSim over the ZeroMQ Remote API."
    )
    parser.add_argument(
        "recording",
        nargs="?",
        type=Path,
        help="Path to a line-delimited JSON recording file. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the scene object mapping config.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_PLAYBACK_SPEED,
        help="Playback speed multiplier when using recorded timing.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep according to recorded timestamps instead of stepping each frame immediately.",
    )
    parser.add_argument(
        "--start-sim",
        action="store_true",
        help="Start the CoppeliaSim simulation if it is currently stopped.",
    )
    parser.add_argument(
        "--no-stepping",
        action="store_true",
        help="Disable Remote API stepping mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and normalize the recording without connecting to CoppeliaSim.",
    )
    parser.add_argument(
        "--root-pose",
        choices=[
            "handRootTableFrozen",
            "handRootTable",
            "OpenXRRightHand",
            "handRootWorld",
            "handRootLocal",
        ],
        help="Force a specific root pose field to be used first.",
    )
    parser.add_argument(
        "--host",
        help="Override the host from the config file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override the ZeroMQ Remote API port from the config file.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified recording from the recordings folder.",
    )
    parser.add_argument(
        "--no-dialog",
        action="store_true",
        help="Do not open a file picker when no recording path is supplied.",
    )
    return parser


def choose_recording_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = filedialog.askopenfilename(
        title="Select hand recording",
        initialdir=str(DEFAULT_RECORDINGS_DIR),
        filetypes=[("JSON recordings", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(filename) if filename else None


def get_latest_recording() -> Path | None:
    recordings = sorted(
        DEFAULT_RECORDINGS_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return recordings[0] if recordings else None


def main() -> int:
    args = build_parser().parse_args()
    recording = args.recording
    if recording is None and args.latest:
        recording = get_latest_recording()
    if recording is None and DEFAULT_OPEN_FILE_DIALOG and not args.no_dialog:
        recording = choose_recording_file()
    if recording is None:
        raise SystemExit(
            "No recording selected. Pass a file path, use --latest, or allow the file dialog."
        )

    config = load_scene_config(args.config)
    host_override = args.host or DEFAULT_REMOTE_API_HOST_OVERRIDE
    port_override = args.port if args.port is not None else DEFAULT_REMOTE_API_PORT_OVERRIDE
    if host_override:
        config["remoteApi"]["host"] = host_override
    if port_override is not None:
        config["remoteApi"]["port"] = port_override
    root_pose_override = args.root_pose or DEFAULT_ROOT_POSE
    if root_pose_override:
        existing = [
            candidate
            for candidate in config["playback"]["rootPosePreference"]
            if candidate != root_pose_override
        ]
        config["playback"]["rootPosePreference"] = [root_pose_override, *existing]
    frames = load_recording(
        recording,
        root_pose_preference=config["playback"]["rootPosePreference"],
    )

    print(
        f"Loaded {len(frames)} frames from {recording} "
        f"using root preference {config['playback']['rootPosePreference']}."
    )
    if args.dry_run:
        first = frames[0]
        print(
            "First frame:",
            {
                "time": first.timestamp,
                "source_space": first.source_space,
                "root_position": first.root_pose.position,
            },
        )
        return 0

    controller = SceneController(config)
    controller.connect()
    use_realtime = args.realtime or DEFAULT_USE_REALTIME
    start_simulation = args.start_sim or DEFAULT_START_SIMULATION
    stepping = not (args.no_stepping or DEFAULT_STEPPING_DISABLED)
    if use_realtime and stepping:
        print("Realtime playback requested; disabling stepping mode.")
        stepping = False
    controller.playback(
        frames,
        use_recorded_timing=use_realtime,
        speed=args.speed,
        start_simulation=start_simulation,
        stepping=stepping,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
