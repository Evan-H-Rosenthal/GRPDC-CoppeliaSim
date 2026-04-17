from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COPPELIA_DIR = SCRIPT_DIR.parent
PYTHON_DIR = COPPELIA_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from hand_playback.scene import load_scene_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List scene object aliases from a running CoppeliaSim instance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=COPPELIA_DIR / "config" / "scene_paths.json",
    )
    parser.add_argument("--host", help="Override the host from config.")
    parser.add_argument("--port", type=int, help="Override the ZeroMQ port from config.")
    parser.add_argument(
        "--contains",
        default="",
        help="Only print aliases containing this text (case-insensitive).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of aliases to print.",
    )
    args = parser.parse_args()

    config = load_scene_config(args.config)
    host = args.host or config["remoteApi"]["host"]
    port = args.port if args.port is not None else int(config["remoteApi"]["port"])

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print(f"Connecting to {host}:{port}...")
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")

    root = sim.getObject("/")
    objects = sim.getObjectsInTree(root, sim.handle_all, 0)

    needle = args.contains.lower()
    count = 0
    for handle in objects:
        alias = sim.getObjectAlias(handle, 5)
        if needle and needle not in alias.lower():
            continue
        print(alias)
        count += 1
        if count >= args.limit:
            break

    print(f"Printed {count} aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
