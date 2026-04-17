from __future__ import annotations

import argparse
from pathlib import Path


def resolve_package_uris(urdf_path: Path, package_root: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    replacement = package_root.as_posix().rstrip("/") + "/"
    resolved = text.replace("package://", replacement)

    output_path = urdf_path.with_name(urdf_path.stem + "_resolved" + urdf_path.suffix)
    output_path.write_text(resolved, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a URDF copy with package:// mesh paths resolved to absolute paths."
    )
    parser.add_argument("urdf", type=Path, help="Path to the source URDF file.")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "allegro_hand_ros-master",
        help="Directory that should replace package://",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = resolve_package_uris(args.urdf, args.package_root)
    print(f"Resolved URDF written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
