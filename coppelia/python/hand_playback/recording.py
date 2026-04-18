from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FINGER_PREFIXES = {
    "index": "XRHand_Index",
    "middle": "XRHand_Middle",
    "ring": "XRHand_Ring",
    "little": "XRHand_Little",
    "thumb": "XRHand_Thumb",
}


@dataclass(frozen=True)
class Pose:
    position: list[float]
    rotation: list[float]


@dataclass(frozen=True)
class HandFrame:
    timestamp: float
    source_space: str
    root_pose: Pose
    joint_rotations: dict[str, list[float]]
    table_origin_world: Pose | None
    table_origin_recording_start_world: Pose | None
    table_origin_frozen: bool
    tracked_objects: dict[str, Pose]
    raw: dict[str, Any]


def _read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first_non_empty = ""
        while True:
            cursor = handle.tell()
            line = handle.readline()
            if not line:
                return
            if line.strip():
                first_non_empty = line.lstrip()
                handle.seek(cursor)
                break

        if first_non_empty.startswith("["):
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"{path} is not a JSON array.")
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _read_pose(frame: dict[str, Any], candidates: list[str]) -> tuple[str, Pose]:
    for field_name in candidates:
        if field_name == "handRootTableFrozen":
            if not frame.get("tableOriginFrozen"):
                continue
            value = frame.get("handRootTable")
            if isinstance(value, dict):
                pos = value.get("pos")
                rot = value.get("rot")
                if _is_vec(pos, 3) and _is_vec(rot, 4):
                    return field_name, Pose(position=list(map(float, pos)), rotation=list(map(float, rot)))
        value = frame.get(field_name)
        if not isinstance(value, dict):
            continue
        pos = value.get("pos")
        rot = value.get("rot")
        if _is_vec(pos, 3) and _is_vec(rot, 4):
            return field_name, Pose(position=list(map(float, pos)), rotation=list(map(float, rot)))
    raise ValueError(
        "Recording frame does not contain any supported root pose fields: "
        + ", ".join(candidates)
    )


def _is_vec(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length


def load_recording(path: str | Path, root_pose_preference: list[str]) -> list[HandFrame]:
    recording_path = Path(path)
    frames: list[HandFrame] = []

    for payload in _read_json_lines(recording_path):
        source_space, root_pose = _read_pose(payload, root_pose_preference)
        frames.append(
            HandFrame(
                timestamp=float(payload.get("time", 0.0)),
                source_space=str(payload.get("rootSpace", source_space)),
                root_pose=root_pose,
                joint_rotations=_extract_joint_rotations(payload),
                table_origin_world=_read_optional_pose(payload, "tableOriginWorld"),
                table_origin_recording_start_world=_read_optional_pose(
                    payload,
                    "tableOriginRecordingStartWorld",
                ),
                table_origin_frozen=bool(payload.get("tableOriginFrozen", False)),
                tracked_objects=_extract_tracked_objects(payload),
                raw=payload,
            )
        )

    if not frames:
        raise ValueError(f"No frames were loaded from {recording_path}.")

    return frames


def _extract_joint_rotations(frame: dict[str, Any]) -> dict[str, list[float]]:
    joints: dict[str, list[float]] = {}
    for prefix in FINGER_PREFIXES.values():
        for suffix in ("Metacarpal", "Proximal", "Intermediate", "Distal"):
            key = f"{prefix}{suffix}"
            value = frame.get(key)
            if _is_vec(value, 4):
                joints[key] = list(map(float, value))
    return joints


def _read_optional_pose(frame: dict[str, Any], field_name: str) -> Pose | None:
    value = frame.get(field_name)
    if not isinstance(value, dict):
        return None
    pos = value.get("pos")
    rot = value.get("rot")
    if _is_vec(pos, 3) and _is_vec(rot, 4):
        return Pose(position=list(map(float, pos)), rotation=list(map(float, rot)))
    return None


def _extract_tracked_objects(frame: dict[str, Any]) -> dict[str, Pose]:
    tracked: dict[str, Pose] = {}
    for key in frame.keys():
        if not key.endswith(("Table", "World", "Local")):
            continue
        if key in {
            "handRootTable",
            "handRootWorld",
            "handRootLocal",
            "tableOriginWorld",
            "tableOriginRecordingStartWorld",
            "OpenXRRightHand",
        }:
            continue
        pose = _read_optional_pose(frame, key)
        if pose is None:
            continue
        prefix = key.removesuffix("Table").removesuffix("World").removesuffix("Local")
        tracked_flag = frame.get(f"{prefix}Tracked")
        if tracked_flag is False:
            continue
        tracked[key] = pose
    return tracked
