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

VECTOR_TRACKED_FIELDS = {
    "indexTipRelativeToWrist",
    "middleTipRelativeToWrist",
    "ringTipRelativeToWrist",
    "littleTipRelativeToWrist",
    "thumbTipRelativeToWrist",
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


@dataclass(frozen=True)
class RecordingMetadata:
    record_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class RecordingData:
    frames: list[HandFrame]
    metadata: list[RecordingMetadata]


ALLEGRO_REFERENCE_DISTANCES_METERS = {
    ("XRHand_IndexProximal", "XRHand_IndexIntermediate"): 0.0540,
    ("XRHand_IndexIntermediate", "XRHand_IndexDistal"): 0.0384,
    ("XRHand_IndexDistal", "XRHand_IndexTip"): 0.0267,
    ("XRHand_MiddleProximal", "XRHand_MiddleIntermediate"): 0.0540,
    ("XRHand_MiddleIntermediate", "XRHand_MiddleDistal"): 0.0384,
    ("XRHand_MiddleDistal", "XRHand_MiddleTip"): 0.0267,
    ("XRHand_RingProximal", "XRHand_RingIntermediate"): 0.0540,
    ("XRHand_RingIntermediate", "XRHand_RingDistal"): 0.0384,
    ("XRHand_RingDistal", "XRHand_RingTip"): 0.0267,
}

HAND_ALIGNMENT_MEASUREMENT_KEYS = (
    "XRHand_IndexProximal",
    "XRHand_MiddleProximal",
    "XRHand_RingProximal",
    "XRHand_ThumbMetacarpal",
)


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


def load_recording_data(path: str | Path, root_pose_preference: list[str]) -> RecordingData:
    recording_path = Path(path)
    frames: list[HandFrame] = []
    metadata: list[RecordingMetadata] = []

    for payload in _read_json_lines(recording_path):
        record_type = str(payload.get("recordType", "frame"))
        if record_type != "frame":
            metadata.append(
                RecordingMetadata(
                    record_type=record_type,
                    payload=payload,
                )
            )
            continue
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

    return RecordingData(frames=frames, metadata=metadata)


def load_recording(path: str | Path, root_pose_preference: list[str]) -> list[HandFrame]:
    return load_recording_data(path, root_pose_preference).frames


def compute_hand_scale_factor(metadata: list[RecordingMetadata]) -> float | None:
    metadata_record = next(
        (entry.payload for entry in metadata if entry.record_type == "metadata"),
        None,
    )
    if not metadata_record:
        return None

    if str(metadata_record.get("jointDistanceUnit", "meters")).lower() != "meters":
        return None

    joint_distances = metadata_record.get("jointDistances")
    if not isinstance(joint_distances, list):
        return None

    ratios: list[float] = []
    for item in joint_distances:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("from", "")), str(item.get("to", "")))
        reference_distance = ALLEGRO_REFERENCE_DISTANCES_METERS.get(key)
        if reference_distance is None or reference_distance <= 0.0:
            continue
        measured_distance = item.get("distance")
        if not isinstance(measured_distance, (int, float)) or measured_distance <= 0.0:
            continue
        ratios.append(float(measured_distance) / reference_distance)

    if not ratios:
        return None

    ratios.sort()
    midpoint = len(ratios) // 2
    if len(ratios) % 2 == 1:
        return ratios[midpoint]
    return 0.5 * (ratios[midpoint - 1] + ratios[midpoint])


def compute_hand_alignment_offset(metadata: list[RecordingMetadata]) -> list[float] | None:
    metadata_record = next(
        (entry.payload for entry in metadata if entry.record_type == "metadata"),
        None,
    )
    if not metadata_record:
        return None

    measurements = metadata_record.get("wristToFingerBaseMeasurements")
    if not isinstance(measurements, list):
        return None

    offsets: list[list[float]] = []
    for key in HAND_ALIGNMENT_MEASUREMENT_KEYS:
        match = next(
            (
                item
                for item in measurements
                if isinstance(item, dict)
                and str(item.get("to", "")) == key
                and _is_vec(item.get("offset"), 3)
            ),
            None,
        )
        if match is None:
            continue
        offsets.append(list(map(float, match["offset"])))

    if not offsets:
        return None

    return [
        sum(offset[index] for offset in offsets) / len(offsets)
        for index in range(3)
    ]


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
        if key in VECTOR_TRACKED_FIELDS and _is_vec(frame.get(key), 3):
            tracked[key] = Pose(
                position=list(map(float, frame[key])),
                rotation=[0.0, 0.0, 0.0, 1.0],
            )
            continue
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
