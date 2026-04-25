from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recording import HandFrame
from .transforms import (
    align_quaternion_hemisphere,
    ThumbCalibration,
    lerp_vector,
    nearest_cube_equivalent_rotation,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_yaw_only,
    quat_to_single_axis_joint,
    remap_quaternion,
    remap_vector,
    solve_thumb_root_angles,
    thumb_angles_from_metacarpal,
    thumb_angles_from_metacarpal_xz,
)


@dataclass(frozen=True)
class JointHandles:
    pointer_prox: int | None = None
    pointer_mid: int | None = None
    pointer_dist: int | None = None
    middle_prox: int | None = None
    middle_mid: int | None = None
    middle_dist: int | None = None
    ring_prox: int | None = None
    ring_mid: int | None = None
    ring_dist: int | None = None
    thumb_meta: int | None = None
    thumb_base: int | None = None
    thumb_prox: int | None = None
    thumb_dist: int | None = None


@dataclass(frozen=True)
class SceneHandles:
    root_object: int
    hand_model: int | None
    moving_dummy: int | None
    kuka_target: int | None
    joints: JointHandles
    tracked_objects: dict[str, int]


@dataclass
class FilterState:
    position: list[float] | None = None
    quaternion: list[float] | None = None
    held_rotation_frames: int = 0
    held_position_frames: int = 0
    rotation_correction: list[float] | None = None


class SceneController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._client = None
        self.sim = None
        self.handles: SceneHandles | None = None
        self.filtered_position: list[float] | None = None
        self.filtered_quaternion: list[float] | None = None
        self.thumb_calibration = ThumbCalibration()
        self.thumb_heuristic_calibration = ThumbCalibration()
        self.tracked_object_filters: dict[str, FilterState] = {}

    def _joint_gain(self, key: str, default: float = 1.0) -> float:
        gains = self.config.get("jointAngleGains", {})
        return float(gains.get(key, gains.get("default", default)))

    def connect(self) -> None:
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'coppeliasim-zmqremoteapi-client'. "
                "Install it with 'python -m pip install -r coppelia/requirements.txt'."
            ) from exc

        remote_api = self.config["remoteApi"]
        print(
            f"Connecting to CoppeliaSim ZeroMQ Remote API at "
            f"{remote_api.get('host', 'localhost')}:{int(remote_api.get('port', 23001))}..."
        )
        self._client = RemoteAPIClient(
            host=remote_api.get("host", "localhost"),
            port=int(remote_api.get("port", 23001)),
        )
        self.sim = self._client.require("sim")
        self.handles = self._resolve_handles()
        self._apply_scene_scaling()
        self._apply_hand_alignment()
        print("Connected and resolved scene handles.")

    def _resolve_handles(self) -> SceneHandles:
        if self.sim is None:
            raise RuntimeError("Remote API is not connected.")

        aliases = self.config["sceneObjects"]
        tracked_objects_config = self.config.get("trackedObjects", {})
        return SceneHandles(
            root_object=self._get_object(aliases["rootObject"]),
            hand_model=self._get_optional_object(aliases.get("handModel")),
            moving_dummy=self._get_optional_object(aliases.get("movingDummy")),
            kuka_target=self._get_optional_object(aliases.get("kukaTarget")),
            joints=JointHandles(
                pointer_prox=self._get_optional_object(aliases.get("pointerProx")),
                pointer_mid=self._get_optional_object(aliases.get("pointerMid")),
                pointer_dist=self._get_optional_object(aliases.get("pointerDist")),
                middle_prox=self._get_optional_object(aliases.get("middleProx")),
                middle_mid=self._get_optional_object(aliases.get("middleMid")),
                middle_dist=self._get_optional_object(aliases.get("middleDist")),
                ring_prox=self._get_optional_object(aliases.get("ringProx")),
                ring_mid=self._get_optional_object(aliases.get("ringMid")),
                ring_dist=self._get_optional_object(aliases.get("ringDist")),
                thumb_meta=self._get_optional_object(aliases.get("thumbMeta")),
                thumb_base=self._get_optional_object(aliases.get("thumbBase")),
                thumb_prox=self._get_optional_object(aliases.get("thumbProx")),
                thumb_dist=self._get_optional_object(aliases.get("thumbDist")),
            ),
            tracked_objects={
                field_name: self._get_object(alias)
                for field_name, alias in tracked_objects_config.items()
            },
        )

    def _get_object(self, alias: str) -> int:
        if self.sim is None:
            raise RuntimeError("Remote API is not connected.")
        return self.sim.getObject(alias)

    def _get_optional_object(self, alias: str | None) -> int | None:
        if not alias:
            return None
        return self._get_object(alias)

    def _apply_scene_scaling(self) -> None:
        if self.sim is None or self.handles is None:
            return

        scaling = self.config.get("sceneScaling", {})
        hand_scale = float(scaling.get("handScaleFactor", 1.0))
        if self.handles.hand_model is not None:
            property_name = "customData.codexAppliedHandScale"
            applied_scale = self.sim.getFloatProperty(
                self.handles.hand_model,
                property_name,
                {"noError": True},
            )
            if applied_scale is None or applied_scale <= 0.0:
                applied_scale = 1.0
            relative_scale = hand_scale / applied_scale
            if abs(relative_scale - 1.0) > 1e-6:
                self.sim.scaleObjects([self.handles.hand_model], relative_scale, False)
                print(
                    f"Scaled hand model by factor {relative_scale:.3f} "
                    f"(target absolute scale {hand_scale:.3f})."
                )
            self.sim.setFloatProperty(
                self.handles.hand_model,
                property_name,
                hand_scale,
            )

    def _apply_hand_alignment(self) -> None:
        if self.sim is None or self.handles is None or self.handles.hand_model is None:
            return

        alignment = self.config.get("sceneAlignment", {})
        target_hand_model_offset = alignment.get("targetHandModelOffset")
        if isinstance(target_hand_model_offset, list) and len(target_hand_model_offset) == 3:
            property_names = [
                "customData.codexAppliedHandOffsetX",
                "customData.codexAppliedHandOffsetY",
                "customData.codexAppliedHandOffsetZ",
            ]
            applied_delta: list[float] = []
            for property_name in property_names:
                value = self.sim.getFloatProperty(
                    self.handles.hand_model,
                    property_name,
                    {"noError": True},
                )
                applied_delta.append(0.0 if value is None else float(value))

            requested_delta = [float(value) for value in target_hand_model_offset]
            relative_delta = [
                requested_delta[index] - applied_delta[index]
                for index in range(3)
            ]

            if any(abs(value) > 1e-6 for value in relative_delta):
                local_position = self.sim.getObjectPosition(
                    self.handles.hand_model,
                    self.handles.root_object,
                )
                new_local_position = [
                    local_position[index] + relative_delta[index]
                    for index in range(3)
                ]
                self.sim.setObjectPosition(
                    self.handles.hand_model,
                    self.handles.root_object,
                    new_local_position,
                )
                print(
                    "Applied direct hand model offset delta "
                    f"{[round(value, 6) for value in relative_delta]} "
                    f"toward target {[round(value, 6) for value in requested_delta]}."
                )

            for index, property_name in enumerate(property_names):
                self.sim.setFloatProperty(
                    self.handles.hand_model,
                    property_name,
                    requested_delta[index],
                )
            return

        target_offset = alignment.get("targetFingerBaseOffset")
        if not isinstance(target_offset, list) or len(target_offset) != 3:
            return

        apply_axes = alignment.get("applyAxes", [False, False, True])
        if not isinstance(apply_axes, list) or len(apply_axes) != 3:
            apply_axes = [False, False, True]

        samples: list[list[float]] = []
        joints = self.handles.joints
        for handle in (
            joints.pointer_prox,
            joints.middle_prox,
            joints.ring_prox,
            joints.thumb_meta,
        ):
            if handle is None:
                continue
            samples.append(self.sim.getObjectPosition(handle, self.handles.root_object))

        if not samples:
            return

        current_average = [
            sum(sample[index] for sample in samples) / len(samples)
            for index in range(3)
        ]
        requested_delta = [
            (float(target_offset[index]) - current_average[index]) if bool(apply_axes[index]) else 0.0
            for index in range(3)
        ]

        property_names = [
            "customData.codexAppliedHandOffsetX",
            "customData.codexAppliedHandOffsetY",
            "customData.codexAppliedHandOffsetZ",
        ]
        applied_delta: list[float] = []
        for property_name in property_names:
            value = self.sim.getFloatProperty(
                self.handles.hand_model,
                property_name,
                {"noError": True},
            )
            applied_delta.append(0.0 if value is None else float(value))

        relative_delta = [
            requested_delta[index] - applied_delta[index]
            for index in range(3)
        ]

        if any(abs(value) > 1e-6 for value in relative_delta):
            local_position = self.sim.getObjectPosition(
                self.handles.hand_model,
                self.handles.root_object,
            )
            new_local_position = [
                local_position[index] + relative_delta[index]
                for index in range(3)
            ]
            self.sim.setObjectPosition(
                self.handles.hand_model,
                self.handles.root_object,
                new_local_position,
            )
            print(
                "Applied hand alignment delta "
                f"{[round(value, 6) for value in relative_delta]} "
                f"toward target offset {[round(value, 6) for value in target_offset]}."
            )

        for index, property_name in enumerate(property_names):
            self.sim.setFloatProperty(
                self.handles.hand_model,
                property_name,
                requested_delta[index],
            )

    def apply_frame(self, frame: HandFrame) -> None:
        if self.sim is None or self.handles is None:
            raise RuntimeError("Remote API is not connected.")

        alpha = float(self.config["playback"].get("smoothingAlpha", 0.2))
        transform = self.config.get("transform", {})
        position = remap_vector(
            frame.root_pose.position,
            axes=transform.get("positionAxes", ["z", "x", "y"]),
            signs=transform.get("positionSigns", [1.0, 1.0, 1.0]),
            offset=transform.get("positionOffset", [0.0, 0.0, 0.0]),
        )
        rotation = remap_quaternion(
            frame.root_pose.rotation,
            axes=transform.get("rotationAxes", ["z", "x", "y"]),
            signs=transform.get("rotationSigns", [1.0, 1.0, 1.0]),
        )
        rotation = align_quaternion_hemisphere(self.filtered_quaternion, rotation)

        self.filtered_position = lerp_vector(self.filtered_position, position, alpha)
        self.filtered_quaternion = normalize_quaternion(
            lerp_vector(self.filtered_quaternion, rotation, alpha)
        )

        if self.handles.kuka_target is not None:
            pose_handle = self.handles.kuka_target
        else:
            pose_handle = self.handles.root_object

        self.sim.setObjectPosition(pose_handle, -1, self.filtered_position)
        self.sim.setObjectQuaternion(pose_handle, -1, self.filtered_quaternion)

        joints = self.handles.joints
        rotations = frame.joint_rotations
        finger_playback = self.config.get("fingerPlayback", {})
        if bool(finger_playback.get("index", True)):
            self._set_joint_if_present(joints.pointer_prox, rotations, "XRHand_IndexProximal")
            self._set_joint_if_present(joints.pointer_mid, rotations, "XRHand_IndexIntermediate")
            self._set_joint_if_present(joints.pointer_dist, rotations, "XRHand_IndexDistal")
        if bool(finger_playback.get("middle", True)):
            self._set_joint_if_present(joints.middle_prox, rotations, "XRHand_MiddleProximal")
            self._set_joint_if_present(joints.middle_mid, rotations, "XRHand_MiddleIntermediate")
            self._set_joint_if_present(joints.middle_dist, rotations, "XRHand_MiddleDistal")
        if bool(finger_playback.get("ring", True)):
            self._set_joint_if_present(joints.ring_prox, rotations, "XRHand_RingProximal")
            self._set_joint_if_present(joints.ring_mid, rotations, "XRHand_RingIntermediate")
            self._set_joint_if_present(joints.ring_dist, rotations, "XRHand_RingDistal")
        self._apply_thumb(joints, rotations)
        self._apply_tracked_objects(frame)

    def _set_joint_if_present(
        self,
        joint_handle: int | None,
        rotations: dict[str, list[float]],
        key: str,
    ) -> None:
        rotation = rotations.get(key)
        if joint_handle is None or rotation is None or self.sim is None:
            return
        gain = self._joint_gain(key)
        self.sim.setJointPosition(joint_handle, quat_to_single_axis_joint(rotation) * gain)

    def _apply_thumb(self, joints: JointHandles, rotations: dict[str, list[float]]) -> None:
        if self.sim is None:
            return
        if not bool(self.config.get("thumbPlayback", {}).get("enabled", True)):
            return

        thumb_coupling = self.config.get("thumbCoupling", {})
        base_min = math.radians(float(thumb_coupling.get("baseMinDegrees", -6.0)))
        base_max = math.radians(float(thumb_coupling.get("baseMaxDegrees", 66.0)))
        meta_solve_blend = float(thumb_coupling.get("metaSolveBlend", 0.3))
        base_solve_blend = float(thumb_coupling.get("baseSolveBlend", 0.75))

        proximal = rotations.get("XRHand_ThumbProximal")
        scaled_proximal = None
        if proximal is not None:
            proximal_angle = quat_to_single_axis_joint(proximal)
            gain = self._joint_gain("XRHand_ThumbProximal", 1.0)
            scaled_proximal = proximal_angle * gain
            if joints.thumb_prox is not None:
                self.sim.setJointPosition(joints.thumb_prox, scaled_proximal)

        distal = rotations.get("XRHand_ThumbDistal")
        scaled_distal = None
        if distal is not None:
            distal_angle = quat_to_single_axis_joint(distal)
            gain = self._joint_gain("XRHand_ThumbDistal", 1.0)
            scaled_distal = distal_angle * gain
            if joints.thumb_dist is not None:
                self.sim.setJointPosition(joints.thumb_dist, scaled_distal)

        metacarpal = rotations.get("XRHand_ThumbMetacarpal")
        if metacarpal is not None and joints.thumb_meta is not None and joints.thumb_base is not None:
            solved_proximal = 0.0 if scaled_proximal is None else scaled_proximal
            solved_meta_angle, solved_base_angle = solve_thumb_root_angles(
                metacarpal,
                proximal,
                solved_proximal,
                self.thumb_calibration,
            )
            heuristic_meta_angle, heuristic_base_angle = thumb_angles_from_metacarpal_xz(
                metacarpal,
                self.thumb_heuristic_calibration,
            )
            meta_angle = (
                (1.0 - meta_solve_blend) * heuristic_meta_angle
                + meta_solve_blend * solved_meta_angle
            )
            base_angle = (
                (1.0 - base_solve_blend) * heuristic_base_angle
                + base_solve_blend * solved_base_angle
            )
            scaled_meta = meta_angle
            scaled_base = base_angle
            scaled_base = min(max(scaled_base, base_min), base_max)
            self.sim.setJointPosition(joints.thumb_meta, scaled_meta)
            self.sim.setJointPosition(joints.thumb_base, scaled_base)

    def _apply_tracked_objects(self, frame: HandFrame) -> None:
        if self.sim is None or self.handles is None:
            return

        transform = self.config.get("transform", {})
        smoothing = self.config.get("trackedObjectSmoothing", {})
        rotation_outlier = self.config.get("trackedObjectRotationOutlier", {})
        rotation_retarget = self.config.get("trackedObjectRotationRetargeting", {})
        position_outlier = self.config.get("trackedObjectPositionOutlier", {})
        playback = self.config.get("trackedObjectPlayback", {})
        reference_frames = self.config.get("trackedObjectReferenceFrames", {})
        local_transform = self.config.get("trackedObjectLocalTransform", {})
        default_position_alpha = float(
            smoothing.get("defaultPositionAlpha", smoothing.get("defaultAlpha", 1.0))
        )
        default_rotation_alpha = float(
            smoothing.get("defaultRotationAlpha", smoothing.get("defaultAlpha", 1.0))
        )
        per_object_position_alpha = smoothing.get(
            "positionAlphaByField",
            smoothing.get("alphaByField", {}),
        )
        per_object_rotation_alpha = smoothing.get(
            "rotationAlphaByField",
            smoothing.get("alphaByField", {}),
        )
        default_max_rotation_jump = float(rotation_outlier.get("defaultMaxDegrees", 180.0))
        default_hold_frames = int(rotation_outlier.get("defaultHoldFrames", 0))
        max_rotation_by_field = rotation_outlier.get("maxDegreesByField", {})
        hold_frames_by_field = rotation_outlier.get("holdFramesByField", {})
        default_retarget_enabled = bool(rotation_retarget.get("defaultEnabled", False))
        enabled_retarget_by_field = rotation_retarget.get("enabledByField", {})
        default_retarget_threshold = float(rotation_retarget.get("defaultMaxDegrees", 180.0))
        retarget_threshold_by_field = rotation_retarget.get("maxDegreesByField", {})
        default_max_position_jump = float(position_outlier.get("defaultMaxMeters", 1e9))
        default_position_hold_frames = int(position_outlier.get("defaultHoldFrames", 0))
        max_position_by_field = position_outlier.get("maxMetersByField", {})
        position_hold_by_field = position_outlier.get("holdFramesByField", {})
        default_apply_rotation = bool(playback.get("defaultApplyRotation", True))
        apply_rotation_by_field = playback.get("applyRotationByField", {})
        default_rotation_mode = str(playback.get("defaultRotationMode", "full"))
        rotation_mode_by_field = playback.get("rotationModeByField", {})
        for field_name, handle in self.handles.tracked_objects.items():
            pose = frame.tracked_objects.get(field_name)
            if pose is None:
                continue
            reference_frame = str(reference_frames.get(field_name, "world"))
            if reference_frame == "rootObject":
                position = remap_vector(
                    pose.position,
                    axes=local_transform.get("positionAxes", ["x", "y", "z"]),
                    signs=local_transform.get("positionSigns", [1.0, 1.0, 1.0]),
                    offset=local_transform.get("positionOffset", [0.0, 0.0, 0.0]),
                )
                rotation = remap_quaternion(
                    pose.rotation,
                    axes=local_transform.get("rotationAxes", ["x", "y", "z"]),
                    signs=local_transform.get("rotationSigns", [1.0, 1.0, 1.0]),
                )
                relative_to = self.handles.root_object
            else:
                position = remap_vector(
                    pose.position,
                    axes=transform.get("positionAxes", ["z", "x", "y"]),
                    signs=transform.get("positionSigns", [1.0, 1.0, 1.0]),
                    offset=transform.get("positionOffset", [0.0, 0.0, 0.0]),
                )
                rotation = remap_quaternion(
                    pose.rotation,
                    axes=transform.get("rotationAxes", ["z", "x", "y"]),
                    signs=transform.get("rotationSigns", [1.0, 1.0, 1.0]),
                )
                relative_to = -1
            position_alpha = float(
                per_object_position_alpha.get(field_name, default_position_alpha)
            )
            rotation_alpha = float(
                per_object_rotation_alpha.get(field_name, default_rotation_alpha)
            )
            apply_rotation = bool(
                apply_rotation_by_field.get(field_name, default_apply_rotation)
            )
            rotation_mode = str(
                rotation_mode_by_field.get(field_name, default_rotation_mode)
            ).lower()
            filter_state = self.tracked_object_filters.setdefault(field_name, FilterState())
            max_position_jump = float(
                max_position_by_field.get(field_name, default_max_position_jump)
            )
            position_hold_frames = int(
                position_hold_by_field.get(field_name, default_position_hold_frames)
            )
            if (
                filter_state.position is not None
                and max_position_jump < 1e8
                and position_hold_frames > 0
            ):
                position_delta = math.sqrt(
                    sum((position[i] - filter_state.position[i]) ** 2 for i in range(3))
                )
                if position_delta > max_position_jump:
                    filter_state.held_position_frames += 1
                    if filter_state.held_position_frames < position_hold_frames:
                        position = filter_state.position
                    else:
                        filter_state.held_position_frames = 0
                else:
                    filter_state.held_position_frames = 0
            filter_state.position = lerp_vector(
                filter_state.position,
                position,
                position_alpha,
            )
            self.sim.setObjectPosition(handle, relative_to, filter_state.position)
            if apply_rotation:
                if rotation_mode == "yaw_only":
                    rotation = quaternion_to_yaw_only(rotation)
                elif rotation_mode == "cube_symmetry":
                    rotation = nearest_cube_equivalent_rotation(
                        filter_state.quaternion,
                        rotation,
                    )
                if filter_state.rotation_correction is None:
                    filter_state.rotation_correction = [0.0, 0.0, 0.0, 1.0]
                retarget_enabled = bool(
                    enabled_retarget_by_field.get(field_name, default_retarget_enabled)
                )
                retarget_threshold = float(
                    retarget_threshold_by_field.get(field_name, default_retarget_threshold)
                )
                corrected_rotation = normalize_quaternion(
                    quaternion_multiply(filter_state.rotation_correction, rotation)
                )
                corrected_rotation = align_quaternion_hemisphere(
                    filter_state.quaternion,
                    corrected_rotation,
                )
                if (
                    retarget_enabled
                    and filter_state.quaternion is not None
                    and retarget_threshold < 180.0
                ):
                    corrected_delta = self._quaternion_angle_degrees(
                        filter_state.quaternion,
                        corrected_rotation,
                    )
                    if corrected_delta > retarget_threshold:
                        filter_state.rotation_correction = normalize_quaternion(
                            quaternion_multiply(
                                filter_state.quaternion,
                                quaternion_inverse(rotation),
                            )
                        )
                        corrected_rotation = normalize_quaternion(
                            quaternion_multiply(filter_state.rotation_correction, rotation)
                        )
                        corrected_rotation = align_quaternion_hemisphere(
                            filter_state.quaternion,
                            corrected_rotation,
                        )
                rotation = corrected_rotation
                max_rotation_jump = float(
                    max_rotation_by_field.get(field_name, default_max_rotation_jump)
                )
                hold_frames = int(hold_frames_by_field.get(field_name, default_hold_frames))
                if (
                    filter_state.quaternion is not None
                    and max_rotation_jump < 180.0
                    and hold_frames > 0
                ):
                    rotation_delta = self._quaternion_angle_degrees(
                        filter_state.quaternion,
                        rotation,
                    )
                    if rotation_delta > max_rotation_jump:
                        filter_state.held_rotation_frames += 1
                        if filter_state.held_rotation_frames < hold_frames:
                            rotation = filter_state.quaternion
                        else:
                            filter_state.held_rotation_frames = 0
                    else:
                        filter_state.held_rotation_frames = 0
                filter_state.quaternion = normalize_quaternion(
                    lerp_vector(filter_state.quaternion, rotation, rotation_alpha)
                )
                self.sim.setObjectQuaternion(handle, relative_to, filter_state.quaternion)

    def _quaternion_angle_degrees(self, left: list[float], right: list[float]) -> float:
        dot = sum(left[i] * right[i] for i in range(4))
        dot = max(-1.0, min(1.0, abs(dot)))
        return math.degrees(2.0 * math.acos(dot))

    def playback(
        self,
        frames: list[HandFrame],
        *,
        use_recorded_timing: bool,
        speed: float,
        start_simulation: bool,
        stepping: bool,
    ) -> None:
        if self.sim is None or self._client is None:
            raise RuntimeError("Remote API is not connected.")

        if stepping:
            self._client.setStepping(True)
            print("Stepping mode enabled.")

        started_here = False
        if start_simulation:
            state = self.sim.getSimulationState()
            if state == self.sim.simulation_stopped:
                self.sim.startSimulation()
                started_here = True
                print("Simulation started.")

        previous_timestamp = None
        try:
            for index, frame in enumerate(frames, start=1):
                self.apply_frame(frame)

                if index == 1:
                    print(f"Applied first frame at t={frame.timestamp:.3f}s.")
                elif index % 300 == 0 or index == len(frames):
                    print(f"Applied frame {index}/{len(frames)}.")

                if stepping:
                    self._client.step()
                    continue

                if use_recorded_timing and previous_timestamp is not None:
                    delay = max((frame.timestamp - previous_timestamp) / max(speed, 1e-6), 0.0)
                    if delay > 0:
                        time.sleep(delay)
                previous_timestamp = frame.timestamp
        finally:
            if started_here:
                self.sim.stopSimulation()
                print("Simulation stopped.")


def load_scene_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
